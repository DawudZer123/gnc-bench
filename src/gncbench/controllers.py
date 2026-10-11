"""Controllers. Every controller tested by the bench inherits from Controller."""

import numpy as np

from .dynamics import OMEGA, POS, QUAT, VEL
from .math3d import cross3, quat_to_rotmat, vee


class Controller:
    """Common interface: takes the (estimated) state and returns the 4 rotor thrusts.

    If the flight has a navigation filter, the simulation hands it over with `attach`,
    so a controller can read more than position and velocity (like a real autopilot
    reading its EKF). Controllers that do not need it simply ignore it.
    """

    estimator = None

    def attach(self, estimator):
        self.estimator = estimator

    def reset(self):
        pass

    def update(self, t, state, reference):
        raise NotImplementedError


class HoverController(Controller):
    """Cascaded controller: position PID -> geometric attitude control (Lee et al., 2010).

    reference: dict with 'position' (3,) and, optionally, 'velocity' (3,), 'acceleration' (3,)
               and 'yaw' (rad). Reference velocity and acceleration are used as feedforward:
               without them, the drone follows a trajectory with a lag.
    """

    def __init__(self, params, dt, kp=6.0, kd=4.5, ki=0.3, k_att=0.6, k_rate=0.12,
                 i_limit=3.0):
        self.p = params
        self.dt = dt
        self.kp, self.kd, self.ki = kp, kd, ki
        self.k_att, self.k_rate = k_att, k_rate
        self.i_limit = i_limit  # m/s², maximum action of the integral term
        self.mixer_inv = np.linalg.inv(params.mixer())
        self.reset()

    def reset(self):
        self.int_err = np.zeros(3)

    def desired_acceleration(self, state, reference):
        """Outer loop: the acceleration the thrust must produce (gravity included)."""
        p, v = state[POS], state[VEL]
        err = reference["position"] - p
        # Anti-windup: only integrate close to the target, otherwise the integral
        # built up during take-off causes a long overshoot.
        if np.linalg.norm(err) < 0.3:
            lim = self.i_limit / self.ki if self.ki > 0 else 0.0
            self.int_err = np.clip(self.int_err + err * self.dt, -lim, lim)
        v_ref = reference.get("velocity", np.zeros(3))
        a_ref = reference.get("acceleration", np.zeros(3))
        return (self.kp * err + self.kd * (v_ref - v) + self.ki * self.int_err
                + a_ref + np.array([0, 0, self.p.gravity]))

    def update(self, t, state, reference):
        q, w = state[QUAT], state[OMEGA]

        # 1. Outer loop: desired acceleration.
        a_des = self.desired_acceleration(state, reference)

        # 2. Desired attitude: the body z axis points along a_des.
        R = quat_to_rotmat(q)
        z_d = a_des / np.linalg.norm(a_des)
        yaw = reference.get("yaw", 0.0)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_d = cross3(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        R_d = np.column_stack([cross3(y_d, z_d), y_d, z_d])

        # 3. Total thrust and torques (geometric control on SO(3)).
        thrust = self.p.mass * a_des @ R[:, 2]
        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        J = self.p.inertia
        torque = -self.k_att * e_R - self.k_rate * w + cross3(w, J @ w)

        # 4. Allocation to the 4 rotors.
        rotors = self.mixer_inv @ np.concatenate(([thrust], torque))
        return np.clip(rotors, 0.0, self.p.max_thrust)


class WindRejectionController(HoverController):
    """Same controller, plus disturbance rejection: it cancels the wind as soon as it is felt.

    The navigation filter estimates the external acceleration (wind + drag) from the
    accelerometer. Instead of waiting for a position error to build up, the controller
    subtracts it right away from the acceleration it asks the thrust to produce.
    """

    def __init__(self, params, dt, rejection=1.0, **kwargs):
        super().__init__(params, dt, **kwargs)
        self.rejection = rejection   # 0 = plain HoverController, 1 = full compensation

    def desired_acceleration(self, state, reference):
        a_des = super().desired_acceleration(state, reference)
        if self.estimator is not None:
            a_des = a_des - self.rejection * self.estimator.disturbance
        return a_des
