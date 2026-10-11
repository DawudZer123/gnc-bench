"""Controllers. Every controller tested by the bench inherits from Controller."""

import numpy as np

from .dynamics import OMEGA, POS, QUAT, VEL
from .math3d import cross3, quat_to_rotmat, vee


class Controller:
    """Common interface: takes the (estimated) state and returns the 4 rotor thrusts."""

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

    def update(self, t, state, reference):
        p, v, q, w = state[POS], state[VEL], state[QUAT], state[OMEGA]
        m, g = self.p.mass, self.p.gravity

        # 1. Outer loop: desired acceleration.
        err = reference["position"] - p
        # Anti-windup: only integrate close to the target, otherwise the integral
        # built up during take-off causes a long overshoot.
        if np.linalg.norm(err) < 0.3:
            lim = self.i_limit / self.ki if self.ki > 0 else 0.0
            self.int_err = np.clip(self.int_err + err * self.dt, -lim, lim)
        v_ref = reference.get("velocity", np.zeros(3))
        a_ref = reference.get("acceleration", np.zeros(3))
        a_des = (self.kp * err + self.kd * (v_ref - v) + self.ki * self.int_err
                 + a_ref + np.array([0, 0, g]))

        # 2. Desired attitude: the body z axis points along a_des.
        R = quat_to_rotmat(q)
        z_d = a_des / np.linalg.norm(a_des)
        yaw = reference.get("yaw", 0.0)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_d = cross3(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        R_d = np.column_stack([cross3(y_d, z_d), y_d, z_d])

        # 3. Total thrust and torques (geometric control on SO(3)).
        thrust = m * a_des @ R[:, 2]
        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        J = self.p.inertia
        torque = -self.k_att * e_R - self.k_rate * w + cross3(w, J @ w)

        # 4. Allocation to the 4 rotors.
        rotors = self.mixer_inv @ np.concatenate(([thrust], torque))
        return np.clip(rotors, 0.0, self.p.max_thrust)
