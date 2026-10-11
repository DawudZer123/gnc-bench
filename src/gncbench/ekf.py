"""Full navigation EKF: estimates the attitude too, instead of borrowing the true one.

This is how a real autopilot (PX4, ArduPilot) navigates. It is an "error-state" EKF:
  - the nominal state is integrated from the IMU:  p, v, q, accelerometer bias, gyro bias;
  - the filter tracks the covariance of a small ERROR around it (15 numbers):
        δp (3), δv (3), δθ (3, small rotation in the body frame), δb_a (3), δb_g (3);
  - after each correction, the error is folded back into the nominal state and reset to 0.
Working on the small error keeps the quaternion a unit quaternion and the maths linear.

Sensors: IMU (prediction), GPS, barometer, optical flow, magnetometer (heading).
Roll and pitch become observable through gravity: a tilt error makes the filter
"see" a horizontal acceleration that GPS or optical flow then contradict.
Yaw needs the magnetometer (or strong horizontal accelerations).
"""

import numpy as np

from .math3d import hat, quat_mult, quat_to_euler, quat_to_rotmat

# Slices in the 15-element error state.
EP, EV, ET, EBA, EBG = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15))
N = 15
I3 = np.eye(3)


def small_rotation(dtheta):
    """Quaternion of a rotation vector (axis × angle, rad)."""
    angle = np.linalg.norm(dtheta)
    if angle < 1e-12:
        return np.array([1.0, *(0.5 * dtheta)])
    axis = dtheta / angle
    return np.concatenate(([np.cos(angle / 2)], np.sin(angle / 2) * axis))


def wrap(angle):
    """Angle in [-π, π)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class NavigationEKF:
    def __init__(self, params, p0, v0=(0.0, 0.0, 0.0), q0=(1.0, 0.0, 0.0, 0.0),
                 accel_noise=0.05, gyro_noise=0.005, accel_bias_walk=1e-3, gyro_bias_walk=1e-4,
                 gps_noise=(0.5, 0.5, 1.0, 0.1, 0.1, 0.1), baro_noise=0.1, flow_noise=0.05,
                 mag_noise=0.02, disturbance_tau=0.05):
        self.g = np.array([0.0, 0.0, params.gravity])
        self.p = np.array(p0, dtype=float)
        self.v = np.array(v0, dtype=float)
        self.q = np.array(q0, dtype=float)
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        # Initial uncertainty: 1 m, 0.5 m/s, 5° in attitude, 0.1 m/s², 0.02 rad/s.
        self.P = np.diag([1.0] * 3 + [0.25] * 3 + [np.radians(5) ** 2] * 3
                         + [0.01] * 3 + [0.02**2] * 3)
        self.accel_noise, self.gyro_noise = accel_noise, gyro_noise
        self.accel_bias_walk, self.gyro_bias_walk = accel_bias_walk, gyro_bias_walk
        self.R_gps = np.diag(np.square(gps_noise))
        self.R_baro = np.array([[baro_noise**2]])
        self.R_flow = np.eye(2) * flow_noise**2
        self.R_mag = np.array([[mag_noise**2]])
        self.omega = np.zeros(3)                 # last bias-corrected gyro reading
        self.disturbance_tau = disturbance_tau
        self.disturbance = np.zeros(3)

    # ----- Prediction ----------------------------------------------------------------
    def predict(self, imu, dt):
        f = imu[:3] - self.ba                    # specific force, body frame
        w = imu[3:] - self.bg                    # angular rate, body frame
        R = quat_to_rotmat(self.q)
        a = R @ f - self.g

        self.p = self.p + self.v * dt + 0.5 * a * dt**2
        self.v = self.v + a * dt
        self.q = quat_mult(self.q, small_rotation(w * dt))
        self.q /= np.linalg.norm(self.q)
        self.omega = w

        # Error dynamics, discretized: F = I + A dt.
        F = np.eye(N)
        F[EP, EV] = I3 * dt
        F[EV, ET] = -R @ hat(f) * dt             # a tilt error rotates the measured force
        F[EV, EBA] = -R * dt
        F[EP, ET] = 0.5 * F[EV, ET] * dt
        F[EP, EBA] = 0.5 * F[EV, EBA] * dt
        F[ET, ET] = I3 - hat(w) * dt
        F[ET, EBG] = -I3 * dt

        Q = np.zeros((N, N))
        Q[EV, EV] = I3 * (self.accel_noise * dt) ** 2
        Q[ET, ET] = I3 * (self.gyro_noise * dt) ** 2
        Q[EBA, EBA] = I3 * self.accel_bias_walk**2 * dt
        Q[EBG, EBG] = I3 * self.gyro_bias_walk**2 * dt
        self.P = F @ self.P @ F.T + Q

        # Wind disturbance, as in NavigationFilter (horizontal part only).
        d = R @ np.array([f[0], f[1], 0.0])
        d[2] = 0.0
        self.disturbance += dt / (self.disturbance_tau + dt) * (d - self.disturbance)

    # ----- Correction ----------------------------------------------------------------
    def _correct(self, y, H, R_meas):
        """y = innovation (measurement - prediction), H = d(measurement) / d(error state)."""
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        I_KH = np.eye(N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T   # Joseph form: stays symmetric
        # Fold the estimated error into the nominal state, then the error is 0 again.
        self.p += dx[EP]
        self.v += dx[EV]
        self.q = quat_mult(self.q, small_rotation(dx[ET]))
        self.q /= np.linalg.norm(self.q)
        self.ba += dx[EBA]
        self.bg += dx[EBG]

    def update_gps(self, z):
        H = np.zeros((6, N))
        H[0:3, EP] = I3
        H[3:6, EV] = I3
        self._correct(z - np.concatenate([self.p, self.v]), H, self.R_gps)

    def update_baro(self, z):
        H = np.zeros((1, N))
        H[0, 2] = 1.0
        self._correct(z - self.p[2:3], H, self.R_baro)

    def flow_prediction(self):
        """What the optical flow should read: the first 2 components of R^T v."""
        return (quat_to_rotmat(self.q).T @ self.v)[:2]

    def flow_jacobian(self):
        R = quat_to_rotmat(self.q)
        H = np.zeros((2, N))
        H[:, EV] = R.T[:2, :]
        H[:, ET] = hat(R.T @ self.v)[:2, :]      # R^T v changes when the attitude is wrong
        return H

    def update_flow(self, z):
        self._correct(z - self.flow_prediction(), self.flow_jacobian(), self.R_flow)

    def update_mag(self, z):
        H = np.zeros((1, N))
        H[0, ET] = quat_to_rotmat(self.q)[2, :]  # a body-frame rotation, seen around world z
        self._correct(np.array([wrap(z[0] - self.yaw)]), H, self.R_mag)

    # ----- Interface with the simulation ---------------------------------------------
    def process(self, name, z, quat, dt):
        """Processes one measurement. `quat` (the TRUE attitude) is ignored: no cheating."""
        if np.isnan(z).any():
            return
        if name == "imu":
            self.predict(z, dt)
        elif name == "gps":
            self.update_gps(z)
        elif name == "baro":
            self.update_baro(z)
        elif name == "flow":
            self.update_flow(z)
        elif name == "mag":
            self.update_mag(z)

    @property
    def position(self):
        return self.p.copy()

    @property
    def velocity(self):
        return self.v.copy()

    @property
    def attitude(self):
        return self.q.copy()

    @property
    def angular_rate(self):
        return self.omega.copy()

    @property
    def yaw(self):
        return quat_to_euler(self.q)[2]

    @property
    def position_std(self):
        return np.sqrt(np.diag(self.P)[EP])

    @property
    def attitude_std(self):
        """Estimated attitude uncertainty (rad) around the body x, y, z axes."""
        return np.sqrt(np.diag(self.P)[ET])
