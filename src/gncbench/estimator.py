"""Navigation Kalman filter: fuses IMU, GPS and barometer.

Estimated state (9): s = [p (3), v (3), b (3)]
  p: position, world frame (m)
  v: velocity, world frame (m/s)
  b: accelerometer bias, body frame (m/s²)

Optional sensor: optical flow (horizontal velocity in the body frame), which keeps
the velocity accurate, so the position drifts slowly, without GPS.

Bonus output: `disturbance`, the external acceleration (wind + drag) felt by the drone.
Thrust always points along the body z axis, so whatever the accelerometer reads on the
body x and y axes can only come from aerodynamic forces: that is the wind, measured directly
(horizontal part only).

Assumption: the attitude is known (provided by a separate attitude filter).
With this assumption the model is linear: it is a standard Kalman filter.

Cycle:
  - prediction at each IMU measurement (250 Hz): integrate the measured acceleration;
  - correction at each GPS (10 Hz), barometer (50 Hz) or optical flow (50 Hz) measurement.
"""

import numpy as np

from .math3d import quat_to_rotmat

P_, V_, B_ = slice(0, 3), slice(3, 6), slice(6, 9)
I3 = np.eye(3)


class NavigationFilter:
    def __init__(self, params, p0, v0=(0.0, 0.0, 0.0), accel_noise=0.05,
                 bias_walk=1e-3, gps_noise=(0.5, 0.5, 1.0, 0.1, 0.1, 0.1), baro_noise=0.1, flow_noise=0.05,
                 disturbance_tau=0.05):
        self.g = np.array([0.0, 0.0, params.gravity])
        self.s = np.concatenate([p0, v0, np.zeros(3)])
        # Initial uncertainty: 1 m in position, 0.5 m/s in velocity, 0.1 m/s² in bias.
        self.P = np.diag([1.0] * 3 + [0.25] * 3 + [0.01] * 3)
        self.accel_noise = accel_noise
        self.bias_walk = bias_walk
        self.R_gps = np.diag(np.square(gps_noise))
        self.R_baro = np.array([[baro_noise**2]])
        self.R_flow = np.eye(2) * flow_noise**2
        self.disturbance_tau = disturbance_tau   # s, low-pass filter on the disturbance
        self.disturbance = np.zeros(3)           # m/s², world frame

    # ----- Prediction ----------------------------------------------------------------
    def predict(self, accel_meas, quat, dt):
        R = quat_to_rotmat(quat)
        p, v, b = self.s[P_], self.s[V_], self.s[B_]
        a = R @ (accel_meas - b) - self.g          # acceleration in the world frame

        self.s[P_] = p + v * dt + 0.5 * a * dt**2
        self.s[V_] = v + a * dt

        # Jacobian F = d(next state) / d(current state)
        F = np.eye(9)
        F[P_, V_] = I3 * dt
        F[P_, B_] = -0.5 * R * dt**2
        F[V_, B_] = -R * dt

        # Process noise: accelerometer noise enters the velocity,
        # the bias drifts slowly (random walk).
        Q = np.zeros((9, 9))
        Q[V_, V_] = I3 * (self.accel_noise * dt) ** 2
        Q[B_, B_] = I3 * (self.bias_walk**2) * dt
        self.P = F @ self.P @ F.T + Q

        # Disturbance: body x/y accelerometer reading (bias removed), back in the world frame,
        # smoothed by a first-order low-pass filter to remove the sensor noise.
        # Only the horizontal part is kept: when the drone tilts into the wind, the rotation
        # would otherwise leak a fake vertical component (vertical gusts are not compensated).
        f_body = accel_meas - b
        d = R @ np.array([f_body[0], f_body[1], 0.0])
        d[2] = 0.0
        alpha = dt / (self.disturbance_tau + dt)
        self.disturbance = self.disturbance + alpha * (d - self.disturbance)

    # ----- Correction ----------------------------------------------------------------
    def _correct(self, z, H, R_meas):
        y = z - H @ self.s                         # innovation: measurement - prediction
        S = H @ self.P @ H.T + R_meas              # innovation uncertainty
        K = self.P @ H.T @ np.linalg.inv(S)        # Kalman gain
        self.s = self.s + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P

    def update_gps(self, z):
        H = np.zeros((6, 9))
        H[0:3, P_] = I3
        H[3:6, V_] = I3
        self._correct(z, H, self.R_gps)

    def update_baro(self, z):
        H = np.zeros((1, 9))
        H[0, 2] = 1.0
        self._correct(z, H, self.R_baro)

    def update_flow(self, z, quat):
        # Measurement = first 2 components of R^T v: it depends on the attitude.
        R = quat_to_rotmat(quat)
        H = np.zeros((2, 9))
        H[:, V_] = R.T[:2, :]
        self._correct(z, H, self.R_flow)

    # ----- Interface with the simulation ---------------------------------------------
    def process(self, name, z, quat, dt):
        """Processes one measurement. A NaN measurement (failed sensor) is ignored."""
        if np.isnan(z).any():
            return
        if name == "imu":
            self.predict(z[:3], quat, dt)
        elif name == "gps":
            self.update_gps(z)
        elif name == "baro":
            self.update_baro(z)
        elif name == "flow":
            self.update_flow(z, quat)

    @property
    def position(self):
        return self.s[P_].copy()

    @property
    def velocity(self):
        return self.s[V_].copy()

    @property
    def position_std(self):
        """Estimated position standard deviation (m): what the filter thinks of its own error."""
        return np.sqrt(np.diag(self.P)[P_])
