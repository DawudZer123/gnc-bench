"""Simulated sensors. Each sensor has its own rate, Gaussian white noise and sometimes a bias."""

import numpy as np

from .dynamics import OMEGA, POS, QUAT, VEL
from .math3d import quat_to_rotmat


class Sensor:
    """Common interface: `measure` returns a measurement vector from the true state."""

    name = "sensor"

    def __init__(self, rng, rate):
        self.rng = rng      # random generator (reproducible with a seed)
        self.rate = rate    # Hz

    def measure(self, t, x, dx, params):
        raise NotImplementedError


class IMU(Sensor):
    """Accelerometer + gyroscope, in the body frame. Measurement: [ax, ay, az, gx, gy, gz]."""

    name = "imu"

    def __init__(self, rng, rate=250.0, accel_noise=0.05, gyro_noise=0.005,
                 accel_bias=0.05, gyro_bias=0.01):
        super().__init__(rng, rate)
        self.accel_noise = accel_noise  # m/s²
        self.gyro_noise = gyro_noise    # rad/s
        # Bias drawn once at start-up, then constant (like a real, poorly calibrated IMU).
        self.accel_bias = rng.normal(0.0, accel_bias, 3)
        self.gyro_bias = rng.normal(0.0, gyro_bias, 3)

    def measure(self, t, x, dx, params):
        R = quat_to_rotmat(x[QUAT])
        # An accelerometer measures specific force: acceleration minus gravity.
        # At rest it therefore reads +g upward, not zero.
        specific_force = R.T @ (dx[VEL] + np.array([0.0, 0.0, params.gravity]))
        accel = specific_force + self.accel_bias + self.rng.normal(0.0, self.accel_noise, 3)
        gyro = x[OMEGA] + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise, 3)
        return np.concatenate([accel, gyro])


class GPS(Sensor):
    """Position and velocity in the world frame. Measurement: [px, py, pz, vx, vy, vz]."""

    name = "gps"

    def __init__(self, rng, rate=10.0, pos_noise=(0.5, 0.5, 1.0), vel_noise=0.1):
        super().__init__(rng, rate)
        self.pos_noise = np.asarray(pos_noise)  # m (vertical is less accurate)
        self.vel_noise = vel_noise              # m/s

    def measure(self, t, x, dx, params):
        pos = x[POS] + self.rng.normal(0.0, 1.0, 3) * self.pos_noise
        vel = x[VEL] + self.rng.normal(0.0, self.vel_noise, 3)
        return np.concatenate([pos, vel])


class Barometer(Sensor):
    """Altitude only, from air pressure. Measurement: [altitude]."""

    name = "baro"

    def __init__(self, rng, rate=50.0, alt_noise=0.1):
        super().__init__(rng, rate)
        self.alt_noise = alt_noise  # m

    def measure(self, t, x, dx, params):
        alt = x[POS][2] + self.rng.normal(0.0, self.alt_noise)
        return np.array([alt])


class OpticalFlow(Sensor):
    """Downward-facing camera + rangefinder: measures horizontal velocity in the body frame.

    A real sensor measures how fast pixels move (rad/s); multiplied by the height
    (rangefinder), this gives a velocity. We simulate that result directly.
    Measurement: [vx_body, vy_body].
    """

    name = "flow"

    def __init__(self, rng, rate=50.0, vel_noise=0.05):
        super().__init__(rng, rate)
        self.vel_noise = vel_noise  # m/s

    def measure(self, t, x, dx, params):
        R = quat_to_rotmat(x[QUAT])
        v_body = R.T @ x[VEL]
        return v_body[:2] + self.rng.normal(0.0, self.vel_noise, 2)
