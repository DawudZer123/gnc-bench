"""Capteurs simulés. Chaque capteur a sa fréquence, un bruit blanc gaussien et parfois un biais."""

import numpy as np

from .dynamics import OMEGA, POS, QUAT, VEL
from .math3d import quat_to_rotmat


class Sensor:
    """Interface commune : `measure` renvoie un vecteur de mesures à partir de l'état vrai."""

    name = "capteur"

    def __init__(self, rng, rate):
        self.rng = rng      # générateur aléatoire (reproductible avec une graine)
        self.rate = rate    # Hz

    def measure(self, t, x, dx, params):
        raise NotImplementedError


class IMU(Sensor):
    """Accéléromètre + gyroscope, dans le repère corps. Mesure : [ax, ay, az, gx, gy, gz]."""

    name = "imu"

    def __init__(self, rng, rate=250.0, accel_noise=0.05, gyro_noise=0.005,
                 accel_bias=0.05, gyro_bias=0.01):
        super().__init__(rng, rate)
        self.accel_noise = accel_noise  # m/s²
        self.gyro_noise = gyro_noise    # rad/s
        # Biais tiré une fois au démarrage, puis constant (comme une vraie IMU mal calibrée).
        self.accel_bias = rng.normal(0.0, accel_bias, 3)
        self.gyro_bias = rng.normal(0.0, gyro_bias, 3)

    def measure(self, t, x, dx, params):
        R = quat_to_rotmat(x[QUAT])
        # Un accéléromètre mesure la force spécifique : accélération moins gravité.
        # Au repos, il lit donc +g vers le haut, pas zéro.
        specific_force = R.T @ (dx[VEL] + np.array([0.0, 0.0, params.gravity]))
        accel = specific_force + self.accel_bias + self.rng.normal(0.0, self.accel_noise, 3)
        gyro = x[OMEGA] + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise, 3)
        return np.concatenate([accel, gyro])


class GPS(Sensor):
    """Position et vitesse dans le repère monde. Mesure : [px, py, pz, vx, vy, vz]."""

    name = "gps"

    def __init__(self, rng, rate=10.0, pos_noise=(0.5, 0.5, 1.0), vel_noise=0.1):
        super().__init__(rng, rate)
        self.pos_noise = np.asarray(pos_noise)  # m (la verticale est moins précise)
        self.vel_noise = vel_noise              # m/s

    def measure(self, t, x, dx, params):
        pos = x[POS] + self.rng.normal(0.0, 1.0, 3) * self.pos_noise
        vel = x[VEL] + self.rng.normal(0.0, self.vel_noise, 3)
        return np.concatenate([pos, vel])


class Barometer(Sensor):
    """Altitude seule, à partir de la pression. Mesure : [altitude]."""

    name = "baro"

    def __init__(self, rng, rate=50.0, alt_noise=0.1):
        super().__init__(rng, rate)
        self.alt_noise = alt_noise  # m

    def measure(self, t, x, dx, params):
        alt = x[POS][2] + self.rng.normal(0.0, self.alt_noise)
        return np.array([alt])


class OpticalFlow(Sensor):
    """Caméra pointée vers le sol + télémètre : mesure la vitesse horizontale dans le repère corps.

    Un vrai capteur mesure le défilement des pixels (rad/s) ; multiplié par la hauteur
    (télémètre), on obtient une vitesse. On simule directement ce résultat.
    Mesure : [vx_corps, vy_corps].
    """

    name = "flow"

    def __init__(self, rng, rate=50.0, vel_noise=0.05):
        super().__init__(rng, rate)
        self.vel_noise = vel_noise  # m/s

    def measure(self, t, x, dx, params):
        R = quat_to_rotmat(x[QUAT])
        v_body = R.T @ x[VEL]
        return v_body[:2] + self.rng.normal(0.0, self.vel_noise, 2)
