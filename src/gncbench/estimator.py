"""Filtre de Kalman de navigation : fusionne IMU, GPS et baromètre.

État estimé (9) : s = [p (3), v (3), b (3)]
  p : position, repère monde (m)
  v : vitesse, repère monde (m/s)
  b : biais de l'accéléromètre, repère corps (m/s²)

Capteur optionnel : flux optique (vitesse horizontale dans le repère corps), qui
permet de garder une bonne vitesse, donc une position qui dérive lentement, sans GPS.

Hypothèse : l'attitude est connue (fournie par un filtre d'attitude séparé).
Avec cette hypothèse, le modèle est linéaire : c'est un filtre de Kalman classique.

Cycle :
  - prédiction à chaque mesure IMU (250 Hz) : on intègre l'accélération mesurée ;
  - correction à chaque mesure GPS (10 Hz) ou baromètre (50 Hz).
"""

import numpy as np

from .math3d import quat_to_rotmat

P_, V_, B_ = slice(0, 3), slice(3, 6), slice(6, 9)
I3 = np.eye(3)


class NavigationFilter:
    def __init__(self, params, p0, v0=(0.0, 0.0, 0.0), accel_noise=0.05,
                 bias_walk=1e-3, gps_noise=(0.5, 0.5, 1.0, 0.1, 0.1, 0.1), baro_noise=0.1, flow_noise=0.05):
        self.g = np.array([0.0, 0.0, params.gravity])
        self.s = np.concatenate([p0, v0, np.zeros(3)])
        # Incertitude initiale : 1 m en position, 0,5 m/s en vitesse, 0,1 m/s² en biais.
        self.P = np.diag([1.0] * 3 + [0.25] * 3 + [0.01] * 3)
        self.accel_noise = accel_noise
        self.bias_walk = bias_walk
        self.R_gps = np.diag(np.square(gps_noise))
        self.R_baro = np.array([[baro_noise**2]])
        self.R_flow = np.eye(2) * flow_noise**2

    # ----- Prédiction --------------------------------------------------------------
    def predict(self, accel_meas, quat, dt):
        R = quat_to_rotmat(quat)
        p, v, b = self.s[P_], self.s[V_], self.s[B_]
        a = R @ (accel_meas - b) - self.g          # accélération dans le repère monde

        self.s[P_] = p + v * dt + 0.5 * a * dt**2
        self.s[V_] = v + a * dt

        # Jacobienne F = d(état suivant) / d(état actuel)
        F = np.eye(9)
        F[P_, V_] = I3 * dt
        F[P_, B_] = -0.5 * R * dt**2
        F[V_, B_] = -R * dt

        # Bruit de processus : le bruit de l'accéléromètre entre dans la vitesse,
        # le biais dérive lentement (marche aléatoire).
        Q = np.zeros((9, 9))
        Q[V_, V_] = I3 * (self.accel_noise * dt) ** 2
        Q[B_, B_] = I3 * (self.bias_walk**2) * dt
        self.P = F @ self.P @ F.T + Q

    # ----- Correction --------------------------------------------------------------
    def _correct(self, z, H, R_meas):
        y = z - H @ self.s                         # innovation : mesure - prédiction
        S = H @ self.P @ H.T + R_meas              # incertitude de l'innovation
        K = self.P @ H.T @ np.linalg.inv(S)        # gain de Kalman
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
        # Mesure = 2 premières composantes de R^T v : elle dépend de l'attitude.
        R = quat_to_rotmat(quat)
        H = np.zeros((2, 9))
        H[:, V_] = R.T[:2, :]
        self._correct(z, H, self.R_flow)

    # ----- Interface avec la simulation ----------------------------------------------
    def process(self, name, z, quat, dt):
        """Traite une mesure. Une mesure NaN (capteur en panne) est ignorée."""
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
        """Écart-type estimé de la position (m) : ce que le filtre pense de sa propre erreur."""
        return np.sqrt(np.diag(self.P)[P_])
