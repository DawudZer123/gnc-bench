"""Contrôleurs. Tout contrôleur testé par le banc hérite de Controller."""

import numpy as np

from .dynamics import OMEGA, POS, QUAT, VEL
from .math3d import quat_to_rotmat, vee


class Controller:
    """Interface commune : reçoit l'état (estimé) et renvoie les 4 poussées des rotors."""

    def reset(self):
        pass

    def update(self, t, state, reference):
        raise NotImplementedError


class HoverController(Controller):
    """Contrôleur en cascade : PID de position -> contrôle géométrique d'attitude (Lee et al., 2010).

    reference : dict avec 'position' (3,) et, en option, 'yaw' (rad).
    """

    def __init__(self, params, dt, kp=6.0, kd=4.5, ki=0.3, k_att=0.6, k_rate=0.12,
                 i_limit=3.0):
        self.p = params
        self.dt = dt
        self.kp, self.kd, self.ki = kp, kd, ki
        self.k_att, self.k_rate = k_att, k_rate
        self.i_limit = i_limit  # m/s², action maximale de l'intégrale
        self.mixer_inv = np.linalg.inv(params.mixer())
        self.reset()

    def reset(self):
        self.int_err = np.zeros(3)

    def update(self, t, state, reference):
        p, v, q, w = state[POS], state[VEL], state[QUAT], state[OMEGA]
        m, g = self.p.mass, self.p.gravity

        # 1. Boucle externe : accélération désirée.
        err = reference["position"] - p
        # Anti-emballement : on n'intègre que près de la cible, sinon l'intégrale
        # accumulée pendant le décollage provoque un long dépassement.
        if np.linalg.norm(err) < 0.3:
            lim = self.i_limit / self.ki if self.ki > 0 else 0.0
            self.int_err = np.clip(self.int_err + err * self.dt, -lim, lim)
        a_des = self.kp * err - self.kd * v + self.ki * self.int_err + np.array([0, 0, g])

        # 2. Attitude désirée : l'axe z du corps pointe selon a_des.
        R = quat_to_rotmat(q)
        z_d = a_des / np.linalg.norm(a_des)
        yaw = reference.get("yaw", 0.0)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_d = np.cross(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        R_d = np.column_stack([np.cross(y_d, z_d), y_d, z_d])

        # 3. Poussée totale et couples (contrôle géométrique sur SO(3)).
        thrust = m * a_des @ R[:, 2]
        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        J = self.p.inertia
        torque = -self.k_att * e_R - self.k_rate * w + np.cross(w, J @ w)

        # 4. Allocation vers les 4 rotors.
        rotors = self.mixer_inv @ np.concatenate(([thrust], torque))
        return np.clip(rotors, 0.0, self.p.max_thrust)
