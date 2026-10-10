"""Paramètres physiques du quadricoptère (ordre de grandeur d'un drone de 1 kg)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QuadParams:
    mass: float = 1.0                # kg
    arm_length: float = 0.2          # m, centre -> rotor
    inertia: np.ndarray = field(
        default_factory=lambda: np.diag([0.01, 0.01, 0.02])
    )                                # kg·m²
    max_thrust: float = 8.0          # N par rotor (rapport poussée/poids ≈ 3,3)
    torque_coeff: float = 0.016      # m, couple de lacet = coeff × poussée
    drag_coeff: float = 0.1          # N·s/m, traînée linéaire simple
    gravity: float = 9.81            # m/s²
    motor_tau: float = 0.03          # s, constante de temps des moteurs

    def rotor_positions(self):
        """Configuration en X. Rotors : 0 avant-droit, 1 arrière-gauche, 2 avant-gauche, 3 arrière-droit."""
        d = self.arm_length / np.sqrt(2)
        return np.array([[d, -d, 0], [-d, d, 0], [d, d, 0], [-d, -d, 0]])

    def rotor_spins(self):
        """+1 = antihoraire vu de dessus, -1 = horaire."""
        return np.array([1, 1, -1, -1])

    def mixer(self):
        """Matrice M telle que [poussée totale, τx, τy, τz] = M @ poussées des rotors."""
        r = self.rotor_positions()
        s = self.rotor_spins()
        return np.vstack([
            np.ones(4),
            r[:, 1],                    # τx = y × T
            -r[:, 0],                   # τy = -x × T
            -s * self.torque_coeff,     # τz : réaction du rotor, opposée à sa rotation
        ])
