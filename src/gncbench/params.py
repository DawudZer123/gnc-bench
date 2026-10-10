"""Physical parameters of the quadcopter (typical of a 1 kg drone)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QuadParams:
    mass: float = 1.0                # kg
    arm_length: float = 0.2          # m, center -> rotor
    inertia: np.ndarray = field(
        default_factory=lambda: np.diag([0.01, 0.01, 0.02])
    )                                # kg·m²
    max_thrust: float = 8.0          # N per rotor (thrust-to-weight ratio ≈ 3.3)
    torque_coeff: float = 0.016      # m, yaw torque = coeff × thrust
    drag_coeff: float = 0.3          # N·s/m, simple linear drag
    gravity: float = 9.81            # m/s²
    motor_tau: float = 0.03          # s, motor time constant

    def rotor_positions(self):
        """X configuration. Rotors: 0 front-right, 1 rear-left, 2 front-left, 3 rear-right."""
        d = self.arm_length / np.sqrt(2)
        return np.array([[d, -d, 0], [-d, d, 0], [d, d, 0], [-d, -d, 0]])

    def rotor_spins(self):
        """+1 = counterclockwise seen from above, -1 = clockwise."""
        return np.array([1, 1, -1, -1])

    def mixer(self):
        """Matrix M such that [total thrust, τx, τy, τz] = M @ rotor thrusts."""
        r = self.rotor_positions()
        s = self.rotor_spins()
        return np.vstack([
            np.ones(4),
            r[:, 1],                    # τx = y × T
            -r[:, 0],                   # τy = -x × T
            -s * self.torque_coeff,     # τz: rotor reaction, opposite to its spin
        ])
