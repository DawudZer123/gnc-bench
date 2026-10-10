"""Injection de pannes capteurs, pour tester la robustesse de la navigation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorFault:
    """Une panne sur un capteur, active entre `start` et `end` (s).

    mode = "dropout" : le capteur ne renvoie plus rien (NaN), comme un GPS qui perd ses satellites.
    mode = "freeze"  : le capteur renvoie toujours sa dernière valeur, comme un pilote figé.
    """

    sensor: str           # nom du capteur visé : "gps", "imu", "baro"
    start: float
    end: float
    mode: str = "dropout"

    def is_active(self, t):
        return self.start <= t < self.end

    def apply(self, measurement, last_good):
        if self.mode == "dropout":
            return np.full_like(measurement, np.nan)
        if self.mode == "freeze":
            return last_good.copy() if last_good is not None else measurement
        raise ValueError(f"Mode de panne inconnu : {self.mode}")
