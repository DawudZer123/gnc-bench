"""Sensor fault injection, to test the robustness of the navigation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorFault:
    """A fault on one sensor, active between `start` and `end` (s).

    mode = "dropout": the sensor returns nothing (NaN), like a GPS that loses its satellites.
    mode = "freeze":  the sensor keeps returning its last value, like a frozen driver.
    """

    sensor: str           # name of the target sensor: "gps", "imu", "baro", "flow"
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
        raise ValueError(f"Unknown fault mode: {self.mode}")
