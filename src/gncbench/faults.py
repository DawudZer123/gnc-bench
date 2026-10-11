"""Sensor fault injection, to test the robustness of the navigation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorFault:
    """A fault on one sensor, active between `start` and `end` (s).

    mode = "dropout": the sensor returns nothing (NaN), like a GPS that loses its satellites.
    mode = "freeze":  the sensor keeps returning its last value, like a frozen driver.
    mode = "offset":  `value` is added to the measurement: a sudden jump (GPS spoofing, multipath).
    mode = "drift":   `value` × (time since start) is added: a slow, sneaky spoofing attack.
    `value` covers the first components of the measurement (for the GPS: x, y, z position).
    """

    sensor: str           # name of the target sensor: "gps", "imu", "baro", "flow", "mag"
    start: float
    end: float
    mode: str = "dropout"
    value: tuple = ()

    def is_active(self, t):
        return self.start <= t < self.end

    def _padded(self, measurement):
        out = np.zeros_like(measurement)
        out[:len(self.value)] = self.value
        return out

    def apply(self, measurement, last_good, t=None):
        if self.mode == "dropout":
            return np.full_like(measurement, np.nan)
        if self.mode == "freeze":
            return last_good.copy() if last_good is not None else measurement
        if self.mode == "offset":
            return measurement + self._padded(measurement)
        if self.mode == "drift":
            return measurement + self._padded(measurement) * (t - self.start)
        raise ValueError(f"Unknown fault mode: {self.mode}")
