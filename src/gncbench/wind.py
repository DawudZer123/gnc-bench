"""Wind: a mean component + random gusts that are correlated in time."""

import numpy as np


class Wind:
    """Gusts modeled by a first-order Gauss-Markov process (filtered white noise).

    Each gust lasts `gust_time` seconds on average, with a standard deviation of `gust_std`.
    This is a simplified version of the Dryden turbulence model used in aeronautics.
    """

    def __init__(self, rng, mean=(0.0, 0.0, 0.0), gust_std=1.5, gust_time=1.0):
        self.rng = rng
        self.mean = np.asarray(mean, dtype=float)  # m/s, world frame
        self.gust_std = gust_std                     # m/s
        self.gust_time = gust_time                   # s
        self.gust = np.zeros(3)

    def step(self, dt):
        """Advances one time step and returns the wind velocity (m/s, world frame)."""
        a = np.exp(-dt / self.gust_time)
        self.gust = a * self.gust + self.gust_std * np.sqrt(1 - a * a) * self.rng.normal(size=3)
        return self.mean + self.gust
