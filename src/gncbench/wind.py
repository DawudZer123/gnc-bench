"""Vent : une composante moyenne + des rafales aléatoires corrélées dans le temps."""

import numpy as np


class Wind:
    """Rafales modélisées par un processus de Gauss-Markov du 1er ordre (bruit blanc filtré).

    Chaque rafale dure en moyenne `gust_time` secondes, avec une amplitude d'écart-type `gust_std`.
    C'est une version simplifiée du modèle de turbulence de Dryden utilisé en aéronautique.
    """

    def __init__(self, rng, mean=(0.0, 0.0, 0.0), gust_std=1.5, gust_time=1.0):
        self.rng = rng
        self.mean = np.asarray(mean, dtype=float)  # m/s, repère monde
        self.gust_std = gust_std                     # m/s
        self.gust_time = gust_time                   # s
        self.gust = np.zeros(3)

    def step(self, dt):
        """Avance d'un pas de temps et renvoie la vitesse du vent (m/s, repère monde)."""
        a = np.exp(-dt / self.gust_time)
        self.gust = a * self.gust + self.gust_std * np.sqrt(1 - a * a) * self.rng.normal(size=3)
        return self.mean + self.gust
