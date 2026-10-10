"""Intégrateur Runge-Kutta d'ordre 4."""

import numpy as np

from .dynamics import QUAT


def rk4_step(f, x, dt):
    """Un pas de RK4 pour dx/dt = f(x). Le quaternion est renormalisé à la fin."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[QUAT] /= np.linalg.norm(x_next[QUAT])
    return x_next
