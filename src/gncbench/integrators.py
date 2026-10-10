"""Fourth-order Runge-Kutta integrator."""

import numpy as np

from .dynamics import QUAT


def rk4_step(f, x, dt):
    """One RK4 step for dx/dt = f(x). The quaternion is renormalized at the end."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[QUAT] /= np.linalg.norm(x_next[QUAT])
    return x_next
