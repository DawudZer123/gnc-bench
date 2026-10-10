"""Boucle de simulation : physique à pas fixe, contrôleur à sa propre fréquence."""

from dataclasses import dataclass

import numpy as np

from .dynamics import state_derivative
from .integrators import rk4_step


@dataclass
class SimResult:
    t: np.ndarray        # (N,)
    states: np.ndarray   # (N, 13)
    rotors: np.ndarray   # (N, 4)


def simulate(x0, controller, reference, params, duration, dt=0.002, ctrl_every=2):
    """Simule `duration` secondes. Le contrôleur tourne tous les `ctrl_every` pas (ici 250 Hz)."""
    n = int(round(duration / dt))
    t = np.arange(n + 1) * dt
    states = np.empty((n + 1, 13))
    rotors = np.empty((n + 1, 4))
    x = np.array(x0, dtype=float)
    u = np.zeros(4)
    controller.reset()

    for k in range(n + 1):
        if k % ctrl_every == 0:
            ref = reference(t[k]) if callable(reference) else reference
            u = controller.update(t[k], x, ref)
        states[k], rotors[k] = x, u
        if k < n:
            x = rk4_step(lambda s: state_derivative(s, u, params), x, dt)

    return SimResult(t, states, rotors)
