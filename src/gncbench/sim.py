"""Boucle de simulation : physique à pas fixe, contrôleur à sa propre fréquence."""

from dataclasses import dataclass, field

import numpy as np

from .dynamics import state_derivative
from .integrators import rk4_step


@dataclass
class SimResult:
    t: np.ndarray        # (N,)
    states: np.ndarray   # (N, 17)
    rotors: np.ndarray   # (N, 4)
    measurements: dict = field(default_factory=dict)  # nom -> (temps (M,), mesures (M, k))


def simulate(x0, controller, reference, params, duration, dt=0.002, ctrl_every=2, sensors=()):
    """Simule `duration` secondes. Le contrôleur tourne tous les `ctrl_every` pas (ici 250 Hz)."""
    n = int(round(duration / dt))
    t = np.arange(n + 1) * dt
    states = np.empty((n + 1, len(x0)))
    rotors = np.empty((n + 1, 4))
    x = np.array(x0, dtype=float)
    u = np.zeros(4)
    controller.reset()
    logs = {sensor.name: ([], []) for sensor in sensors}

    for k in range(n + 1):
        if k % ctrl_every == 0:
            ref = reference(t[k]) if callable(reference) else reference
            u = controller.update(t[k], x, ref)
        states[k], rotors[k] = x, u
        for sensor in sensors:
            if k % round(1.0 / (sensor.rate * dt)) == 0:
                times, values = logs[sensor.name]
                times.append(t[k])
                values.append(sensor.measure(t[k], x, state_derivative(x, u, params), params))
        if k < n:
            x = rk4_step(lambda s: state_derivative(s, u, params), x, dt)

    measurements = {name: (np.array(ts), np.array(vs)) for name, (ts, vs) in logs.items()}
    return SimResult(t, states, rotors, measurements)
