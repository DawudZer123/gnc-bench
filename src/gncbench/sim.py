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
    wind: np.ndarray = None                            # (N, 3), vitesse du vent


def simulate(x0, controller, reference, params, duration, dt=0.002, ctrl_every=2, sensors=(),
             wind=None, faults=()):
    """Simule `duration` secondes. Le contrôleur tourne tous les `ctrl_every` pas (ici 250 Hz)."""
    n = int(round(duration / dt))
    t = np.arange(n + 1) * dt
    states = np.empty((n + 1, len(x0)))
    rotors = np.empty((n + 1, 4))
    winds = np.zeros((n + 1, 3))
    x = np.array(x0, dtype=float)
    u = np.zeros(4)
    controller.reset()
    logs = {sensor.name: ([], []) for sensor in sensors}
    last_good = {sensor.name: None for sensor in sensors}   # dernière mesure valide

    for k in range(n + 1):
        if k % ctrl_every == 0:
            ref = reference(t[k]) if callable(reference) else reference
            u = controller.update(t[k], x, ref)
        # Le vent pousse le drone via la traînée : F = c × (vitesse du vent).
        winds[k] = wind.step(dt) if wind is not None else 0.0
        f_wind = params.drag_coeff * winds[k]
        states[k], rotors[k] = x, u
        for sensor in sensors:
            if k % round(1.0 / (sensor.rate * dt)) == 0:
                z = sensor.measure(t[k], x, state_derivative(x, u, params, f_wind), params)
                active = [f for f in faults if f.sensor == sensor.name and f.is_active(t[k])]
                if active:
                    z = active[0].apply(z, last_good[sensor.name])
                else:
                    last_good[sensor.name] = z
                times, values = logs[sensor.name]
                times.append(t[k])
                values.append(z)
        if k < n:
            x = rk4_step(lambda s: state_derivative(s, u, params, f_wind), x, dt)

    measurements = {name: (np.array(ts), np.array(vs)) for name, (ts, vs) in logs.items()}
    return SimResult(t, states, rotors, measurements, winds)
