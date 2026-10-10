"""Boucle de simulation : physique à pas fixe, contrôleur à sa propre fréquence."""

from dataclasses import dataclass, field

import numpy as np

from .dynamics import POS, QUAT, VEL, state_derivative
from .integrators import rk4_step


@dataclass
class SimResult:
    t: np.ndarray        # (N,)
    states: np.ndarray   # (N, 17)
    rotors: np.ndarray   # (N, 4)
    measurements: dict = field(default_factory=dict)  # nom -> (temps (M,), mesures (M, k))
    wind: np.ndarray = None                            # (N, 3), vitesse du vent
    estimates: np.ndarray = None                       # (N, 6), position et vitesse estimées
    estimate_std: np.ndarray = None                    # (N, 3), écart-type estimé de la position


def simulate(x0, controller, reference, params, duration, dt=0.002, ctrl_every=2, sensors=(),
             wind=None, faults=(), estimator=None, use_estimate=False):
    """Simule `duration` secondes. Le contrôleur tourne tous les `ctrl_every` pas (ici 250 Hz).

    estimator    : filtre de navigation qui reçoit les mesures des capteurs (optionnel).
    use_estimate : si True, le contrôleur vole avec la position et la vitesse ESTIMÉES
                   au lieu des vraies. C'est la situation d'un vrai drone.
    """
    n = int(round(duration / dt))
    t = np.arange(n + 1) * dt
    states = np.empty((n + 1, len(x0)))
    rotors = np.empty((n + 1, 4))
    winds = np.zeros((n + 1, 3))
    estimates = np.full((n + 1, 6), np.nan)
    estimate_std = np.full((n + 1, 3), np.nan)
    x = np.array(x0, dtype=float)
    u = np.zeros(4)
    controller.reset()
    logs = {sensor.name: ([], []) for sensor in sensors}
    last_good = {sensor.name: None for sensor in sensors}   # dernière mesure valide

    for k in range(n + 1):
        if k % ctrl_every == 0:
            ref = reference(t[k]) if callable(reference) else reference
            x_ctrl = x.copy()
            if estimator is not None and use_estimate:
                x_ctrl[POS], x_ctrl[VEL] = estimator.position, estimator.velocity
            u = controller.update(t[k], x_ctrl, ref)
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
                if estimator is not None:
                    estimator.process(sensor.name, z, x[QUAT], 1.0 / sensor.rate)
        if estimator is not None:
            estimates[k] = np.concatenate([estimator.position, estimator.velocity])
            estimate_std[k] = estimator.position_std
        if k < n:
            x = rk4_step(lambda s: state_derivative(s, u, params, f_wind), x, dt)

    measurements = {name: (np.array(ts), np.array(vs)) for name, (ts, vs) in logs.items()}
    return SimResult(t, states, rotors, measurements, winds, estimates, estimate_std)
