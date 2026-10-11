"""Simulation loop: fixed-step physics, controller at its own rate."""

from dataclasses import dataclass, field

import numpy as np

from .dynamics import OMEGA, POS, QUAT, VEL, state_derivative
from .integrators import rk4_step


@dataclass
class SimResult:
    t: np.ndarray        # (N,)
    states: np.ndarray   # (N, 17)
    rotors: np.ndarray   # (N, 4)
    measurements: dict = field(default_factory=dict)  # name -> (times (M,), measurements (M, k))
    wind: np.ndarray = None                            # (N, 3), wind velocity
    estimates: np.ndarray = None                       # (N, 6), estimated position and velocity
    estimate_std: np.ndarray = None                    # (N, 3), estimated position standard deviation
    attitude_estimates: np.ndarray = None              # (N, 4), estimated quaternion (EKF only)


def simulate(x0, controller, reference, params, duration, dt=0.002, ctrl_every=2, sensors=(),
             wind=None, faults=(), estimator=None, use_estimate=False):
    """Simulates `duration` seconds. The controller runs every `ctrl_every` steps (here 250 Hz).

    estimator:    navigation filter that receives the sensor measurements (optional).
    use_estimate: if True, the controller flies with the ESTIMATED position and velocity
                  instead of the true ones. This is what a real drone does. If the filter
                  also estimates the attitude (NavigationEKF), the attitude and angular
                  rate given to the controller are the estimated ones too.
    """
    n = int(round(duration / dt))
    t = np.arange(n + 1) * dt
    states = np.empty((n + 1, len(x0)))
    rotors = np.empty((n + 1, 4))
    winds = np.zeros((n + 1, 3))
    estimates = np.full((n + 1, 6), np.nan)
    estimate_std = np.full((n + 1, 3), np.nan)
    has_attitude = hasattr(estimator, "attitude")
    attitude_estimates = np.full((n + 1, 4), np.nan) if has_attitude else None
    x = np.array(x0, dtype=float)
    u = np.zeros(4)
    controller.reset()
    if estimator is not None:
        controller.attach(estimator)
    logs = {sensor.name: ([], []) for sensor in sensors}
    last_good = {sensor.name: None for sensor in sensors}   # last valid measurement

    for k in range(n + 1):
        if k % ctrl_every == 0:
            ref = reference(t[k]) if callable(reference) else reference
            x_ctrl = x.copy()
            if estimator is not None and use_estimate:
                x_ctrl[POS], x_ctrl[VEL] = estimator.position, estimator.velocity
                if has_attitude:
                    x_ctrl[QUAT], x_ctrl[OMEGA] = estimator.attitude, estimator.angular_rate
            u = controller.update(t[k], x_ctrl, ref)
        # The wind pushes the drone through drag: F = c × (wind velocity).
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
            if has_attitude:
                attitude_estimates[k] = estimator.attitude
        if k < n:
            x = rk4_step(lambda s: state_derivative(s, u, params, f_wind), x, dt)

    measurements = {name: (np.array(ts), np.array(vs)) for name, (ts, vs) in logs.items()}
    return SimResult(t, states, rotors, measurements, winds, estimates, estimate_std,
                     attitude_estimates)
