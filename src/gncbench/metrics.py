"""Automatic test bench report: did the controller hold its trajectory, and at what cost?"""

import numpy as np

from .dynamics import POS
from .params import QuadParams


def tracking_report(res, reference, corridor=0.5, window=None, params=None):
    """Compares the drone's true position with the desired trajectory.

    corridor: half-width of the allowed corridor (m).
    window:   (start, end) in s to judge only part of the flight (e.g. the GPS outage).
    params:   the drone that actually flew, for the hover thrust and the motor limit.

    The verdict only looks at the corridor. The control effort tells you the price:
    two controllers can both pass, one of them by hammering the motors.
    """
    params = params or QuadParams()
    ref = np.array([reference(t)["position"] for t in res.t])
    error = np.linalg.norm(res.states[:, POS] - ref, axis=1)
    mask = np.ones_like(res.t, dtype=bool)
    if window is not None:
        mask = (res.t >= window[0]) & (res.t < window[1])
    dt = res.t[1] - res.t[0]
    out = error[mask] > corridor

    # Control effort: how far the rotor commands stray from hover thrust (RMS),
    # and how long at least one motor sits at its limit (0 or max): no margin left there.
    hover = params.mass * params.gravity / 4
    cmd = res.rotors[mask]
    saturated = ((cmd <= 1e-6) | (cmd >= params.max_thrust - 1e-6)).any(axis=1)
    return {
        "mean_error_m": float(error[mask].mean()),
        "max_error_m": float(error[mask].max()),
        "time_out_of_corridor_s": float(out.sum() * dt),
        "control_effort_N": float(np.sqrt(np.mean((cmd - hover) ** 2))),
        "peak_thrust_N": float(cmd.max()),
        "saturation_time_s": float(saturated.sum() * dt),
        "verdict": "PASS" if not out.any() else "FAIL",
        "error": error,
    }
