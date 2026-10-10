"""Automatic test bench report: did the controller hold its trajectory?"""

import numpy as np

from .dynamics import POS


def tracking_report(res, reference, corridor=0.5, window=None):
    """Compares the drone's true position with the desired trajectory.

    corridor: half-width of the allowed corridor (m).
    window:   (start, end) in s to judge only part of the flight (e.g. the GPS outage).
    """
    ref = np.array([reference(t)["position"] for t in res.t])
    error = np.linalg.norm(res.states[:, POS] - ref, axis=1)
    mask = np.ones_like(res.t, dtype=bool)
    if window is not None:
        mask = (res.t >= window[0]) & (res.t < window[1])
    dt = res.t[1] - res.t[0]
    out = error[mask] > corridor
    return {
        "mean_error_m": float(error[mask].mean()),
        "max_error_m": float(error[mask].max()),
        "time_out_of_corridor_s": float(out.sum() * dt),
        "verdict": "PASS" if not out.any() else "FAIL",
        "error": error,
    }
