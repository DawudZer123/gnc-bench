"""Monte Carlo campaigns: fly the same controller through many random scenarios.

One flight can be lucky. A hundred flights in different winds, outages and masses
give a pass rate and a worst case: that is what tells you whether a controller is safe.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

from .scenario import Scenario, default_controller


def _fly_one(scenario, controller, sensors, corridor, navigation):
    _, report = scenario.fly(controller, sensors, corridor, navigation=navigation)
    return {**asdict(scenario), **{k: v for k, v in report.items() if k != "error"}}


def run_campaign(n, controller=default_controller, sensors=("imu", "gps", "baro", "flow"),
                 first_seed=0, duration=30.0, corridor=0.5, workers=None, verbose=True,
                 navigation="kf"):
    """Flies n random scenarios and returns one row (dict) per flight, in seed order.

    workers: number of flights run in parallel (None = one per CPU core).
             The controller must then be a top-level function or class, not a lambda.
    """
    scenarios = [Scenario.random(first_seed + i, duration) for i in range(n)]
    rows = [None] * n
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_fly_one, s, controller, sensors, corridor, navigation): i
                   for i, s in enumerate(scenarios)}
        for done, future in enumerate(as_completed(futures), start=1):
            rows[futures[future]] = future.result()
            if verbose:
                print(f"\r  {done}/{n} flights", end="", flush=True)
    if verbose:
        print()
    return rows


def summarize(rows):
    """Campaign summary: pass rate, typical and worst errors, and which seed to replay."""
    max_err = np.array([r["max_error_m"] for r in rows])
    passed = np.array([r["verdict"] == "PASS" for r in rows])
    worst = int(np.argmax(max_err))
    effort = np.array([r["control_effort_N"] for r in rows])
    return {
        "flights": len(rows),
        "pass_rate": float(passed.mean()),
        "median_max_error_m": float(np.median(max_err)),
        "p95_max_error_m": float(np.percentile(max_err, 95)),
        "worst_max_error_m": float(max_err[worst]),
        "worst_seed": rows[worst]["seed"],
        "mean_control_effort_N": float(effort.mean()),
        "saturated_flights": int(sum(r["saturation_time_s"] > 0 for r in rows)),
    }
