"""Monte Carlo campaigns on the real PX4: random wind, gusts and GPS faults, many flights in parallel.

Each flight starts a fresh PX4 instance (its own ports: 4560 + i and 14540 + i), so several
flights run at the same time without seeing each other.
"""

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from .faults import SensorFault
from .px4flight import fly_px4

# PX4 main modes (custom_mode >> 16)
MODE_NAMES = {1: "Manual", 2: "Altitude", 3: "Position", 4: "Auto", 5: "Acro", 6: "Offboard",
              7: "Stabilized"}
AUTO_NAMES = {2: "Takeoff", 3: "Hold", 4: "Mission", 5: "Return", 6: "Land"}


@dataclass
class Px4Scenario:
    """The conditions of one PX4 flight. Fault times count from the start of the trajectory."""

    seed: int = 1
    duration: float = 30.0
    wind_speed: float = 3.0
    wind_heading: float = 0.0
    gust_std: float = 1.5
    gps_fault: str = "dropout"       # "none", "dropout", "freeze", "offset", "drift"
    fault_start: float = 10.0
    fault_length: float = 15.0
    fault_value: tuple = ()
    flow: bool = False
    twin: str = ""                   # path to a twin.json: fly that drone instead of the Iris
    inertia_scale: float = 1.0       # multiplies the twin's inertia (it is often a prior)
    mass: float = 1.8
    arm: float = 0.24

    @classmethod
    def random(cls, seed, duration=30.0, flow=False, **drone):
        rng = np.random.default_rng(seed + 100_000)
        return cls(seed=seed, duration=duration, flow=flow, **drone,
                   wind_speed=float(rng.uniform(0, 6)), wind_heading=float(rng.uniform(0, 360)),
                   gust_std=float(rng.uniform(0, 2)),
                   fault_start=float(rng.uniform(0.15, 0.4) * duration),
                   fault_length=float(rng.uniform(0.15, 0.5) * duration))

    def faults(self):
        if self.gps_fault == "none":
            return []
        return [SensorFault("gps", self.fault_start, self.fault_start + self.fault_length,
                            self.gps_fault, tuple(self.fault_value))]

    def drone(self):
        """fly_px4 arguments for the drone: nothing for the Iris, the twin's models otherwise."""
        if not self.twin:
            return {}
        from .twin import DigitalTwin

        twin = DigitalTwin.load(self.twin)
        params = twin.quad_params(self.mass, self.arm)
        params.inertia = params.inertia * self.inertia_scale
        return {"params": params, "px4_params": twin.px4_params, "noise": twin.sensor_noise(),
                "battery": twin.battery_model()}


def flight_report(flight, scenario, corridor=1.0):
    """What happened: did PX4 finish the mission, how well did it track, where did it end up."""
    from .px4flight import LANDING_TIME, SETTLE_TIME, TAKEOFF_TIME

    t0 = flight.armed_at + TAKEOFF_TIME + SETTLE_TIME
    window = (flight.t >= t0) & (flight.t < t0 + scenario.duration)
    err = flight.states[window, :3] - flight.reference[window]
    horizontal = np.linalg.norm(err[:, :2], axis=1)
    # Any mode change while armed before the landing phase = PX4 left our mission (failsafe).
    end = t0 + scenario.duration
    failsafe = [(t, m, s) for t, m, s, armed in flight.modes
                if flight.armed_at < t < end and armed and m != 6]
    reason = ""
    if failsafe:
        t, m, s = failsafe[0]
        reason = f"{MODE_NAMES.get(m, m)}" + (f" {AUTO_NAMES.get(s, s)}" if m == 4 else "")
        reason += f" at t+{t - t0:.1f} s"
    # After a failsafe PX4 flies on its own: how far did it drift from where it gave up?
    drift = 0.0
    if failsafe:
        k = np.searchsorted(flight.t, failsafe[0][0])
        drift = float(np.linalg.norm(flight.states[-1, :2] - flight.states[k, :2]))
    completed = not failsafe
    max_h = float(horizontal.max())
    return {
        **{k: v for k, v in asdict(scenario).items() if k != "fault_value"},
        "completed": completed,
        "failsafe": reason,
        "mean_error_m": float(horizontal.mean()),
        "max_error_m": max_h,
        "vertical_error_m": float(np.abs(err[:, 2]).mean()),
        "failsafe_drift_m": drift,
        "verdict": "PASS" if completed and max_h < corridor else "FAIL",
        "missed_steps": flight.missed_steps,
        "messages": [m for _, m in flight.messages],
    }


_instance = None


def _init_worker(queue):
    global _instance
    _instance = queue.get()


def _fly_one(args):
    scenario, corridor, px4_dir = args
    for attempt in range(2):             # a PX4 start-up can fail on a busy machine: retry once
        try:
            flight = fly_px4(duration=scenario.duration, seed=scenario.seed,
                             wind_speed=scenario.wind_speed, wind_heading=scenario.wind_heading,
                             gust_std=scenario.gust_std, faults=scenario.faults(),
                             flow=scenario.flow, instance=_instance, px4_dir=px4_dir,
                             **scenario.drone())
            return flight_report(flight, scenario, corridor)
        except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
            error = exc
            time.sleep(1.0)
    return {**asdict(scenario), "verdict": "ERROR", "error": str(error)}


def run_px4_campaign(scenarios, workers=4, corridor=1.0, px4_dir=None, verbose=True):
    """Fly every scenario on its own PX4 instance, `workers` at a time."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for i in range(workers):
        queue.put(i)
    rows = []
    start = time.time()
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(queue,)) as pool:
        for i, row in enumerate(pool.map(_fly_one, [(s, corridor, px4_dir) for s in scenarios])):
            rows.append(row)
            if verbose:
                print(f"  {i + 1}/{len(scenarios)} seed {row['seed']}: {row['verdict']}"
                      f" {row.get('failsafe', '')}  ({time.time() - start:.0f} s)", flush=True)
    return rows


def summarize_px4(rows):
    ok = [r for r in rows if r["verdict"] != "ERROR"]
    errors = np.array([r["max_error_m"] for r in ok])
    worst = max(ok, key=lambda r: r["max_error_m"])
    return {
        "flights": len(rows),
        "errors": len(rows) - len(ok),
        "pass_rate": np.mean([r["verdict"] == "PASS" for r in ok]),
        "completed_rate": np.mean([r["completed"] for r in ok]),
        "median_max_error_m": float(np.median(errors)),
        "worst_max_error_m": float(errors.max()),
        "worst_seed": worst["seed"],
        "failsafes": sum(bool(r["failsafe"]) for r in ok),
        "mean_failsafe_drift_m": float(np.mean([r["failsafe_drift_m"] for r in ok
                                                 if r["failsafe"]] or [0.0])),
    }
