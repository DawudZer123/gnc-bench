"""FINAL DEMO: the drone loses its GPS for 20 s while flying a figure-8 in the wind.

Version A: Kalman filter with only IMU + barometer during the outage.
Version B: same filter + optical flow.
The drone flies on its ESTIMATE (like a real drone), not on the truth.

Run: python scripts/final_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, initial_state
from gncbench.estimator import NavigationFilter
from gncbench.faults import SensorFault
from gncbench.metrics import tracking_report
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer, OpticalFlow
from gncbench.sim import simulate
from gncbench.wind import Wind

OUTAGE = (10.0, 30.0)   # s
CORRIDOR = 0.5          # m


def figure_eight(t, period=20.0, size=3.0):
    """A 6 m × 3 m figure-8 at 2 m altitude, with velocity and acceleration for feedforward."""
    w = 2 * np.pi / period
    return {
        "position": np.array([size * np.sin(w * t), size / 2 * np.sin(2 * w * t), 2.0]),
        "velocity": np.array([size * w * np.cos(w * t), size * w * np.cos(2 * w * t), 0.0]),
        "acceleration": np.array([-size * w**2 * np.sin(w * t),
                                  -2 * size * w**2 * np.sin(2 * w * t), 0.0]),
    }


def fly(with_flow, seed=1):
    params = QuadParams()
    rng = np.random.default_rng(seed)
    x0 = initial_state(position=figure_eight(0)["position"])
    x0[MOTORS] = params.mass * params.gravity / 4
    sensors = [IMU(rng), GPS(rng), Barometer(rng)]
    if with_flow:
        sensors.append(OpticalFlow(rng))
    return simulate(x0, HoverController(params, dt=0.004, ki=1.0), figure_eight, params,
                    duration=40.0, sensors=sensors,
                    faults=[SensorFault("gps", start=OUTAGE[0], end=OUTAGE[1])],
                    estimator=NavigationFilter(params, p0=x0[POS]), use_estimate=True,
                    wind=Wind(np.random.default_rng(seed + 10), mean=(3.0, 0.0, 0.0), gust_std=1.5))


def main():
    runs = {"Version A: IMU + baro": fly(False), "Version B: + optical flow": fly(True)}
    ref = np.array([figure_eight(t)["position"] for t in np.linspace(0, 20, 400)])

    fig = plt.figure(figsize=(11, 7))
    top = fig.add_subplot(2, 2, 1)
    top_b = fig.add_subplot(2, 2, 2, sharex=top, sharey=top)
    bottom = fig.add_subplot(2, 1, 2)

    print(f"{OUTAGE[1] - OUTAGE[0]:.0f} s GPS outage, ±{CORRIDOR} m corridor\n")
    for (name, res), ax, color in zip(runs.items(), (top, top_b), ("C3", "C2")):
        rep = tracking_report(res, figure_eight, CORRIDOR, window=OUTAGE)
        print(f"{name}")
        print(f"  max error           : {rep['max_error_m']:.2f} m")
        print(f"  time out of corridor: {rep['time_out_of_corridor_s']:.1f} s")
        print(f"  verdict             : {rep['verdict']}\n")

        m = (res.t >= OUTAGE[0]) & (res.t < OUTAGE[1])
        ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1, label="desired trajectory")
        ax.plot(res.states[m, 0], res.states[m, 1], color=color, label="actual flight during outage")
        ax.set_title(f"{name}\n{rep['verdict']} (max {rep['max_error_m']:.2f} m)", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower right")
        bottom.plot(res.t, rep["error"], color=color, label=name)
    top.set_ylabel("y (m)")

    bottom.axvspan(*OUTAGE, color="red", alpha=0.08, label="GPS outage")
    bottom.axhline(CORRIDOR, color="k", ls=":", lw=1, label="corridor limit")
    bottom.set_xlabel("Time (s)")
    bottom.set_ylabel("Distance from trajectory (m)")
    bottom.legend(fontsize=8)
    fig.suptitle("Final demo: figure-8 flight in the wind, GPS lost for 20 s")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "final_demo.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
