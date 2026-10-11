"""FINAL DEMO: the drone loses its GPS for 20 s while flying a figure-8 in the wind.

Version A: Kalman filter with only IMU + barometer during the outage.
Version B: same filter + optical flow.
The drone flies on its ESTIMATE (like a real drone), not on the truth.

Run: python scripts/final_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.scenario import Scenario
from gncbench.trajectories import figure_eight

SCENARIO = Scenario()   # figure-8, 3 m/s wind + 1.5 m/s gusts, GPS lost from 10 to 30 s
OUTAGE = SCENARIO.outage
CORRIDOR = 0.5          # m


def main():
    runs = {"Version A: IMU + baro": SCENARIO.fly(sensors=("imu", "gps", "baro"), corridor=CORRIDOR),
            "Version B: + optical flow": SCENARIO.fly(corridor=CORRIDOR)}
    ref = np.array([figure_eight(t)["position"] for t in np.linspace(0, 20, 400)])

    fig = plt.figure(figsize=(11, 7))
    top = fig.add_subplot(2, 2, 1)
    top_b = fig.add_subplot(2, 2, 2, sharex=top, sharey=top)
    bottom = fig.add_subplot(2, 1, 2)

    print(f"{OUTAGE[1] - OUTAGE[0]:.0f} s GPS outage, ±{CORRIDOR} m corridor\n")
    for (name, (res, rep)), ax, color in zip(runs.items(), (top, top_b), ("C3", "C2")):
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
