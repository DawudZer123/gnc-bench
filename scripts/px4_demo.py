"""THE REAL AUTOPILOT: PX4 flies on our physics and loses its GPS for 20 s.

Same final demo as before, but the controller and the estimator are now PX4's own flight code
(the one on real drones), connected to our simulator by MAVLink.
Version A: PX4 with GPS only.  Version B: PX4 + downward optical flow camera.

Needs a PX4 SITL build (README, section 7).
Run: python scripts/px4_demo.py            (PX4 folder in $PX4_DIR or ~/PX4-Autopilot)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.px4campaign import MODE_NAMES, Px4Scenario, flight_report
from gncbench.px4flight import SETTLE_TIME, TAKEOFF_TIME, fly_px4

SCENARIO = Px4Scenario(duration=40.0, wind_speed=3.0, gust_std=1.5, fault_start=10.0,
                       fault_length=20.0)
CORRIDOR = 1.0     # m (PX4's own GPS-based estimate wanders by a few tens of cm)


def fly(flow):
    s = Px4Scenario(**{**SCENARIO.__dict__, "flow": flow})
    flight = fly_px4(duration=s.duration, seed=s.seed, wind_speed=s.wind_speed,
                     wind_heading=s.wind_heading, gust_std=s.gust_std, faults=s.faults(),
                     flow=flow, verbose=True)
    return flight, flight_report(flight, s, CORRIDOR)


def main():
    runs = {}
    for name, flow in (("Version A: PX4, GPS only", False), ("Version B: PX4 + optical flow", True)):
        print(name)
        runs[name] = fly(flow)

    fig = plt.figure(figsize=(11, 7.5))
    axes = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)]
    bottom = fig.add_subplot(2, 1, 2)
    print(f"\n{SCENARIO.fault_length:.0f} s GPS outage, ±{CORRIDOR} m corridor\n")
    for (name, (f, rep)), ax, color in zip(runs.items(), axes, ("C3", "C2")):
        print(name)
        print(f"  mission completed : {'yes' if rep['completed'] else 'NO, failsafe: ' + rep['failsafe']}")
        print(f"  max error         : {rep['max_error_m']:.2f} m")
        if rep["failsafe"]:
            print(f"  drift after failsafe: {rep['failsafe_drift_m']:.1f} m")
        print(f"  verdict           : {rep['verdict']}\n")

        t0 = f.armed_at + TAKEOFF_TIME + SETTLE_TIME
        rel = f.t - t0
        fly_window = (rel >= 0) & (rel < SCENARIO.duration + 6)
        ref = (rel >= 0) & (rel < SCENARIO.duration)
        ax.plot(f.reference[ref, 0], f.reference[ref, 1], "k--", lw=1, label="mission")
        ax.plot(f.states[fly_window, 0], f.states[fly_window, 1], color=color, label="actual flight")
        ax.plot(*f.states[-1, :2], "kx", ms=9, label="touchdown")
        ax.set_title(f"{name}\n{rep['verdict']}" + (f" ({rep['failsafe']})" if rep["failsafe"] else ""),
                     fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, loc="lower right")
        err = np.linalg.norm(f.states[ref, :2] - f.reference[ref, :2], axis=1)
        bottom.plot(rel[ref], err, color=color, label=name)
        for t, m, s, armed in f.modes:
            if armed and t > t0 and m != 6:
                bottom.axvline(t - t0, color=color, ls="--", lw=1)
                bottom.text(t - t0, bottom.get_ylim()[1] * 0.9 if bottom.get_ylim()[1] > 1 else 1,
                            f" PX4 -> {MODE_NAMES.get(m, m)}", color=color, fontsize=8)
                break
    axes[0].set_ylabel("y (m)")
    out_start, out_end = SCENARIO.fault_start, SCENARIO.fault_start + SCENARIO.fault_length
    bottom.axvspan(out_start, out_end, color="red", alpha=0.08, label="GPS outage")
    bottom.axhline(CORRIDOR, color="k", ls=":", lw=1, label="corridor limit")
    bottom.set_xlabel("Time since mission start (s)")
    bottom.set_ylabel("Horizontal distance from mission (m)")
    bottom.legend(fontsize=8)
    fig.suptitle("PX4 autopilot on our physics: figure-8 in the wind, GPS lost for 20 s")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "px4_demo.png", dpi=120)
    print(f"Plot saved: {out / 'px4_demo.png'}")
    plt.show()


if __name__ == "__main__":
    main()
