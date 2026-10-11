"""Week 4: 3D view of a flight, Version A vs Version B on the same scenario.

Run: python scripts/flight3d_demo.py              (final demo scenario)
     python scripts/flight3d_demo.py 64           (replay campaign flight with seed 64)
     python scripts/flight3d_demo.py --animate    (animated drone, Version B)
     python scripts/flight3d_demo.py --animate --A   (animated drone, Version A: watch it drift)
     python scripts/flight3d_demo.py --gif        (also saves outputs/flight3d_B.gif for the slides; add --A for A)
Tip: in the 3D window, click and drag to rotate the view.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from gncbench.scenario import Scenario
from gncbench.trajectories import figure_eight
from gncbench.viz import animate_flight, plot_flight_3d

VERSIONS = {"Version A: IMU + baro": ("imu", "gps", "baro"),
            "Version B: + optical flow": ("imu", "gps", "baro", "flow")}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scenario = Scenario.random(int(args[0])) if args else Scenario()
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)

    if "--animate" in sys.argv or "--gif" in sys.argv:
        name = list(VERSIONS)[0 if "--A" in sys.argv else 1]
        res, rep = scenario.fly(sensors=VERSIONS[name])
        fig, anim = animate_flight(res, figure_eight, scenario.outage,
                                   title=f"{name} ({rep['verdict']})")
        if "--gif" in sys.argv:
            print("Saving the GIF (about a minute)...")
            anim.save(out / f"flight3d_{name[8]}.gif", writer="pillow", fps=25, dpi=80)
            print(f"GIF saved: {out / f'flight3d_{name[8]}.gif'}")
        plt.show()
        return

    fig = plt.figure(figsize=(12, 5.5))
    for i, (name, sensors) in enumerate(VERSIONS.items()):
        res, rep = scenario.fly(sensors=sensors)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        plot_flight_3d(ax, res, figure_eight, scenario.outage,
                       title=f"{name}\n{rep['verdict']} (max {rep['max_error_m']:.2f} m)")
    fig.suptitle(f"Week 4: 3D view, seed {scenario.seed}, "
                 f"GPS lost {scenario.outage[0]:.0f}-{scenario.outage[1]:.0f} s")
    fig.tight_layout()
    fig.savefig(out / "flight3d_demo.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
