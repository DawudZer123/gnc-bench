"""OPTION B, step 2: does the twin behave like the real drone?

Replays the real flight's setpoints on PX4 SITL twice, with the drone's own PX4 tuning:
  - on the generic Iris model PX4 ships with,
  - on the digital twin built by scripts/build_twin.py,
and compares both with what the real drone did (throttle over time, motor balance, path).

Run: python scripts/twin_replay.py my_flight.ulg [--mass 1.8] [--arm 0.24]
     (needs outputs/twin.json from build_twin.py, and PX4 SITL as for scripts/px4_demo.py)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.px4flight import fly_px4
from gncbench.twin import DigitalTwin, logged_mission, read_real_flight

OUT = Path(__file__).resolve().parent.parent / "outputs"


def fly(args):
    which, log, mass, arm, instance = args
    twin = DigitalTwin.load(OUT / "twin.json")
    mission = logged_mission(read_real_flight(log))
    if which == "twin":
        return fly_px4(mission=mission, params=twin.quad_params(mass, arm),
                       px4_params=twin.px4_params, noise=twin.sensor_noise(),
                       battery=twin.battery_model(), instance=instance)
    return fly_px4(mission=mission, px4_params=twin.px4_params, instance=instance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--mass", type=float, default=1.8)
    ap.add_argument("--arm", type=float, default=0.24)
    args = ap.parse_args()
    real = read_real_flight(args.log)
    print("Replaying the real mission on PX4 SITL: generic Iris and digital twin, in parallel...")
    with ProcessPoolExecutor(2) as pool:
        iris, twin = pool.map(fly, [("iris", args.log, args.mass, args.arm, 0),
                                    ("twin", args.log, args.mass, args.arm, 1)])

    # Compare in flight, away from take-off and landing transients.
    air = (real.t > 30) & (real.t < real.setpoint_t[-1] - 10)
    real_motor = real.motors[(real.motors_t > 30) & (real.motors_t < real.setpoint_t[-1] - 10)]
    rows = {"real drone": (real.t[air], real.thrust[air], real_motor.mean(axis=0))}
    for name, f in (("generic Iris", iris), ("digital twin", twin)):
        t = f.t - f.armed_at
        m = (t > 30) & (t < real.setpoint_t[-1] - 10)
        rows[name] = (t[m], f.commands[m].mean(axis=1), f.commands[m].mean(axis=0))

    real_curve = rows["real drone"]
    print(f"\n{'':14}{'throttle':>10}{'start->end':>14}{'motor balance (FR, RL, FL, RR)':>36}")
    for name, (t, thr, motors) in rows.items():
        early, late = thr[t < t.min() + 20].mean(), thr[t > t.max() - 20].mean()
        print(f"{name:14}{thr.mean():>10.2f}{early:>8.2f}->{late:.2f}    {np.round(motors, 2)}")
    for name in ("generic Iris", "digital twin"):
        t, thr, motors = rows[name]
        err = np.interp(real_curve[0], t, thr) - real_curve[1]
        print(f"{name}: throttle error {np.sqrt(np.mean(err ** 2)):.3f} RMS, "
              f"motor balance error {np.abs(motors - real_curve[2]).mean():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {"real drone": "k", "generic Iris": "C3", "digital twin": "C2"}
    for name, (t, thr, motors) in rows.items():
        k = max(len(t) // 400, 1)
        axes[0].plot(t[::k], thr[::k], color=colors[name], lw=1, label=name)
    axes[0].set_xlabel("Time since take-off (s)")
    axes[0].set_ylabel("Throttle (collective command)")
    axes[0].set_title("Throttle: battery fade and weight")
    axes[0].legend(fontsize=8)
    x = np.arange(4)
    for j, (name, (_, _, motors)) in enumerate(rows.items()):
        axes[1].bar(x + (j - 1) * 0.27, motors, 0.27, color=colors[name], label=name)
    axes[1].set_xticks(x, ["front-right", "rear-left", "front-left", "rear-right"], fontsize=8)
    axes[1].set_ylim(0.3, 0.9)
    axes[1].set_title("Motor balance in flight")
    axes[1].legend(fontsize=8)
    axes[2].plot(real.position[:, 0], real.position[:, 1], "k", label="real drone (its estimate)")
    for name, f in (("generic Iris", iris), ("digital twin", twin)):
        axes[2].plot(f.states[:, 0], f.states[:, 1], color=colors[name], lw=1, label=name)
    axes[2].set_xlabel("East (m)")
    axes[2].set_ylabel("North (m)")
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].set_title("Path")
    axes[2].legend(fontsize=8)
    fig.suptitle("Real flight replayed on PX4 SITL with the drone's own tuning")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "twin_replay.png", dpi=120)
    print(f"\nPlot saved: {OUT / 'twin_replay.png'}")
    plt.show()


if __name__ == "__main__":
    main()
