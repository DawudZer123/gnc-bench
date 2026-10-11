"""Step 2: the full EKF estimates the attitude instead of borrowing the true one.

Same final-demo flight (figure-8, wind, GPS lost 10-30 s), flown three ways:
  KF  + flow : week 3 filter, attitude taken from the simulation (a shortcut);
  EKF + flow : attitude estimated from gyro, gravity, flow and compass (no shortcut);
  EKF, no flow : the honest Version A.

Run: python scripts/ekf_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.dynamics import QUAT
from gncbench.math3d import quat_to_euler
from gncbench.scenario import Scenario

SCENARIO = Scenario()
OUTAGE = SCENARIO.outage
RUNS = {"KF + flow (true attitude)": dict(navigation="kf"),
        "EKF + flow (estimated attitude)": dict(navigation="ekf"),
        "EKF, no flow": dict(navigation="ekf", sensors=("imu", "gps", "baro"))}


def attitude_error_deg(res):
    """Estimated minus true roll, pitch, yaw, in degrees."""
    est = np.array([quat_to_euler(q) for q in res.attitude_estimates])
    true = np.array([quat_to_euler(q) for q in res.states[:, QUAT]])
    err = est - true
    err[:, 2] = (err[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(err)


def main():
    results = {name: SCENARIO.fly(**kw) for name, kw in RUNS.items()}

    print(f"{OUTAGE[1] - OUTAGE[0]:.0f} s GPS outage, ±0.5 m corridor\n")
    for name, (res, rep) in results.items():
        print(f"{name:32s} max error {rep['max_error_m']:5.2f} m   {rep['verdict']}")
    res_ekf = results["EKF + flow (estimated attitude)"][0]
    att = attitude_error_deg(res_ekf)
    print(f"\nEKF + flow, worst attitude error after 2 s: "
          f"roll {np.abs(att[res_ekf.t > 2, 0]).max():.2f}°, "
          f"pitch {np.abs(att[res_ekf.t > 2, 1]).max():.2f}°, "
          f"yaw {np.abs(att[res_ekf.t > 2, 2]).max():.2f}°")

    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for (name, (res, rep)), color in zip(results.items(), ("C0", "C2", "C3")):
        ax[0].plot(res.t, rep["error"], color=color, label=f"{name} ({rep['verdict']})")
    ax[0].axhline(0.5, color="k", ls=":", lw=1, label="corridor limit")
    ax[0].set_ylabel("Distance from\ntrajectory (m)")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)

    for i, (name, color) in enumerate(zip(["roll", "pitch"], ("C0", "C1"))):
        ax[1].plot(res_ekf.t, att[:, i], color=color, lw=0.8, label=name)
    ax[1].set_ylabel("Attitude error (°)")
    ax[1].legend(fontsize=8)
    ax[2].plot(res_ekf.t, att[:, 2], color="C2", lw=0.8, label="yaw")
    ax[2].set_ylabel("Yaw error (°)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.axvspan(*OUTAGE, color="red", alpha=0.08)
    fig.suptitle("Step 2: full EKF, the drone flies on its estimated attitude")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "ekf_demo.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
