"""Step 3: can the filter tell that the GPS is lying?

Final-demo flight (figure-8, wind), but from 10 to 30 s the GPS does not go silent:
it goes WRONG. Each fault is flown twice with the full EKF + optical flow:
without fault detection (it believes everything) and with it (stale check + χ² test).

Run: python scripts/fault_detection_demo.py
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.scenario import Scenario, wind_rejection_controller

FAULTS = {"frozen": ("freeze", ()),
          "jump 5 m": ("offset", (5.0, 0.0, 0.0)),
          "drift 0.3 m/s": ("drift", (0.3, 0.0, 0.0)),
          "drift 0.1 m/s": ("drift", (0.1, 0.0, 0.0))}
NAVIGATION = {"no detection": "ekf_naive", "detection": "ekf"}


def main():
    base = Scenario()
    outage = base.outage
    results = {}
    print(f"GPS faulty from {outage[0]:.0f} to {outage[1]:.0f} s, ±0.5 m corridor\n")
    print(f"{'GPS fault':15s}" + "".join(f"{n:>22s}" for n in NAVIGATION))
    for fault, (mode, value) in FAULTS.items():
        scenario = replace(base, gps_fault=mode, gps_fault_value=value)
        row = f"{fault:15s}"
        for name, navigation in NAVIGATION.items():
            res, rep = scenario.fly(controller=wind_rejection_controller, navigation=navigation)
            results[fault, name] = (res, rep)
            row += f"{rep['max_error_m']:15.2f} m {rep['verdict']:>4s}"
        print(row)

    # Bonus: a frozen GPS without optical flow and without detection.
    frozen = replace(base, gps_fault="freeze")
    _, rep = frozen.fly(controller=wind_rejection_controller, navigation="ekf_naive",
                        sensors=("imu", "gps", "baro"))
    print(f"\nFrozen GPS, no optical flow, no detection: {rep['max_error_m']:.0f} m (fly-away)")

    fig, ax = plt.subplots(2, 1, figsize=(9, 8))
    x = np.arange(len(FAULTS))
    for i, (name, color) in enumerate(zip(NAVIGATION, ("C3", "C2"))):
        errs = [results[f, name][1]["max_error_m"] for f in FAULTS]
        ax[0].bar(x + (i - 0.5) * 0.38, errs, width=0.38, color=color, label=name)
    ax[0].axhline(0.5, color="k", ls=":", lw=1, label="corridor limit")
    ax[0].set_xticks(x, list(FAULTS))
    ax[0].set_ylabel("Max error during the fault (m)")
    ax[0].legend(fontsize=8)

    # Zoom on the 5 m jump: what the GPS says, the truth, and both estimates.
    res_n = results["jump 5 m", "no detection"][0]
    res_d = results["jump 5 m", "detection"][0]
    gps_t, gps = res_d.measurements["gps"]
    ax[1].axvspan(*outage, color="red", alpha=0.08, label="GPS lying (+5 m in x)")
    ax[1].plot(gps_t, gps[:, 0], ".", ms=3, color="C0", label="GPS x")
    ax[1].plot(res_d.t, res_d.states[:, 0], "k", lw=1.5, label="true x (with detection)")
    ax[1].plot(res_n.t, res_n.states[:, 0], color="C3", lw=1, label="true x (no detection)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("x (m)")
    ax[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Step 3: GPS fault detection (full EKF + optical flow)")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fault_detection_demo.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
