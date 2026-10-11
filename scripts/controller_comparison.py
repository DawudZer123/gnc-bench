"""Plug-in demo: two controllers, same 100 random flights, one report.

Baseline:       cascaded PID + geometric attitude (the controller used so far).
Wind rejection: same controller + compensation of the wind measured by the filter.
Both fly with IMU + GPS + baro + optical flow, through exactly the same scenarios.

Run: python scripts/controller_comparison.py        (100 flights per controller)
     python scripts/controller_comparison.py 20     (quick test)
The report is written to outputs/controller_report.html.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.campaign import run_campaign, summarize
from gncbench.report import write_report
from gncbench.scenario import default_controller, wind_rejection_controller

CORRIDOR = 0.5   # m
CONTROLLERS = {"Baseline PID": default_controller,
               "Wind rejection": wind_rejection_controller}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)

    results = {}
    for name, controller in CONTROLLERS.items():
        print(name)
        results[name] = rows = run_campaign(n, controller=controller, corridor=CORRIDOR)
        s = summarize(rows)
        print(f"  pass rate           : {s['pass_rate'] * 100:.0f} %")
        print(f"  median max error    : {s['median_max_error_m']:.2f} m")
        print(f"  95th pct max error  : {s['p95_max_error_m']:.2f} m")
        print(f"  worst flight        : {s['worst_max_error_m']:.2f} m (seed {s['worst_seed']})")
        print(f"  control effort (RMS): {s['mean_control_effort_N']:.3f} N\n")

    base, rej = (np.array([r["max_error_m"] for r in rows]) for rows in results.values())
    gusts = np.array([r["gust_std"] for r in results["Baseline PID"]])
    print(f"Wind rejection is better on {np.mean(rej < base) * 100:.0f} % of the flights")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    dots = ax[0].scatter(base, rej, s=14, c=gusts, cmap="viridis")
    fig.colorbar(dots, ax=ax[0], label="Gust strength (m/s)")
    lim = max(base.max(), rej.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8, label="same error")
    ax[0].set_xlim(0, lim)
    ax[0].set_ylim(0, lim)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("Baseline PID: max error (m)")
    ax[0].set_ylabel("Wind rejection: max error (m)")
    ax[0].set_title("One dot = one flight (below the line: wind rejection wins)", fontsize=9)
    ax[0].legend(fontsize=8)
    for (name, rows), color in zip(results.items(), ("C0", "C1")):
        g = [r["gust_std"] for r in rows]
        e = [r["max_error_m"] for r in rows]
        ax[1].scatter(g, e, s=14, color=color, alpha=0.7, label=name)
    ax[1].axhline(CORRIDOR, color="k", ls=":", lw=1, label="corridor limit")
    ax[1].set_xlabel("Gust strength (m/s)")
    ax[1].set_ylabel("Max error during the outage (m)")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Controller comparison on the same {n} random flights")
    fig.tight_layout()
    fig.savefig(out / "controller_comparison.png", dpi=120)
    report = write_report(out / "controller_report.html", results, CORRIDOR, figures=[fig],
                          title="GNC Bench: controller comparison")
    print(f"Report saved: {report}")
    plt.show()


if __name__ == "__main__":
    main()
