"""Week 4: Monte Carlo campaign. Version A vs Version B over many random flights.

Each flight draws its own wind (speed, direction, gusts), GPS outage (start, length)
and mass error (±10%). Both versions fly exactly the same scenarios.

The bench writes outputs/campaign_report.html: open it in a browser.

Run: python scripts/campaign_demo.py        (100 flights per version)
     python scripts/campaign_demo.py 20     (quick test)
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.campaign import run_campaign, summarize
from gncbench.report import write_report

CORRIDOR = 0.5   # m
VERSIONS = {"Version A: IMU + baro": ("imu", "gps", "baro"),
            "Version B: + optical flow": ("imu", "gps", "baro", "flow")}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)

    results = {}
    for name, sensors in VERSIONS.items():
        print(name)
        rows = run_campaign(n, sensors=sensors, corridor=CORRIDOR)
        results[name] = rows
        s = summarize(rows)
        print(f"  pass rate           : {s['pass_rate'] * 100:.0f} %")
        print(f"  median max error    : {s['median_max_error_m']:.2f} m")
        print(f"  95th pct max error  : {s['p95_max_error_m']:.2f} m")
        print(f"  worst flight        : {s['worst_max_error_m']:.2f} m (seed {s['worst_seed']})")
        print(f"  control effort (RMS): {s['mean_control_effort_N']:.3f} N\n")
        # Every flight is saved: open the CSV in Excel to dig into the failures.
        with open(out / f"campaign_{'AB'[len(results) - 1]}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(0, max(max(r["max_error_m"] for r in rows) for rows in results.values()), 30)
    for (name, rows), color in zip(results.items(), ("C3", "C2")):
        max_err = [r["max_error_m"] for r in rows]
        wind = [r["wind_speed"] for r in rows]
        outage = [r["outage_length"] for r in rows]
        ax[0].hist(max_err, bins=bins, color=color, alpha=0.6, label=name)
        ax[1].scatter(outage, max_err, s=[8 + 8 * w for w in wind], color=color, alpha=0.6,
                      label=name)
    ax[0].axvline(CORRIDOR, color="k", ls=":", lw=1, label="corridor limit")
    ax[1].axhline(CORRIDOR, color="k", ls=":", lw=1, label="corridor limit")
    ax[0].set_xlabel("Max error during the outage (m)")
    ax[0].set_ylabel("Number of flights")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("GPS outage length (s)")
    ax[1].set_ylabel("Max error during the outage (m)")
    ax[1].set_title("Dot size = wind speed", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Week 4: Monte Carlo campaign, {n} random flights per version")
    fig.tight_layout()
    fig.savefig(out / "campaign_demo.png", dpi=120)
    report = write_report(out / "campaign_report.html", results, CORRIDOR, figures=[fig])
    print(f"Report saved: {report}  (open it in your browser)")
    plt.show()


if __name__ == "__main__":
    main()
