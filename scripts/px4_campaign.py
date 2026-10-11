"""Monte Carlo on the real PX4: the same random flights with and without optical flow.

Each flight draws its wind, gusts and GPS outage from its seed. Every flight is a fresh PX4
instance; several run in parallel.

Run: python scripts/px4_campaign.py [number of flights] [parallel PX4 instances]
     (default: 100 flights per version, 4 instances)
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.px4campaign import Px4Scenario, run_px4_campaign, summarize_px4

CORRIDOR = 1.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    versions = {"A: PX4, GPS only": False, "B: PX4 + optical flow": True}
    results = {}
    for name, flow in versions.items():
        print(f"\n{name}: {n} flights")
        results[name] = run_px4_campaign([Px4Scenario.random(s, flow=flow) for s in range(1, n + 1)],
                                         workers=workers, corridor=CORRIDOR)

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    print(f"\n{'':24}{'pass':>7}{'completed':>11}{'median max':>12}{'worst':>8}  failsafes")
    for name, rows in results.items():
        s = summarize_px4(rows)
        print(f"{name:24}{s['pass_rate']:>7.0%}{s['completed_rate']:>11.0%}"
              f"{s['median_max_error_m']:>10.2f} m{s['worst_max_error_m']:>6.2f} m"
              f"  {s['failsafes']} (drift after failsafe {s['mean_failsafe_drift_m']:.1f} m)"
              + (f"  [{s['errors']} errors]" if s["errors"] else ""))
        with open(out / f"px4_campaign_{'B' if versions[name] else 'A'}.csv", "w", newline="") as fh:
            keys = list(dict.fromkeys(k for r in rows for k in r if k != "messages"))
            writer = csv.DictWriter(fh, keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    plot_results(results, n, out / "px4_campaign.png")
    print(f"\nPlot saved: {out / 'px4_campaign.png'}; flights in outputs/px4_campaign_A.csv and _B.csv")
    plt.show()


def plot_results(results, n, path):
    """Left: distribution of the worst error. Right: worst error vs GPS outage length."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(0, 6, 31)
    for (name, rows), color in zip(results.items(), ("C3", "C2")):
        ok = [r for r in rows if r["verdict"] != "ERROR"]
        worst = np.array([min(float(r["max_error_m"]), 6.0) for r in ok])
        outage = np.array([float(r["fault_length"]) for r in ok])
        failsafe = np.array([bool(r["failsafe"]) for r in ok])
        left.hist(worst, bins, color=color, alpha=0.6, label=name)
        right.scatter(outage[~failsafe], worst[~failsafe], color=color, s=16,
                      label=f"{name}: mission completed")
        right.scatter(outage[failsafe], worst[failsafe], color=color, marker="x", s=28,
                      label=f"{name}: PX4 failsafe")
    left.axvline(CORRIDOR, color="k", ls=":", label="corridor")
    right.axhline(CORRIDOR, color="k", ls=":", lw=1)
    left.set_xlabel("Max distance from mission (m, capped at 6)")
    left.set_ylabel("Flights")
    left.legend(fontsize=8)
    right.set_xlabel("GPS outage length (s)")
    right.set_ylabel("Max distance from mission (m, capped at 6)")
    right.legend(fontsize=7, loc="center right")
    fig.suptitle(f"PX4 on our physics: {n} random flights per version")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
