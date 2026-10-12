"""OPTION B, step 3: how would YOUR drone cope with losing GPS?

The same random GPS-loss flights as scripts/px4_campaign.py, flown by PX4 with your drone's
tuning on your digital twin (outputs/twin.json from build_twin.py): its weight margin, battery,
uneven motors and in-flight vibration. Then the same flights with an optical flow sensor added.

The twin's inertia is a guess when the log had no sharp manoeuvres (build_twin.py says so).
--inertia repeats the flow campaign with the inertia scaled, to see if the answer depends on it.

Run: python scripts/twin_campaign.py [flights per version] [parallel PX4 instances]
         [--inertia 0.5 2] [--mass 1.8] [--arm 0.24]
     (default: 100 flights, 4 instances, no inertia sweep)
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.px4campaign import Px4Scenario, run_px4_campaign, summarize_px4

CORRIDOR = 1.0
OUT = Path(__file__).resolve().parent.parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flights", type=int, nargs="?", default=100)
    ap.add_argument("workers", type=int, nargs="?", default=4)
    ap.add_argument("--inertia", type=float, nargs="*", default=[])
    ap.add_argument("--mass", type=float, default=1.8)
    ap.add_argument("--arm", type=float, default=0.24)
    args = ap.parse_args()

    twin = str(OUT / "twin.json")
    drone = {"twin": twin, "mass": args.mass, "arm": args.arm}
    versions = {"your drone, GPS only": {"flow": False},
                "your drone + optical flow": {"flow": True}}
    for scale in args.inertia:
        versions[f"+ flow, inertia x{scale:g}"] = {"flow": True, "inertia_scale": scale}

    seeds = range(1, args.flights + 1)
    results = {}
    for name, extra in versions.items():
        print(f"\n{name}: {args.flights} flights")
        results[name] = run_px4_campaign([Px4Scenario.random(s, **drone, **extra) for s in seeds],
                                         workers=args.workers, corridor=CORRIDOR)

    OUT.mkdir(exist_ok=True)
    print(f"\n{'':28}{'pass':>7}{'completed':>11}{'median max':>12}{'worst':>8}  failsafes")
    rows_out = []
    for name, rows in results.items():
        s = summarize_px4(rows)
        print(f"{name:28}{s['pass_rate']:>7.0%}{s['completed_rate']:>11.0%}"
              f"{s['median_max_error_m']:>10.2f} m{s['worst_max_error_m']:>6.2f} m"
              f"  {s['failsafes']} (drift after failsafe {s['mean_failsafe_drift_m']:.1f} m)"
              + (f"  [{s['errors']} errors]" if s["errors"] else ""))
        rows_out += [{"version": name, **r} for r in rows]
    with open(OUT / "twin_campaign.csv", "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows_out for k in r if k != "messages"))
        writer = csv.DictWriter(fh, keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    plot_results(results, args.flights, OUT / "twin_campaign.png")
    print(f"\nPlot saved: {OUT / 'twin_campaign.png'}; flights in outputs/twin_campaign.csv")
    plt.show()


def plot_results(results, n, path):
    """Left: worst error vs GPS outage length. Right: pass rate of each version."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(results)
    for (name, rows), color in zip(results.items(), ("C3", "C2", "C0", "C1", "C4")):
        ok = [r for r in rows if r["verdict"] != "ERROR"]
        if name.startswith("+"):          # inertia sweep: only on the bar chart
            continue
        worst = np.array([min(float(r["max_error_m"]), 6.0) for r in ok])
        outage = np.array([float(r["fault_length"]) for r in ok])
        failsafe = np.array([bool(r["failsafe"]) for r in ok])
        left.scatter(outage[~failsafe], worst[~failsafe], color=color, s=16,
                     label=f"{name}: mission completed")
        left.scatter(outage[failsafe], worst[failsafe], color=color, marker="x", s=28,
                     label=f"{name}: PX4 failsafe")
    left.axhline(CORRIDOR, color="k", ls=":", lw=1)
    left.set_xlabel("GPS outage length (s)")
    left.set_ylabel("Max distance from mission (m, capped at 6)")
    left.legend(fontsize=7, loc="center right")
    rates = [100 * summarize_px4(results[k])["pass_rate"] for k in names]
    bars = right.bar(range(len(names)), rates, color=["C3", "C2", "C0", "C1", "C4"][:len(names)])
    right.bar_label(bars, fmt="%.0f %%")
    right.set_xticks(range(len(names)), names, fontsize=8, rotation=12)
    right.set_ylim(0, 110)
    right.set_ylabel(f"Flights within {CORRIDOR:g} m (%)")
    fig.suptitle(f"Your drone's twin, flown by PX4 with your tuning: {n} random GPS-loss flights")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
