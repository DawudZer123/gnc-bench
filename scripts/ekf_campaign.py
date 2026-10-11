"""Step 2: does the full EKF hold up over 100 random flights?

Both versions use optical flow and the wind-rejection controller. The only difference
is the navigation: attitude borrowed from the simulation (KF) or estimated (EKF).

Run: python scripts/ekf_campaign.py        (100 flights per version)
     python scripts/ekf_campaign.py 20     (quick test)
The report is written to outputs/ekf_report.html.
"""

import sys
from pathlib import Path

from gncbench.campaign import run_campaign, summarize
from gncbench.report import write_report
from gncbench.scenario import wind_rejection_controller

CORRIDOR = 0.5   # m
VERSIONS = {"KF (true attitude)": "kf", "EKF (estimated attitude)": "ekf"}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)

    results = {}
    for name, navigation in VERSIONS.items():
        print(name)
        results[name] = rows = run_campaign(n, controller=wind_rejection_controller,
                                            corridor=CORRIDOR, navigation=navigation)
        s = summarize(rows)
        print(f"  pass rate           : {s['pass_rate'] * 100:.0f} %")
        print(f"  median max error    : {s['median_max_error_m']:.2f} m")
        print(f"  95th pct max error  : {s['p95_max_error_m']:.2f} m")
        print(f"  worst flight        : {s['worst_max_error_m']:.2f} m (seed {s['worst_seed']})\n")

    report = write_report(out / "ekf_report.html", results, CORRIDOR,
                          title="GNC Bench: KF vs full EKF")
    print(f"Report saved: {report}")


if __name__ == "__main__":
    main()
