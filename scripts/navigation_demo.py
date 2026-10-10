"""Week 3: Kalman filter vs dead reckoning, during a 4 s GPS outage.

Both methods receive exactly the same measurements.
Run: python scripts/navigation_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from faults_demo import OUTAGE, circle, dead_reckoning

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, initial_state
from gncbench.estimator import NavigationFilter
from gncbench.faults import SensorFault
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer
from gncbench.sim import simulate


def main():
    params = QuadParams()
    rng = np.random.default_rng(3)
    x0 = initial_state(position=circle(0)["position"])
    x0[MOTORS] = params.mass * params.gravity / 4
    kf = NavigationFilter(params, p0=x0[POS])

    res = simulate(x0, HoverController(params, dt=0.004), circle, params, duration=14.0,
                   sensors=[IMU(rng), GPS(rng), Barometer(rng)],
                   faults=[SensorFault("gps", start=OUTAGE[0], end=OUTAGE[1])],
                   estimator=kf)

    gps_t, gps = res.measurements["gps"]
    dr_t, dr = dead_reckoning(res, params)
    true = res.states[:, POS]
    kf_err = np.linalg.norm(res.estimates[:, :3] - true, axis=1)
    k_end = np.searchsorted(res.t, dr_t[-1])
    ok = ~np.isnan(gps[:, 0])
    gps_err = np.linalg.norm(gps[ok, :3] - true[np.searchsorted(res.t, gps_t[ok])], axis=1)

    print(f"Mean error, GPS alone (outside outage)  : {gps_err.mean():.2f} m")
    print(f"Mean error, filter (before the outage)  : {kf_err[(res.t > 2) & (res.t < OUTAGE[0])].mean():.2f} m")
    print(f"End of outage: dead reckoning {np.linalg.norm(dr[-1] - true[k_end]):.2f} m, "
          f"filter {kf_err[k_end]:.2f} m")

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, name in enumerate("xy"):
        a = ax[i]
        a.axvspan(*OUTAGE, color="red", alpha=0.1, label="GPS outage")
        a.plot(gps_t, gps[:, i], ".", ms=3, color="C0", alpha=0.5, label="GPS")
        a.plot(res.t, true[:, i], "k", lw=1.5, label="true")
        a.plot(dr_t, dr[:, i], color="C3", label="dead reckoning (IMU only)")
        a.plot(res.t, res.estimates[:, i], color="C2", label="Kalman filter")
        a.set_ylabel(f"{name} (m)")
    ax[0].legend(fontsize=8, ncol=5)

    ax[2].axvspan(*OUTAGE, color="red", alpha=0.1)
    # On the x axis: the filter reports its own uncertainty (2σ ≈ 95% interval).
    ax[2].plot(res.t, np.abs(res.estimates[:, 0] - true[:, 0]), color="C2",
               label="actual filter error in x")
    ax[2].plot(res.t, 2 * res.estimate_std[:, 0], "--", color="C2", lw=0.8,
               label="reported uncertainty (2σ)")
    ax[2].set_ylabel("Error in x (m)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"Week 3: {OUTAGE[1] - OUTAGE[0]:.0f} s GPS outage, Kalman filter vs dead reckoning")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "navigation_demo.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
