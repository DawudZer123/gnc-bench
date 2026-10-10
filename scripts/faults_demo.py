"""Semaine 2 : le GPS tombe en panne pendant 4 s. Que peut-on encore savoir de la position ?

Pendant la panne, on essaie la méthode naïve : intégrer l'accéléromètre deux fois
à partir du dernier point GPS (« navigation à l'estime »). On voit la dérive :
c'est exactement le problème que le filtre de Kalman de la semaine 3 doit résoudre.

Lancer : python scripts/faults_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, QUAT, initial_state
from gncbench.faults import SensorFault
from gncbench.math3d import quat_to_rotmat
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU
from gncbench.sim import simulate

OUTAGE = (6.0, 10.0)   # s


def circle(t):
    """Cercle de 2 m de rayon à 2 m d'altitude, un tour en 12 s."""
    w = 2 * np.pi / 12
    return {"position": np.array([2 * np.cos(w * t) - 2, 2 * np.sin(w * t), 2.0])}


def dead_reckoning(res, params):
    """Double intégration de l'IMU pendant la panne, à partir du dernier point GPS valide."""
    gps_t, gps = res.measurements["gps"]
    imu_t, imu = res.measurements["imu"]
    valid = ~np.isnan(gps[:, 0])
    i0 = np.where(valid & (gps_t < OUTAGE[0]))[0][-1]
    p, v = gps[i0, :3].copy(), gps[i0, 3:].copy()
    times, path = [gps_t[i0]], [p.copy()]
    for k in np.where((imu_t > gps_t[i0]) & (imu_t < OUTAGE[1]))[0]:
        dt = imu_t[k] - imu_t[k - 1]
        q = res.states[np.searchsorted(res.t, imu_t[k]), QUAT]   # attitude supposée connue
        a = quat_to_rotmat(q) @ imu[k, :3] - np.array([0, 0, params.gravity])
        v = v + a * dt
        p = p + v * dt
        times.append(imu_t[k])
        path.append(p.copy())
    return np.array(times), np.array(path)


def main():
    params = QuadParams()
    rng = np.random.default_rng(3)
    x0 = initial_state(position=circle(0)["position"])
    x0[MOTORS] = params.mass * params.gravity / 4

    res = simulate(x0, HoverController(params, dt=0.004), circle, params, duration=14.0,
                   sensors=[IMU(rng), GPS(rng)],
                   faults=[SensorFault("gps", start=OUTAGE[0], end=OUTAGE[1])])

    gps_t, gps = res.measurements["gps"]
    dr_t, dr = dead_reckoning(res, params)
    true_at_end = res.states[np.searchsorted(res.t, dr_t[-1]), POS]
    print(f"Dérive de l'estime après {OUTAGE[1] - OUTAGE[0]:.0f} s : "
          f"{np.linalg.norm(dr[-1] - true_at_end):.2f} m")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for i, (name, a) in enumerate(zip("xy", ax)):
        a.axvspan(*OUTAGE, color="red", alpha=0.1, label="panne GPS")
        a.plot(res.t, res.states[:, POS][:, i], "k", lw=1.5, label="vrai")
        a.plot(gps_t, gps[:, i], ".", ms=4, color="C0", label="GPS")
        a.plot(dr_t, dr[:, i], color="C3", label="estime (IMU seule)")
        a.set_ylabel(f"{name} (m)")
    ax[0].legend(fontsize=8, ncol=4)
    ax[1].set_xlabel("Temps (s)")
    fig.suptitle(f"Semaine 2 : panne GPS de {OUTAGE[1] - OUTAGE[0]:.0f} s pendant un vol en cercle")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "faults_demo.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
