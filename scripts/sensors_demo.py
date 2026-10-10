"""Semaine 2 : ce que « voit » le drone, comparé à la réalité.

Lancer : python scripts/sensors_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import POS, initial_state
from gncbench.math3d import quat_from_euler
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer
from gncbench.sim import simulate


def main():
    params = QuadParams()
    rng = np.random.default_rng(42)          # graine fixe : mêmes résultats à chaque lancement
    x0 = initial_state(position=(0, 0, 0), quat=quat_from_euler(0.3, -0.2, 0.0))
    target = {"position": np.array([1.0, -0.5, 2.0]), "yaw": 0.5}

    res = simulate(x0, HoverController(params, dt=0.004), target, params, duration=10.0,
                   sensors=[IMU(rng), GPS(rng), Barometer(rng)])


    gps_t, gps = res.measurements["gps"]
    imu_t, imu = res.measurements["imu"]

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, name in enumerate("xyz"):
        ax[0].plot(res.t, res.states[:, POS][:, i], color=f"C{i}", label=f"{name} vrai")
        ax[0].plot(gps_t, gps[:, i], ".", ms=3, color=f"C{i}", label=f"{name} GPS")
    baro_t, baro = res.measurements["baro"]
    ax[0].plot(baro_t, baro[:, 0], ".", ms=2, color="k", label="z baro")
    ax[0].set_ylabel("Position (m)")
    ax[0].legend(ncol=3, fontsize=8)

    for i, name in enumerate(["ax", "ay", "az"]):
        ax[1].plot(imu_t, imu[:, i], lw=0.6, label=name)
    ax[1].set_ylabel("Accéléromètre (m/s²)")
    ax[1].legend(fontsize=8)

    for i, name in enumerate(["gx", "gy", "gz"]):
        ax[2].plot(imu_t, imu[:, 3 + i], lw=0.6, label=name)
    ax[2].set_ylabel("Gyroscope (rad/s)")
    ax[2].set_xlabel("Temps (s)")
    ax[2].legend(fontsize=8)

    fig.suptitle("Semaine 2 : mesures des capteurs")
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "sensors_demo.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
