"""Démo semaine 1 : décollage depuis une position inclinée et vol stationnaire.

Lancer : python scripts/hover_demo.py   (ou F5 dans VS Code)
Le graphique est enregistré dans outputs/hover_demo.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import POS, QUAT, initial_state, MOTORS
from gncbench.math3d import quat_from_euler, quat_to_euler
from gncbench.params import QuadParams
from gncbench.sim import simulate


def main():
    params = QuadParams()
    x0 = initial_state(position=(0, 0, 0), quat=quat_from_euler(0.3, -0.2, 0.0))
    target = {"position": np.array([1.0, -0.5, 2.0]), "yaw": 0.5}

    res = simulate(x0, HoverController(params, dt=0.004), target, params, duration=10.0)

    pos = res.states[:, POS]
    euler = np.degrees(np.array([quat_to_euler(q) for q in res.states[:, QUAT]]))
    err = np.linalg.norm(pos[-1] - target["position"])
    print(f"Position finale : {np.round(pos[-1], 3)}  (erreur {err * 100:.2f} cm)")

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, name in enumerate("xyz"):
        ax[0].plot(res.t, pos[:, i], label=name)
        ax[0].axhline(target["position"][i], ls="--", lw=0.8, color=f"C{i}")
    ax[0].set_ylabel("Position (m)")
    ax[0].legend()
    for i, name in enumerate(["roulis", "tangage", "lacet"]):
        ax[1].plot(res.t, euler[:, i], label=name)
    ax[1].set_ylabel("Attitude (°)")
    ax[1].legend()
    for i in range(4):
        ax[2].plot(res.t, res.rotors[:, i], lw=0.8, ls="--", color=f"C{i}")
        ax[2].plot(res.t, res.states[:, MOTORS][:, i], color=f"C{i}", label=f"rotor {i}")
    ax[2].legend()                # poussée réelle
    ax[2].set_ylabel("Poussée rotors (N)")
    ax[2].set_xlabel("Temps (s)")
    fig.suptitle("Semaine 1 : vol stationnaire")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "hover_demo.png", dpi=120)
    print(f"Graphique enregistré : {out / 'hover_demo.png'}")
    plt.show()


if __name__ == "__main__":
    main()
