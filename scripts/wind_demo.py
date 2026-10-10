"""Week 2: can the drone hold its position in the wind? Comparison with and without integral.

Run: python scripts/wind_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, initial_state
from gncbench.params import QuadParams
from gncbench.sim import simulate
from gncbench.wind import Wind


def main():
    params = QuadParams()
    target = {"position": np.array([0.0, 0.0, 2.0])}
    x0 = initial_state(position=target["position"])
    x0[MOTORS] = params.mass * params.gravity / 4    # start already hovering

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ki, label in [(0.0, "without integral (ki = 0)"), (2.0, "with integral (ki = 2)")]:
        # Same seed: both controllers face exactly the same gusts.
        wind = Wind(np.random.default_rng(7), mean=(5.0, 0.0, 0.0), gust_std=2.0)
        res = simulate(x0, HoverController(params, dt=0.004, ki=ki), target, params,
                       duration=20.0, wind=wind)
        error = np.linalg.norm(res.states[:, POS] - target["position"], axis=1)
        ax[1].plot(res.t, error * 100, label=label)
        print(f"{label}: mean error {error[len(error) // 2:].mean() * 100:.1f} cm (last 10 s)")

    ax[0].plot(res.t, res.wind[:, 0], label="wind x")
    ax[0].plot(res.t, res.wind[:, 1], label="wind y")
    ax[0].set_ylabel("Wind (m/s)")
    ax[0].legend()
    ax[1].set_ylabel("Position error (cm)")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend()
    fig.suptitle("Week 2: 5 m/s mean wind + gusts")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "wind_demo.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
