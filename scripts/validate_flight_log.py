"""Step 4: check the bench's models against a REAL flight.

Run: python scripts/validate_flight_log.py my_flight.ulg     (a PX4 log, needs: pip install pyulog)
     python scripts/validate_flight_log.py                   (self-check on our own simulator)

Three checks, each skipped if the log does not allow it:
  1. Sensor noise   real sensors at rest vs the noise in our sensor models;
  2. EKF replay     our EKF on the real sensor data vs PX4's own EKF (EKF2);
  3. Drag           drag coefficient seen in flight vs our dynamics model (0.3 N·s/m / 1 kg).
Plots are saved to outputs/validation.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, initial_state
from gncbench.flightlog import (compare_with_px4, fit_drag, from_simulation, load_px4_log,
                                replay_ekf, sensor_noise)
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer
from gncbench.sim import simulate
from gncbench.trajectories import figure_eight
from gncbench.wind import Wind

MODEL = {"accel_std": IMU.__init__.__defaults__[1], "gyro_std": IMU.__init__.__defaults__[2],
         "baro_std": Barometer.__init__.__defaults__[1]}


def simulated_log(params):
    """Our own figure-8 in light wind, packed like a real log (for the self-check)."""
    rng = np.random.default_rng(0)
    x0 = initial_state(position=figure_eight(0)["position"])
    x0[MOTORS] = params.mass * params.gravity / 4
    res = simulate(x0, HoverController(params, dt=0.004, ki=1.0), figure_eight, params,
                   duration=30.0, sensors=[IMU(rng), GPS(rng), Barometer(rng)],
                   wind=Wind(np.random.default_rng(1), mean=(1.0, 0.0, 0.0), gust_std=0.3))
    return from_simulation(res)


def main():
    params = QuadParams()
    if len(sys.argv) > 1:
        print(f"Reading {sys.argv[1]}")
        log = load_px4_log(sys.argv[1])
    else:
        print("No log given: self-check on our own simulator (the answers are known).")
        log = simulated_log(params)
    ref = "PX4" if len(sys.argv) > 1 else "truth"   # what our EKF is compared with
    duration = log.imu_t[-1] - log.imu_t[0]
    flying = log.flying(log.imu_t)
    print(f"{duration:.0f} s of data, {flying.mean() * duration:.0f} s in the air, "
          f"{len(log.gps_t)} GPS fixes\n")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Sensor noise.
    print("1. SENSOR NOISE AT REST (real vs our model)")
    noise = {}
    try:
        noise = sensor_noise(log)
        a, b = noise["window_s"]
        print(f"   still from {a:.0f} to {b:.0f} s")
        for key, label, unit in [("accel_std", "accelerometer", "m/s²"),
                                 ("gyro_std", "gyroscope", "rad/s"), ("baro_std", "barometer", "m")]:
            ratio = noise[key] / MODEL[key]
            print(f"   {label:14s} {noise[key]:.4f} {unit:6s} model {MODEL[key]:.4f}  (x{ratio:.1f})")
        if noise["gps_pos_std"] is not None:
            print(f"   GPS position spread {np.round(noise['gps_pos_std'], 2)} m, "
                  f"fix-to-fix jumps {np.round(noise['gps_step_std'], 2)} m")
            print("   (white noise like our model would make both equal; a real GPS wanders slowly)")
    except ValueError as e:
        print(f"   skipped: {e}")

    # 2. EKF replay.
    print(f"\n2. OUR EKF ON THE SENSOR DATA vs {ref}")
    if len(log.gps_t) > 10:
        t, pos, vel, quat, ekf = replay_ekf(log, params, noise=noise)
        c = compare_with_px4(log, t, pos, vel, quat)
        print(f"   position difference {c['pos_rms_m']:.2f} m RMS "
              f"(x {np.sqrt(np.mean(c['dpos'][:, 0] ** 2)):.2f}, "
              f"y {np.sqrt(np.mean(c['dpos'][:, 1] ** 2)):.2f}, "
              f"z {np.sqrt(np.mean(c['dpos'][:, 2] ** 2)):.2f})")
        print(f"   velocity difference {c['vel_rms_m_s']:.2f} m/s RMS")
        print(f"   attitude difference roll {c['att_rms_deg'][0]:.2f}°, "
              f"pitch {c['att_rms_deg'][1]:.2f}°, yaw {c['att_rms_deg'][2]:.2f}° RMS")
        rejected = {k: f"{100 * h['rejected'] / max(1, sum(h.values())):.1f} %"
                    for k, h in ekf.health.items()}
        print(f"   measurements rejected by our χ² test: {rejected}")
        for i, (axis, color) in enumerate(zip("xyz", ("C0", "C1", "C2"))):
            ax[0, 0].plot(log.ref_t, log.ref_pos[:, i], color=color, lw=2.5, alpha=0.35)
            ax[0, 0].plot(t, pos[:, i], color=color, lw=1, label=f"{axis}: ours (thin) vs {ref} (wide)")
        for i, (axis, color) in enumerate(zip(["roll", "pitch", "yaw"], ("C0", "C1", "C2"))):
            ax[0, 1].plot(c["att_t"], c["datt_deg"][:, i], color=color, lw=0.8, label=axis)
    else:
        print("   skipped: not enough GPS in this log")
    ax[0, 0].set_title(f"Position: our EKF vs {ref}", fontsize=10)
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Position (m, ENU)")
    ax[0, 1].set_title(f"Attitude: ours minus {ref} (in flight)", fontsize=10)
    ax[0, 1].set_xlabel("Time (s)")
    ax[0, 1].set_ylabel("Difference (°)")

    # 3. Drag.
    print("\n3. DRAG IN FLIGHT vs OUR MODEL")
    model_k = params.drag_coeff / params.mass
    try:
        d = fit_drag(log)
        for axis in "xy":
            if d[f"k_{axis}"] is None:
                print(f"   body {axis}: not enough motion along this axis to measure drag")
            else:
                print(f"   body {axis}: k = drag / mass = {d[f'k_{axis}']:.2f} 1/s "
                      f"(R² {d[f'r2_{axis}']:.2f}), our model {model_k:.2f} 1/s")
        print(f"   ({d['samples']} samples while moving; R² near 1 = linear drag explains the data)")
        for i, (axis, color) in enumerate(zip("xy", ("C0", "C1"))):
            if d[f"k_{axis}"] is not None:
                ax[1, 0].plot(d["v_body"][:, i], d["f_body"][:, i], ".", ms=3, color=color,
                              label=f"body {axis}: k = {d[f'k_{axis}']:.2f}")
        v = np.linspace(*ax[1, 0].get_xlim(), 2)
        ax[1, 0].plot(v, -model_k * v, "k--", lw=1, label=f"our model: k = {model_k:.2f}")
    except ValueError as e:
        print(f"   skipped: {e}")
    ax[1, 0].set_title("Drag: horizontal accelerometer vs body velocity", fontsize=10)
    ax[1, 0].set_xlabel("Body velocity (m/s)")
    ax[1, 0].set_ylabel("Accelerometer (m/s²)")

    if noise:
        still = slice(*np.searchsorted(log.imu_t, noise["window_s"]))
        ax[1, 1].hist(log.imu[still, 0] - log.imu[still, 0].mean(), bins=60, density=True,
                      alpha=0.7, label="real accelerometer x at rest")
    x = np.linspace(-4 * MODEL["accel_std"], 4 * MODEL["accel_std"], 200)
    ax[1, 1].plot(x, np.exp(-x**2 / (2 * MODEL["accel_std"] ** 2)) / (MODEL["accel_std"] * np.sqrt(2 * np.pi)),
                  "k--", lw=1, label="our model")
    ax[1, 1].set_title("Accelerometer noise at rest", fontsize=10)
    ax[1, 1].set_xlabel("m/s²")
    ax[1, 1].legend(fontsize=7)

    for a in ax.flat:
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.suptitle(f"Step 4: validation on {Path(sys.argv[1]).name if len(sys.argv) > 1 else 'our simulator'}")
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "validation.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
