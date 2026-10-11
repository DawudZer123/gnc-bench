"""OPTION B, step 1: build the digital twin of YOUR drone from one of its flight logs.

Run: python scripts/build_twin.py my_flight.ulg [--mass 1.8] [--arm 0.24]
Writes outputs/twin.json (used by scripts/twin_replay.py) and outputs/twin.png.
Mass and arm only set the scale (one flight shows ratios, not kilograms): give the real
values if you know them.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gncbench.twin import build_twin, read_log, smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--mass", type=float, default=1.8)
    ap.add_argument("--arm", type=float, default=0.24)
    args = ap.parse_args()

    twin = build_twin(args.log)
    params = twin.quad_params(args.mass, args.arm)
    th, bal, rot = twin.thrust, twin.balance, twin.rotation
    print(f"Digital twin of {twin.source} ({twin.hardware}, airframe {twin.airframe})\n")
    print("THRUST")
    print(f"  hover throttle          : {th['hover_thrust']:.2f}  -> thrust/weight at full "
          f"throttle {1 / th['hover_thrust']:.2f}")
    bat = twin.battery
    v_end = bat["v0"] - bat["volts_per_mah"] * bat["drawn_mah"]
    fade = 1 - (v_end / bat["v0"]) ** th["voltage_exponent"]
    print(f"  battery effect          : thrust ∝ V^{th['voltage_exponent']:.2f}; after "
          f"{bat['drawn_mah']:.0f} mAh the same throttle lifts {100 * fade:.0f} % less")
    print(f"  per rotor (mass {args.mass} kg): {params.max_thrust:.2f} N at full battery")
    print("BALANCE")
    print(f"  motor commands in hover : {np.round(bal['motor_mean'], 2)}  (max {np.round(bal['motor_max'], 2)})")
    d = args.arm / np.sqrt(2) * 1000
    cx, cy = bal["cg_fraction"][0] * d, bal["cg_fraction"][1] * d
    print(f"  centre of gravity       : {abs(cx):.0f} mm {'forward' if cx > 0 else 'back'}, "
          f"{abs(cy):.0f} mm {'right' if cy > 0 else 'left'} of the rotor centre")
    print(f"  yaw balance             : CCW props make {100 * (bal['yaw_torque_ratio'] - 1):+.1f} % "
          f"torque vs CW (props, motors or a tilted motor)")
    print("DRAG")
    for a in "xy":
        k, r2 = twin.drag[f"k_{a}"], twin.drag[f"r2_{a}"]
        print(f"  body {a}                  : " + (f"k = {k:.2f} 1/s (R² {r2:.2f})" if k is not None
                                                  else "not enough motion"))
    print(f"  used in the twin        : k = {twin.drag_k():.2f} 1/s")
    print("ROTATION (inertia, motor lag)")
    for axis in ("roll", "pitch", "yaw"):
        fit = rot[axis]
        print(f"  {axis:24}: " + (f"{fit['gain']:.0f} rad/s² per unit command (R² {fit['r2']:.2f})"
                                  if fit else "NOT identifiable: the drone barely rotated. "
                                              "Prior used."))
    print(f"  inertia used            : {np.round(np.diag(params.inertia), 4)} kg·m²,"
          f" motor lag {params.motor_tau * 1000:.0f} ms")
    print("SENSORS IN FLIGHT")
    print(f"  accelerometer vibration : {np.round(twin.vibration['accel'], 2)} m/s² (at rest ~0.015)")
    print(f"  gyro vibration          : {np.round(twin.vibration['gyro'], 3)} rad/s")
    if twin.baro["total_drift_m"] is not None:
        print(f"  barometer drift         : {twin.baro['total_drift_m']:+.1f} m over the flight")
    print(f"\nPX4 configuration copied : {len(twin.px4_params)} parameters (gains, filters, geometry)")

    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    twin.save(out / "twin.json")
    print(f"Twin saved: {out / 'twin.json'}")

    # Plots: what the identification sees.
    data = read_log(args.log)
    n = int(1.0 / np.median(np.diff(data.t)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    gain = smooth(-data.accel[:, 2], n) / np.maximum(smooth(data.thrust_sp, n), 0.05)
    v = smooth(data.voltage, n)
    s = slice(n, -n)
    axes[0].plot(v[s], gain[s], ".", ms=2, alpha=0.4, label="flight (1 s averages)")
    vv = np.linspace(v[s].min(), v[s].max(), 50)
    axes[0].plot(vv, th["gain_full"] * (vv / th["voltage_full"]) ** th["voltage_exponent"], "k",
                 label=f"twin: ∝ V^{th['voltage_exponent']:.2f}")
    axes[0].set_xlabel("Battery voltage (V)")
    axes[0].set_ylabel("Acceleration per unit throttle (m/s²)")
    axes[0].set_title("Thrust fades with the battery")
    axes[0].legend(fontsize=8)
    names = ["front-right", "rear-left", "front-left", "rear-right"]
    axes[1].bar(names, bal["motor_mean"], color=["C3", "C0", "C3", "C0"])
    axes[1].set_ylim(0.5, 0.9)
    axes[1].set_ylabel("Mean motor command in flight")
    axes[1].set_title("Motor balance (front motors work harder)")
    axes[1].tick_params(axis="x", labelsize=8)
    hf = data.imu[:, :3] - smooth(data.imu[:, :3], 16)
    axes[2].plot(data.imu_t, hf[:, 2], lw=0.3, label="accelerometer z, high-pass")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("m/s²")
    axes[2].set_title(f"Vibration in flight (std {np.mean(twin.vibration['accel']):.2f} m/s²)")
    fig.suptitle(f"Digital twin identified from {twin.source}")
    fig.tight_layout()
    fig.savefig(out / "twin.png", dpi=120)
    print(f"Plot saved: {out / 'twin.png'}")
    plt.show()


if __name__ == "__main__":
    main()
