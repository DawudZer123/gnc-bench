"""Digital twin: the identification recovers known values from synthetic flights."""

import numpy as np
import pytest

from gncbench.twin import (Battery, DigitalTwin, TwinData, first_order_lag, fit_balance,
                           fit_rotation, fit_thrust, px4_mixer)

QUAD_X = {}
for i, (px, py, km) in enumerate([(1, 1, 0.05), (-1, -1, 0.05), (1, -1, -0.05), (-1, 1, -0.05)]):
    QUAD_X.update({f"CA_ROTOR{i}_PX": px, f"CA_ROTOR{i}_PY": py, f"CA_ROTOR{i}_KM": km,
                   f"CA_ROTOR{i}_CT": 6.5})


def synthetic(n=6000, dt=0.02, seed=0, **kw):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    base = dict(t=t, thrust_sp=np.full(n, 0.7), torque_sp=np.zeros((n, 3)), rate=np.zeros((n, 3)),
                rate_dot=np.zeros((n, 3)), accel=np.zeros((n, 3)), voltage=np.full(n, 15.0),
                current=np.full(n, 20.0), drawn_mah=np.linspace(0, 1000, n),
                motors_t=t, motors=np.full((n, 4), 0.7), imu_t=t, imu=rng.normal(0, 1, (n, 6)),
                baro_bias_t=t, baro_bias=np.zeros(n), params=dict(QUAD_X), airframe=4001,
                hardware="test")
    base.update(kw)
    return TwinData(**base)


def test_px4_mixer_matches_px4_quad_x():
    m = px4_mixer(QUAD_X)
    # PX4 quad X: ±0.707 roll/pitch, ±1 yaw, 1 thrust (front-right motor: roll -, pitch +, yaw +)
    assert np.allclose(np.abs(m[:, :2]), np.sqrt(0.5), atol=1e-6)
    assert np.allclose(np.abs(m[:, 2]), 1.0) and np.allclose(m[:, 3], 1.0)
    assert np.allclose(m[0], [-np.sqrt(0.5), np.sqrt(0.5), 1.0, 1.0], atol=1e-6)


def test_thrust_gain_and_battery_fade():
    n = 6000
    v = np.linspace(16.0, 14.0, n)
    gain = 14.0 * (v / 16.0) ** 1.2
    thrust = 9.81 / gain                                    # hover: throttle rises as V drops
    d = synthetic(thrust_sp=thrust, voltage=v, accel=np.column_stack(
        [np.zeros(n), np.zeros(n), -gain * thrust]))
    fit = fit_thrust(d)
    assert fit["gain_full"] == pytest.approx(14.0, rel=0.01)
    assert fit["voltage_exponent"] == pytest.approx(1.2, abs=0.05)


def test_centre_of_gravity_from_motor_balance():
    # Front motors (0 and 2) 10 % harder: the CG is forward, on the centre line.
    d = synthetic(motors=np.tile([0.77, 0.7, 0.77, 0.7], (6000, 1)))
    fit = fit_balance(d, px4_mixer(QUAD_X))
    assert fit["cg_fraction"][0] == pytest.approx(0.14 / 2.94, rel=1e-6)
    assert fit["cg_fraction"][1] == pytest.approx(0.0, abs=1e-9)
    assert fit["yaw_torque_ratio"] == pytest.approx(1.0)


def test_twin_is_balanced_in_hover():
    """With the real hover motor commands, the twin's torques cancel: same CG and yaw balance."""
    motors = np.array([0.74, 0.66, 0.80, 0.68])
    d = synthetic(motors=np.tile(motors, (6000, 1)))
    twin = DigitalTwin(source="x", hardware="x", airframe=0,
                       thrust={"gain_full": 14.0, "voltage_full": 15.0, "voltage_exponent": 1.0,
                               "hover_thrust": 0.72},
                       battery={}, balance=fit_balance(d, px4_mixer(QUAD_X)),
                       rotation={"roll": None, "pitch": None, "yaw": None, "motor_tau": None},
                       drag={}, vibration={"accel": [1, 1, 1], "gyro": [0.02] * 3},
                       baro={"walk": None}, px4_params=dict(QUAD_X))
    p = twin.quad_params()
    total, *torque = p.mixer() @ (motors * p.max_thrust)
    assert np.allclose(torque, 0.0, atol=1e-9)


def test_rotation_identified_only_with_excitation():
    dt, n = 0.02, 6000
    rng = np.random.default_rng(1)
    torque = first_order_lag(rng.normal(0, 0.3, (n, 3)), 0.1, dt)        # doublet-like input
    rate_dot = np.column_stack([60.0 * first_order_lag(torque[:, i], 0.04, dt) for i in range(3)])
    rate = np.cumsum(rate_dot, axis=0) * dt
    rate -= first_order_lag(rate, 1.0, dt)                               # keep it bounded
    d = synthetic(torque_sp=torque, rate=rate, rate_dot=rate_dot + rng.normal(0, 1, (n, 3)))
    fit = fit_rotation(d, px4_mixer(QUAD_X))
    for axis in ("roll", "pitch", "yaw"):
        assert fit[axis]["gain"] == pytest.approx(60.0, rel=0.05)
    assert fit["motor_tau"] == pytest.approx(0.04, abs=0.021)
    calm = synthetic()                                                   # hover: nothing moves
    assert fit_rotation(calm, px4_mixer(QUAD_X))["roll"] is None


def test_battery_voltage_and_thrust_fade():
    b = Battery(v_full=16.0, exponent=1.0, volts_per_mah=0.001, resistance=0.0, i0=0.0, i1=30.0,
                v0=16.0)
    for _ in range(60 * 250):                      # one minute at throttle 0.5 (15 A)
        scale = b.step(0.5, 0.004)
    assert b.drawn_mah == pytest.approx(250.0, rel=1e-3)
    assert scale == pytest.approx((16.0 - 0.25) / 16.0, rel=1e-3)
