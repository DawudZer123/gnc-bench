import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, initial_state
from gncbench.ekf import NavigationEKF
from gncbench.faults import SensorFault
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer, Magnetometer, OpticalFlow
from gncbench.sim import simulate

P = QuadParams()
TARGET = {"position": np.array([0.0, 0.0, 2.0])}


def test_offset_and_drift_faults():
    z = np.zeros(6)
    jump = SensorFault("gps", 1.0, 5.0, mode="offset", value=(5.0, 0.0, 0.0))
    drift = SensorFault("gps", 1.0, 5.0, mode="drift", value=(0.0, 0.5))
    assert np.allclose(jump.apply(z, None, 2.0), [5, 0, 0, 0, 0, 0])
    assert np.allclose(drift.apply(z, None, 3.0), [0, 1.0, 0, 0, 0, 0])


def hover(fault=None, detection=True, duration=12.0):
    x0 = initial_state(position=TARGET["position"])
    x0[MOTORS] = P.mass * P.gravity / 4
    rng = np.random.default_rng(0)
    ekf = NavigationEKF(P, p0=x0[POS], fault_detection=detection)
    res = simulate(x0, HoverController(P, dt=0.004), TARGET, P, duration=duration,
                   sensors=[IMU(rng), GPS(rng), Barometer(rng), OpticalFlow(rng), Magnetometer(rng)],
                   faults=[fault] if fault else [], estimator=ekf, use_estimate=True)
    error = np.linalg.norm(res.states[:, POS] - TARGET["position"], axis=1)
    return ekf, error


def test_false_alarms_match_the_gate():
    # Gate at 99.9 %: about 1 good measurement in 1000 is rejected by bad luck, not more.
    ekf, _ = hover(duration=20.0)
    total = sum(sum(h.values()) for h in ekf.health.values())
    rejected = sum(h["rejected"] for h in ekf.health.values())
    assert rejected / total < 0.005


def test_gps_jump_is_rejected():
    jump = SensorFault("gps", 4.0, 12.0, mode="offset", value=(5.0, 0.0, 0.0))
    ekf, error = hover(jump)
    assert ekf.health["gps"]["rejected"] > 70        # every faulty GPS fix (10 Hz × 8 s)
    assert error[-1] < 0.3


def test_naive_filter_follows_the_jump():
    jump = SensorFault("gps", 4.0, 12.0, mode="offset", value=(5.0, 0.0, 0.0))
    _, error = hover(jump, detection=False)
    assert error[-1] > 3.0                            # the drone moved 5 m to "correct" itself


def test_frozen_gps_is_flagged_stale():
    ekf, error = hover(SensorFault("gps", 4.0, 12.0, mode="freeze"))
    assert ekf.health["gps"]["stale"] > 70
    assert error[-1] < 0.3
