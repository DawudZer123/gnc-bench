import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, initial_state
from gncbench.estimator import NavigationFilter
from gncbench.faults import SensorFault
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer
from gncbench.sim import simulate

P = QuadParams()
TARGET = {"position": np.array([0.0, 0.0, 2.0])}


def run(duration, faults=(), use_estimate=False, seed=0):
    x0 = initial_state(position=TARGET["position"])
    x0[MOTORS] = P.mass * P.gravity / 4
    rng = np.random.default_rng(seed)
    imu = IMU(rng)
    kf = NavigationFilter(P, p0=x0[POS])
    res = simulate(x0, HoverController(P, dt=0.004), TARGET, P, duration=duration,
                   sensors=[imu, GPS(rng), Barometer(rng)], faults=faults,
                   estimator=kf, use_estimate=use_estimate)
    return res, kf, imu


def test_filter_beats_raw_gps():
    res, _, _ = run(10.0)
    last = res.t > 5.0
    kf_error = np.linalg.norm(res.estimates[last, :3] - res.states[last, :3], axis=1).mean()
    assert kf_error < 0.3          # le GPS seul se trompe d'environ 1,2 m en moyenne


def test_filter_estimates_vertical_accel_bias():
    _, kf, imu = run(30.0)
    assert np.isclose(kf.s[8], imu.accel_bias[2], atol=0.01)


def test_uncertainty_grows_during_gps_outage():
    res, _, _ = run(10.0, faults=[SensorFault("gps", start=5.0, end=9.0)])
    before = res.estimate_std[np.searchsorted(res.t, 4.99), 0]
    after = res.estimate_std[np.searchsorted(res.t, 8.99), 0]
    assert after > 2 * before


def test_drone_flies_on_estimate():
    res, _, _ = run(10.0, use_estimate=True)
    error = np.linalg.norm(res.states[:, POS] - TARGET["position"], axis=1)
    assert error.max() < 0.5
