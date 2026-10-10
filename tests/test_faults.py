import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, initial_state
from gncbench.faults import SensorFault
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU
from gncbench.sim import simulate

P = QuadParams()
TARGET = {"position": np.array([0.0, 0.0, 2.0])}


def run(faults):
    x0 = initial_state(position=TARGET["position"])
    x0[MOTORS] = P.mass * P.gravity / 4
    rng = np.random.default_rng(0)
    return simulate(x0, HoverController(P, dt=0.004), TARGET, P, duration=4.0,
                    sensors=[IMU(rng), GPS(rng)], faults=faults)


def test_gps_dropout_gives_nan_only_during_outage():
    res = run([SensorFault("gps", start=1.0, end=2.0)])
    t, gps = res.measurements["gps"]
    during = (t >= 1.0) & (t < 2.0)
    assert np.isnan(gps[during]).all()
    assert not np.isnan(gps[~during]).any()


def test_freeze_repeats_last_value():
    res = run([SensorFault("gps", start=1.0, end=2.0, mode="freeze")])
    t, gps = res.measurements["gps"]
    during = (t >= 1.0) & (t < 2.0)
    last_before = gps[t < 1.0][-1]
    assert np.allclose(gps[during], last_before)


def test_fault_on_gps_leaves_imu_untouched():
    res = run([SensorFault("gps", start=1.0, end=2.0)])
    _, imu = res.measurements["imu"]
    assert not np.isnan(imu).any()
