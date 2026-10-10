import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, QUAT, VEL, initial_state
from gncbench.estimator import NavigationFilter
from gncbench.faults import SensorFault
from gncbench.math3d import quat_from_euler
from gncbench.metrics import tracking_report
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer, OpticalFlow
from gncbench.sim import simulate

P = QuadParams()


def test_flow_measures_body_velocity():
    flow = OpticalFlow(np.random.default_rng(0), vel_noise=0.0)
    x = initial_state(quat=quat_from_euler(0.0, 0.0, np.pi / 2))   # drone rotated by 90°
    x[VEL] = [1.0, 0.0, 0.0]                                       # moving along world x
    # Seen from the drone, world x is on its right: body velocity = (0, -1).
    assert np.allclose(flow.measure(0.0, x, None, P), [0.0, -1.0])


def circle(t):
    w = 2 * np.pi / 10
    return {"position": np.array([np.cos(w * t), np.sin(w * t), 2.0]),
            "velocity": np.array([-w * np.sin(w * t), w * np.cos(w * t), 0.0]),
            "acceleration": np.array([-w**2 * np.cos(w * t), -w**2 * np.sin(w * t), 0.0])}


def fly(with_flow):
    rng = np.random.default_rng(1)
    x0 = initial_state(position=circle(0)["position"])
    x0[MOTORS] = P.mass * P.gravity / 4
    sensors = [IMU(rng), GPS(rng), Barometer(rng)] + ([OpticalFlow(rng)] if with_flow else [])
    return simulate(x0, HoverController(P, dt=0.004, ki=1.0), circle, P, duration=15.0,
                    sensors=sensors, faults=[SensorFault("gps", start=5.0, end=15.0)],
                    estimator=NavigationFilter(P, p0=x0[POS]), use_estimate=True)


def test_optical_flow_keeps_drone_in_corridor_without_gps():
    rep = tracking_report(fly(with_flow=True), circle, corridor=0.5, window=(5.0, 15.0))
    assert rep["verdict"] == "PASS"


def test_report_counts_time_out_of_corridor():
    res = fly(with_flow=True)
    rep = tracking_report(res, circle, corridor=0.0, window=(5.0, 15.0))
    assert rep["verdict"] == "FAIL"
    assert np.isclose(rep["time_out_of_corridor_s"], 10.0, atol=0.01)
