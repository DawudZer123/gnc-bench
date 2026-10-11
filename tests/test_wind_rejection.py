import numpy as np

from gncbench.controllers import HoverController, WindRejectionController
from gncbench.dynamics import MOTORS, POS, initial_state
from gncbench.estimator import NavigationFilter
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer
from gncbench.sim import simulate
from gncbench.wind import Wind

P = QuadParams()
TARGET = {"position": np.array([0.0, 0.0, 2.0])}


def hover_in_steady_wind(controller_cls):
    x0 = initial_state(position=TARGET["position"])
    x0[MOTORS] = P.mass * P.gravity / 4
    rng = np.random.default_rng(0)
    kf = NavigationFilter(P, p0=x0[POS])
    res = simulate(x0, controller_cls(P, dt=0.004, ki=0.0), TARGET, P, duration=10.0,
                   sensors=[IMU(rng), GPS(rng), Barometer(rng)], estimator=kf,
                   wind=Wind(np.random.default_rng(1), mean=(4.0, 0.0, 0.0), gust_std=0.0))
    return res, kf


def test_filter_measures_the_wind_force():
    _, kf = hover_in_steady_wind(HoverController)
    # Drone at rest in a 4 m/s wind: external acceleration = drag × wind / mass.
    assert np.allclose(kf.disturbance, [P.drag_coeff * 4.0 / P.mass, 0.0, 0.0], atol=0.05)


def test_wind_rejection_removes_the_steady_offset():
    # Without integral, the baseline settles about 0.2 m downwind (kp × e = c × w / m).
    base, _ = hover_in_steady_wind(HoverController)
    rej, _ = hover_in_steady_wind(WindRejectionController)
    offset = lambda res: np.linalg.norm(res.states[-1, POS] - TARGET["position"])  # noqa: E731
    assert offset(base) > 0.15
    assert offset(rej) < 0.05


def test_rejection_is_off_without_a_filter():
    x0 = initial_state(position=TARGET["position"])
    x0[MOTORS] = P.mass * P.gravity / 4
    runs = [simulate(x0, cls(P, dt=0.004), TARGET, P, duration=2.0).states
            for cls in (HoverController, WindRejectionController)]
    assert np.array_equal(*runs)
