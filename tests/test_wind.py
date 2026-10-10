import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, initial_state
from gncbench.params import QuadParams
from gncbench.sim import simulate
from gncbench.wind import Wind

P = QuadParams()
TARGET = {"position": np.array([0.0, 0.0, 2.0])}


def hover_at_target():
    x = initial_state(position=TARGET["position"])
    x[MOTORS] = P.mass * P.gravity / 4
    return x


def test_gusts_have_expected_statistics():
    wind = Wind(np.random.default_rng(0), mean=(3.0, 0.0, 0.0), gust_std=1.5, gust_time=1.0)
    samples = np.array([wind.step(0.01) for _ in range(200_000)])
    assert np.allclose(samples.mean(axis=0), [3.0, 0.0, 0.0], atol=0.1)
    assert np.allclose(samples.std(axis=0), 1.5, rtol=0.1)


def test_steady_wind_offset_without_integral():
    # Sans intégrale, le vent est compensé par un écart de position : kp × e = c × w / m.
    wind = Wind(np.random.default_rng(0), mean=(5.0, 0.0, 0.0), gust_std=0.0)
    res = simulate(hover_at_target(), HoverController(P, dt=0.004, ki=0.0), TARGET, P,
                   duration=10.0, wind=wind)
    expected_offset = P.drag_coeff * 5.0 / (P.mass * 6.0)
    assert np.isclose(res.states[-1, 0] - TARGET["position"][0], expected_offset, atol=0.01)


def test_integral_cancels_steady_wind():
    wind = Wind(np.random.default_rng(0), mean=(5.0, 0.0, 0.0), gust_std=0.0)
    res = simulate(hover_at_target(), HoverController(P, dt=0.004, ki=2.0), TARGET, P,
                   duration=20.0, wind=wind)
    assert np.linalg.norm(res.states[-1, POS] - TARGET["position"]) < 0.01
