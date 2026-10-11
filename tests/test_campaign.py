import numpy as np

from gncbench.campaign import run_campaign, summarize
from gncbench.controllers import Controller
from gncbench.math3d import cross3
from gncbench.scenario import Scenario


def test_cross3_matches_numpy():
    a, b = np.array([1.0, -2.0, 0.5]), np.array([0.3, 4.0, -1.0])
    assert np.allclose(cross3(a, b), np.cross(a, b))


def test_random_scenario_is_reproducible_and_in_range():
    s1, s2 = Scenario.random(7), Scenario.random(7)
    assert s1 == s2
    assert s1 != Scenario.random(8)
    assert 0.0 <= s1.wind_speed <= 5.0
    assert -0.1 <= s1.mass_error <= 0.1
    assert s1.outage[1] < s1.duration


def test_same_scenario_gives_same_flight():
    s = Scenario.random(3, duration=10.0)
    _, rep1 = s.fly()
    _, rep2 = s.fly()
    assert rep1["max_error_m"] == rep2["max_error_m"]


class Freefall(Controller):
    """A controller that cuts the motors: the bench must catch it."""

    def __init__(self, params, dt):
        pass

    def update(self, t, state, reference):
        return np.zeros(4)


def test_bench_catches_a_bad_controller():
    _, rep = Scenario.random(0, duration=10.0).fly(controller=Freefall)
    assert rep["verdict"] == "FAIL"


def test_campaign_rows_and_summary():
    rows = run_campaign(3, duration=8.0, workers=2, verbose=False)
    assert [r["seed"] for r in rows] == [0, 1, 2]
    assert {"wind_speed", "max_error_m", "verdict"} <= rows[0].keys()
    summary = summarize(rows)
    assert summary["flights"] == 3
    assert 0.0 <= summary["pass_rate"] <= 1.0
    assert summary["worst_max_error_m"] >= summary["median_max_error_m"]
