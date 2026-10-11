import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gncbench.campaign import summarize  # noqa: E402
from gncbench.dynamics import initial_state  # noqa: E402
from gncbench.math3d import quat_from_euler  # noqa: E402
from gncbench.metrics import tracking_report  # noqa: E402
from gncbench.params import QuadParams  # noqa: E402
from gncbench.report import write_report  # noqa: E402
from gncbench.scenario import Scenario  # noqa: E402
from gncbench.sim import SimResult  # noqa: E402
from gncbench.viz import drone_arms, plot_flight_3d  # noqa: E402

P = QuadParams()
HOVER = P.mass * P.gravity / 4


def fake_result(rotors):
    """A drone that stays exactly on target, with the given rotor commands."""
    t = np.arange(len(rotors)) * 0.01
    states = np.tile(initial_state(position=(0, 0, 2)), (len(t), 1))
    return SimResult(t, states, np.asarray(rotors, dtype=float))


def target(t):
    return {"position": np.array([0.0, 0.0, 2.0])}


def test_hover_has_zero_control_effort():
    rep = tracking_report(fake_result(np.full((100, 4), HOVER)), target, params=P)
    assert rep["control_effort_N"] == 0.0
    assert rep["saturation_time_s"] == 0.0
    assert rep["verdict"] == "PASS"


def test_saturation_time_is_counted():
    rotors = np.full((100, 4), HOVER)
    rotors[:30, 2] = P.max_thrust          # one motor flat out for 30 steps of 0.01 s
    rep = tracking_report(fake_result(rotors), target, params=P)
    assert np.isclose(rep["saturation_time_s"], 0.3)
    assert rep["peak_thrust_N"] == P.max_thrust


def test_drone_arms_follow_attitude():
    level = np.array(drone_arms(np.zeros(3), quat_from_euler(0, 0, 0)))
    yawed = np.array(drone_arms(np.zeros(3), quat_from_euler(0, 0, np.pi / 2)))
    assert np.allclose(level[..., 2], 0.0)                       # level drone: arms flat
    assert np.allclose(np.linalg.norm(yawed, axis=-1), 0.35)     # arm length kept
    assert not np.allclose(level, yawed)


def test_report_and_3d_view_are_written(tmp_path):
    rows = [{"seed": i, "max_error_m": e, "verdict": "PASS" if e < 0.5 else "FAIL",
             "time_out_of_corridor_s": 0.0, "wind_speed": 2.0, "gust_std": 1.0,
             "outage_length": 8.0, "mass_error": 0.05, "control_effort_N": 0.1,
             "saturation_time_s": 0.0} for i, e in enumerate([0.2, 0.3, 0.8])]
    assert summarize(rows)["worst_seed"] == 2

    res, _ = Scenario.random(0, duration=6.0).fly()
    fig = plt.figure()
    plot_flight_3d(fig.add_subplot(projection="3d"), res, target)
    path = write_report(tmp_path / "report.html", {"Test": rows}, 0.5, figures=[fig])
    text = path.read_text()
    assert "67 %" in text and "seed 2" in text and "data:image/png;base64" in text
