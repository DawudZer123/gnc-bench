import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, initial_state
from gncbench.flightlog import (FRD_TO_FLU, NED_TO_ENU, attitude_ned_to_enu, fit_drag,
                                from_simulation, mag_heading)
from gncbench.math3d import quat_from_euler, quat_to_euler, quat_to_rotmat, rotmat_to_quat
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer
from gncbench.sim import simulate
from gncbench.trajectories import figure_eight

P = QuadParams()


def test_rotmat_to_quat_roundtrip():
    for angles in [(0.1, -0.2, 0.3), (2.0, 0.5, -3.0), (0.0, 1.4, 3.1)]:
        q = quat_from_euler(*angles)
        assert np.allclose(quat_to_rotmat(rotmat_to_quat(quat_to_rotmat(q))), quat_to_rotmat(q))


def test_ned_frd_conversion():
    # PX4: level drone with its nose to the north (NED identity).
    # Bench (ENU, yaw from east): the same drone has yaw = +90°, no roll, no pitch.
    q = attitude_ned_to_enu(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(quat_to_euler(q), [0.0, 0.0, np.pi / 2])
    # At rest a PX4 accelerometer reads -g on its z (down) axis; for us it is +g up.
    assert np.allclose(FRD_TO_FLU @ [0, 0, -9.81], [0, 0, 9.81])
    assert np.allclose(NED_TO_ENU @ [1, 2, 3], [2, 1, -3])


def test_mag_heading_finds_the_yaw():
    field = np.array([0.0, 0.2, -0.4])                 # ENU: north and down
    for yaw in (0.0, 1.0, -2.5):
        R = quat_to_rotmat(quat_from_euler(0.1, -0.05, yaw))
        assert np.isclose(mag_heading(R.T @ field, 0.1, -0.05), yaw, atol=1e-9)


def test_drag_fit_finds_our_own_drag_coefficient():
    """Validating the validation: on our simulator, the answer must be drag_coeff / mass."""
    rng = np.random.default_rng(0)
    x0 = initial_state(position=figure_eight(0)["position"])
    x0[MOTORS] = P.mass * P.gravity / 4
    res = simulate(x0, HoverController(P, dt=0.004, ki=1.0), figure_eight, P, duration=20.0,
                   sensors=[IMU(rng), GPS(rng), Barometer(rng)])
    fit = fit_drag(from_simulation(res))
    expected = P.drag_coeff / P.mass
    assert np.isclose(fit["k_x"], expected, rtol=0.05)
    assert np.isclose(fit["k_y"], expected, rtol=0.05)
    assert fit["r2_x"] > 0.9 and fit["r2_y"] > 0.9
