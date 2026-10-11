import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, QUAT, initial_state
from gncbench.ekf import NavigationEKF, small_rotation
from gncbench.math3d import quat_from_euler, quat_mult, quat_to_euler
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer, Magnetometer
from gncbench.sim import simulate

P = QuadParams()
TARGET = {"position": np.array([0.0, 0.0, 2.0]), "yaw": 0.3}


def test_flow_jacobian_matches_finite_differences():
    """The classic EKF bug is a wrong Jacobian: check it numerically."""
    ekf = NavigationEKF(P, p0=np.zeros(3), v0=(1.0, -0.5, 0.2), q0=quat_from_euler(0.2, -0.1, 0.7))
    H = ekf.flow_jacobian()
    h0 = ekf.flow_prediction()
    eps = 1e-6
    for i in range(3):
        ekf.v[i] += eps
        assert np.allclose((ekf.flow_prediction() - h0) / eps, H[:, 3 + i], atol=1e-5)
        ekf.v[i] -= eps
    q0 = ekf.q.copy()
    for i in range(3):
        dtheta = np.zeros(3)
        dtheta[i] = eps
        ekf.q = quat_mult(q0, small_rotation(dtheta))
        assert np.allclose((ekf.flow_prediction() - h0) / eps, H[:, 6 + i], atol=1e-5)
    ekf.q = q0


def hover_on_ekf(q0_error=(0.0, 0.0, 0.0), duration=15.0):
    x0 = initial_state(position=TARGET["position"])
    x0[MOTORS] = P.mass * P.gravity / 4
    rng = np.random.default_rng(0)
    imu = IMU(rng)
    ekf = NavigationEKF(P, p0=x0[POS], q0=quat_from_euler(*q0_error))
    res = simulate(x0, HoverController(P, dt=0.004), TARGET, P, duration=duration,
                   sensors=[imu, GPS(rng), Barometer(rng), Magnetometer(rng)],
                   estimator=ekf, use_estimate=True)
    return res, ekf, imu


def test_ekf_corrects_a_wrong_initial_attitude():
    # The filter starts 3° off in roll and pitch and 10° off in yaw.
    res, ekf, _ = hover_on_ekf(q0_error=np.radians([3.0, -3.0, 10.0]))
    err = quat_to_euler(ekf.attitude) - quat_to_euler(res.states[-1, QUAT])
    assert np.all(np.abs(np.degrees(err)) < 0.5)


def test_ekf_finds_the_gyro_bias():
    _, ekf, imu = hover_on_ekf()
    assert np.allclose(ekf.bg, imu.gyro_bias, atol=2e-3)


def test_drone_hovers_on_estimated_attitude():
    res, _, _ = hover_on_ekf()
    final = res.states[-1]
    assert np.linalg.norm(final[POS] - TARGET["position"]) < 0.3
    assert abs(quat_to_euler(final[QUAT])[2] - TARGET["yaw"]) < np.radians(2)
