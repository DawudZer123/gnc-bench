import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import OMEGA, POS, QUAT, VEL, initial_state, state_derivative
from gncbench.integrators import rk4_step
from gncbench.math3d import quat_from_euler, quat_to_euler, quat_to_rotmat
from gncbench.params import QuadParams
from gncbench.sim import simulate

P = QuadParams()


def test_euler_roundtrip():
    angles = np.array([0.3, -0.2, 1.1])
    assert np.allclose(quat_to_euler(quat_from_euler(*angles)), angles)


def test_rotation_matrix_is_orthonormal():
    R = quat_to_rotmat(quat_from_euler(0.4, 0.1, -0.7))
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_mixer_roundtrip():
    rotors = np.array([1.0, 2.0, 3.0, 4.0])
    M = P.mixer()
    assert np.allclose(np.linalg.solve(M, M @ rotors), rotors)


def test_hover_thrust_is_equilibrium():
    x = initial_state(position=(0, 0, 1))
    dx = state_derivative(x, np.full(4, P.mass * P.gravity / 4), P)
    assert np.allclose(dx, 0.0, atol=1e-12)


def test_free_fall_matches_analytic():
    params = QuadParams(drag_coeff=0.0)
    x = initial_state(position=(0, 0, 10))
    dt = 0.01
    for _ in range(100):
        x = rk4_step(lambda s: state_derivative(s, np.zeros(4), params), x, dt)
    assert np.isclose(x[POS][2], 10 - 0.5 * params.gravity * 1.0**2)
    assert np.isclose(x[VEL][2], -params.gravity * 1.0)


def test_torque_free_rotation_conserves_energy_and_momentum():
    x = initial_state()
    x[OMEGA] = [1.0, 0.5, 2.0]
    J = P.inertia

    def invariants(s):
        w = s[OMEGA]
        R = quat_to_rotmat(s[QUAT])
        return 0.5 * w @ J @ w, R @ J @ w   # énergie, moment cinétique (monde)

    e0, h0 = invariants(x)
    # Poussée nulle mais gravité active : elle n'agit pas sur la rotation.
    for _ in range(2000):
        x = rk4_step(lambda s: state_derivative(s, np.zeros(4), P), x, 0.001)
    e1, h1 = invariants(x)
    assert np.isclose(e0, e1, rtol=1e-6)
    assert np.allclose(h0, h1, atol=1e-6)


def test_hover_controller_reaches_target():
    x0 = initial_state(position=(0, 0, 0), quat=quat_from_euler(0.3, -0.2, 0.0))
    ref = {"position": np.array([1.0, -0.5, 2.0]), "yaw": 0.5}
    res = simulate(x0, HoverController(P, dt=0.004), ref, P, duration=10.0)
    final = res.states[-1]
    assert np.linalg.norm(final[POS] - ref["position"]) < 0.01
    assert np.isclose(quat_to_euler(final[QUAT])[2], 0.5, atol=0.01)
    assert np.linalg.norm(final[VEL]) < 0.01
