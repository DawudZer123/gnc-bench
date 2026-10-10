"""6-DOF quadcopter dynamics (rigid body).

State vector (17): x = [p (3), v (3), q (4), ω (3), T (4)]
  p: position in the world frame (ENU, z up), m
  v: velocity in the world frame, m/s
  q: body -> world attitude, quaternion [w, x, y, z]
  ω: angular velocity in the body frame, rad/s
  T: actual thrust of each rotor (first-order motor lag), N
Input: commanded thrust of each rotor (4), N.
"""

import numpy as np

from .math3d import quat_mult, quat_to_rotmat

POS, VEL, QUAT, OMEGA, MOTORS = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 17)
STATE_SIZE = 17


def initial_state(position=(0, 0, 0), quat=(1, 0, 0, 0)):
    x = np.zeros(STATE_SIZE)
    x[POS] = position
    x[QUAT] = quat
    return x


def state_derivative(x, rotor_thrusts, params, wind_force=np.zeros(3)):
    """Equations of motion: returns dx/dt."""
    v = x[VEL]
    q = x[QUAT]
    w = x[OMEGA]

    thrusts = x[MOTORS]                                          # actual thrust (state)
    cmd = np.clip(rotor_thrusts, 0.0, params.max_thrust)         # commanded thrust
    total, tx, ty, tz = params.mixer() @ thrusts
    torque = np.array([tx, ty, tz])

    R = quat_to_rotmat(q)
    thrust_world = R @ np.array([0.0, 0.0, total])
    gravity = np.array([0.0, 0.0, -params.mass * params.gravity])
    drag = -params.drag_coeff * v

    J = params.inertia
    dx = np.empty(STATE_SIZE)
    dx[POS] = v
    dx[VEL] = (thrust_world + gravity + drag + wind_force) / params.mass
    dx[QUAT] = 0.5 * quat_mult(q, np.concatenate(([0.0], w)))
    dx[OMEGA] = np.linalg.solve(J, torque - np.cross(w, J @ w))
    dx[MOTORS] = (cmd - thrusts) / params.motor_tau
    return dx
