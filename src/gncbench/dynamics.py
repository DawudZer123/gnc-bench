"""Dynamique 6 DDL d'un quadricoptère (corps rigide).

Vecteur d'état (13) : x = [p (3), v (3), q (4), ω (3)]
  p : position dans le repère monde (ENU, z vers le haut), m
  v : vitesse dans le repère monde, m/s
  q : attitude corps -> monde, quaternion [w, x, y, z]
  ω : vitesse angulaire dans le repère corps, rad/s
Entrée : poussée de chaque rotor (4), N.
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
    """Équations du mouvement : renvoie dx/dt."""
    v = x[VEL]
    q = x[QUAT]
    w = x[OMEGA]

    thrusts = x[MOTORS]                                          # poussée réelle (état)
    cmd = np.clip(rotor_thrusts, 0.0, params.max_thrust)         # poussée demandée
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
