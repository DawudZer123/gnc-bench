"""Outils de rotation. Convention : quaternion q = [w, x, y, z], corps -> monde."""

import numpy as np


def quat_mult(p, q):
    """Produit de Hamilton p ⊗ q."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    """Matrice R telle que v_monde = R @ v_corps."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_euler(roll, pitch, yaw):
    """Angles d'Euler ZYX (lacet, tangage, roulis) en radians -> quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_to_euler(q):
    """Quaternion -> (roulis, tangage, lacet) en radians, convention ZYX."""
    w, x, y, z = quat_normalize(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def hat(v):
    """Matrice antisymétrique : hat(a) @ b == cross(a, b)."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def vee(M):
    """Inverse de hat."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])
