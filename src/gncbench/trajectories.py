"""Reference trajectories. Each one is a function t -> dict for the controller."""

import numpy as np


def figure_eight(t, period=20.0, size=3.0, altitude=2.0):
    """A (2 × size) m × size m figure-8, with velocity and acceleration for feedforward."""
    w = 2 * np.pi / period
    return {
        "position": np.array([size * np.sin(w * t), size / 2 * np.sin(2 * w * t), altitude]),
        "velocity": np.array([size * w * np.cos(w * t), size * w * np.cos(2 * w * t), 0.0]),
        "acceleration": np.array([-size * w**2 * np.sin(w * t),
                                  -2 * size * w**2 * np.sin(2 * w * t), 0.0]),
    }
