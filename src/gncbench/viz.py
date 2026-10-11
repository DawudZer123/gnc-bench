"""3D view of a flight: the path in space, and an animation of the drone itself."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .dynamics import POS, QUAT
from .math3d import quat_to_rotmat


def plot_flight_3d(ax, res, trajectory, outage=None, title=""):
    """Draws the desired path, the actual path, and the part flown without GPS in red."""
    ref = np.array([trajectory(t)["position"] for t in res.t])
    pos = res.states[:, POS]
    ax.plot(*ref.T, "k--", lw=0.8, label="desired")
    ax.plot(*pos.T, color="C0", lw=1.2, label="actual")
    if outage is not None:
        m = (res.t >= outage[0]) & (res.t < outage[1])
        ax.plot(*pos[m].T, color="C3", lw=2.0, label="actual, GPS lost")
    if res.estimates is not None and not np.isnan(res.estimates).all():
        ax.plot(*res.estimates[:, :3].T, color="C2", lw=0.6, alpha=0.7, label="estimated")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title, fontsize=10)
    _equal_axes(ax, ref, pos)
    ax.legend(fontsize=7, loc="upper left")


def _equal_axes(ax, *paths):
    """Same scale on x and y, otherwise the figure-8 looks distorted. z from the ground up."""
    pts = np.vstack(paths)
    center = (pts.max(axis=0) + pts.min(axis=0)) / 2
    half = (pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)).max() / 2 + 0.3
    top = max(3.0, pts[:, 2].max() + 0.5)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(0.0, top)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=30, azim=-60)


def drone_arms(position, quat, arm=0.35):
    """The two arms of the X-frame in the world frame (drawn larger than life to be visible)."""
    R = quat_to_rotmat(quat)
    d = arm / np.sqrt(2)
    tips = np.array([[d, -d, 0], [-d, d, 0], [d, d, 0], [-d, -d, 0]])   # rotors 0..3
    world = position + tips @ R.T
    return (world[0], world[1]), (world[2], world[3])


def animate_flight(res, trajectory, outage=None, speed=4.0, fps=25, title=""):
    """Animates the drone along its path. speed = how many times faster than real time.

    The path behind the drone turns red while the GPS is lost.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ref = np.array([trajectory(t)["position"] for t in res.t])
    pos = res.states[:, POS]
    ax.plot(*ref.T, "k--", lw=0.8, label="desired")
    trail, = ax.plot([], [], [], color="C0", lw=1.5, label="actual")
    trail_lost, = ax.plot([], [], [], color="C3", lw=2.0, label="actual, GPS lost")
    arm_a, = ax.plot([], [], [], color="k", lw=3)
    arm_b, = ax.plot([], [], [], color="k", lw=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    _equal_axes(ax, ref, pos)
    label = fig.text(0.5, 0.03, "", ha="center", fontsize=11)

    lost = np.zeros(len(res.t), dtype=bool)
    if outage is not None:
        lost = (res.t >= outage[0]) & (res.t < outage[1])
    step = max(1, int(round(speed / (fps * (res.t[1] - res.t[0])))))

    def draw(k):
        for line, (a, b) in zip((arm_a, arm_b), drone_arms(pos[k], res.states[k, QUAT])):
            line.set_data_3d(*np.array([a, b]).T)
        trail.set_data_3d(*pos[:k + 1].T)
        idx = np.where(lost[:k + 1])[0]
        trail_lost.set_data_3d(*pos[idx].T)
        label.set_text(f"t = {res.t[k]:4.1f} s" + ("    GPS LOST" if lost[k] else ""))
        label.set_color("C3" if lost[k] else "k")
        return arm_a, arm_b, trail, trail_lost, label

    anim = FuncAnimation(fig, draw, frames=range(0, len(res.t), step), interval=1000 / fps)
    return fig, anim
