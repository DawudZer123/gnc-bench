"""One complete PX4 flight on our physics: start PX4, take off, fly a trajectory, record."""

import numpy as np

from .px4bridge import IrisParams, OffboardLink, Px4Bridge, Px4Flight, Px4Process
from .trajectories import figure_eight
from .wind import Wind

# Parameters we change on PX4 before flying: no RC transmitter in simulation, and PX4's
# hover-thrust guess matched to our airframe (it also learns it in flight).
PX4_PARAMS = {
    "COM_RC_IN_MODE": 4,       # no RC needed
    "COM_RCL_EXCEPT": 4,       # no RC-loss failsafe in offboard
    "NAV_RCL_ACT": 0,
    "NAV_DLL_ACT": 0,          # no ground-station link needed
    "COM_DISARM_PRFLT": -1.0,  # don't auto-disarm while we wait on the ground
}

# Fuse the optical flow camera in EKF2 (sensor mounted unrotated, flow valid from 0.1 m to 10 m).
FLOW_PARAMS = {
    "EKF2_OF_CTRL": 1,
    "SENS_FLOW_ROT": 0,
    "SENS_FLOW_MINHGT": 0.1,
    "SENS_FLOW_MAXHGT": 10.0,
}

TAKEOFF_TIME = 5.0     # s to climb to the trajectory's start altitude
SETTLE_TIME = 5.0      # s of hover before the trajectory starts
LANDING_TIME = 6.0     # s of descent at the end


def mission_reference(t, trajectory, duration):
    """Mission timeline (t from arming): climb, hover, trajectory, descend.

    Returns position, velocity and acceleration (ENU).
    """
    start = trajectory(0.0)
    p0, v0 = start["position"], start["velocity"]
    if t < TAKEOFF_TIME:
        s = t / TAKEOFF_TIME
        return p0 * s, np.zeros(3), np.zeros(3)
    t -= TAKEOFF_TIME
    if t < SETTLE_TIME:
        return p0.copy(), np.zeros(3), np.zeros(3)
    t -= SETTLE_TIME
    if t < duration:
        ref = trajectory(t)
        return ref["position"], ref["velocity"], ref["acceleration"]
    end = trajectory(duration)["position"]
    s = min((t - duration) / LANDING_TIME, 1.0)
    return np.array([end[0], end[1], end[2] * (1 - s) - 0.5 * s]), np.zeros(3), np.zeros(3)


def fly_px4(duration=40.0, trajectory=figure_eight, seed=1, wind_speed=0.0, wind_heading=0.0,
            gust_std=0.0, faults=(), instance=0, params=None, px4_dir=None, log_dir=None,
            arm_after=3.0, flow=False, verbose=False):
    """Fly the real PX4 on our physics. `faults` are timed from the start of the trajectory.

    Returns a Px4Flight (truth, PX4 commands, PX4's own estimate, PX4 messages).
    """
    rng = np.random.default_rng(seed)
    heading = np.radians(wind_heading)
    wind = Wind(rng, mean=wind_speed * np.array([np.cos(heading), np.sin(heading), 0.0]),
                gust_std=gust_std) if (wind_speed or gust_std) else None
    bridge = Px4Bridge(params or IrisParams(), seed=seed, instance=instance, wind=wind, faults=faults,
                       flow=flow)
    bridge.listen()
    px4 = Px4Process(instance, px4_dir=px4_dir, log_dir=log_dir)
    link = None
    try:
        bridge.accept(timeout=60.0)
        link = OffboardLink(instance)
        extra = FLOW_PARAMS if flow else {}
        return _run_mission(bridge, link, duration, trajectory, arm_after, verbose, extra)
    finally:
        px4.stop()
        bridge.close()
        if link is not None:
            link.close()


def _run_mission(bridge, link, duration, trajectory, arm_after, verbose, extra_params):
    total = TAKEOFF_TIME + SETTLE_TIME + duration + LANDING_TIME
    log_t, log_x, log_u, log_w, log_ref = [], [], [], [], []
    params_sent = False
    armed_at = np.nan
    next_setpoint = next_try = 0.0
    while True:
        bridge.step()
        t = bridge.t
        link.poll(t)
        if link.connected and not params_sent:
            for name, value in {**PX4_PARAMS, **extra_params}.items():
                link.set_param(name, value)
            params_sent = True
        if np.isnan(armed_at):
            if link.armed:
                armed_at = t
                bridge.mission_start = t + TAKEOFF_TIME + SETTLE_TIME
                if verbose:
                    print(f"  armed at t = {t:.1f} s")
            elif t >= arm_after and t >= next_try and link.connected:
                link.set_offboard()
                link.arm()
                next_try = t + 1.0
            if t > arm_after + 60.0:
                raise RuntimeError("PX4 refused to arm: " + "; ".join(m for _, m in link.messages[-5:]))
        tm = t - armed_at if not np.isnan(armed_at) else 0.0
        pos, vel, acc = mission_reference(tm, trajectory, duration)
        if t >= next_setpoint and link.connected:       # setpoints at 20 Hz (offboard needs > 2 Hz)
            link.setpoint(t, pos, vel, acc)
            next_setpoint = t + 0.05
        log_t.append(t)
        log_x.append(bridge.x.copy())
        log_u.append(bridge.u.copy())
        log_w.append(getattr(bridge, "current_wind", np.zeros(3)))
        log_ref.append(pos if not np.isnan(armed_at) else np.full(3, np.nan))
        if not np.isnan(armed_at) and tm > total:
            break
    est = link.positions
    return Px4Flight(
        t=np.array(log_t), states=np.array(log_x), commands=np.array(log_u), wind=np.array(log_w),
        reference=np.array(log_ref),
        px4_t=np.array([e[0] for e in est]), px4_pos=np.array([e[1] for e in est]).reshape(-1, 3),
        px4_vel=np.array([e[2] for e in est]).reshape(-1, 3),
        messages=link.messages, modes=link.modes, armed_at=armed_at,
        missed_steps=bridge.missed_steps)
