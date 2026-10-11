"""Real flight data: read a PX4 log (.ulg) and check our models against it.

PX4 uses NED for the world (x north, y east, z DOWN) and FRD for the body
(x forward, y right, z DOWN). The bench uses ENU and FLU (z UP). Everything is
converted once, here, so the rest of the code never has to think about it.

Three checks:
  sensor_noise     how noisy the real sensors are, compared with our sensor models;
  replay_ekf       our EKF run on the real sensor data, compared with PX4's own EKF;
  fit_drag         the drag coefficient seen in flight, compared with our dynamics model.
"""

from dataclasses import dataclass, field

import numpy as np

from .ekf import NavigationEKF, wrap
from .math3d import quat_from_euler, quat_to_euler, quat_to_rotmat, rotmat_to_quat

NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
FRD_TO_FLU = np.diag([1.0, -1.0, -1.0])
EARTH_RADIUS = 6_371_000.0  # m


@dataclass
class FlightLog:
    """Sensor data and PX4's own estimates, in the bench's frames. Times in s."""

    imu_t: np.ndarray                 # (N,)
    imu: np.ndarray                   # (N, 6) accel (m/s², FLU) + gyro (rad/s, FLU)
    gps_t: np.ndarray                 # (M,)
    gps: np.ndarray                   # (M, 6) position (m, ENU) + velocity (m/s, ENU)
    baro_t: np.ndarray                # (K,)
    baro: np.ndarray                  # (K, 1) altitude (m), same origin as gps z
    mag_t: np.ndarray                 # (L,)
    mag: np.ndarray                   # (L, 3) magnetic field (gauss, FLU)
    ref_t: np.ndarray                 # PX4 estimate times
    ref_pos: np.ndarray               # (R, 3) PX4 position estimate (ENU)
    ref_vel: np.ndarray               # (R, 3) PX4 velocity estimate (ENU)
    att_t: np.ndarray                 # PX4 attitude times
    att: np.ndarray                   # (A, 4) PX4 attitude estimate, quaternion ENU <- FLU
    in_air_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    in_air: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    gps_std: tuple = (0.5, 0.5, 1.0, 0.1)   # median reported eph, eph, epv, speed accuracy

    def flying(self, t):
        """True where the drone is in the air (all False if the log does not say)."""
        if len(self.in_air_t) == 0:
            return np.zeros_like(t, dtype=bool)
        idx = np.clip(np.searchsorted(self.in_air_t, t, side="right") - 1, 0, None)
        return self.in_air[idx]


def attitude_ned_to_enu(q_ned_frd):
    return rotmat_to_quat(NED_TO_ENU @ quat_to_rotmat(q_ned_frd) @ FRD_TO_FLU)


def load_px4_log(path):
    """Reads a .ulg file. Needs `pip install pyulog`."""
    from pyulog import ULog   # imported here: only this function needs it

    ulog = ULog(str(path))
    t0 = ulog.start_timestamp

    def topic(*names):
        for name in names:
            for d in ulog.data_list:
                if d.name == name and d.multi_id == 0:
                    return d.data
        return None

    def secs(d):
        # Signed: a few messages can be stamped just before the log starts, and an
        # unsigned subtraction would turn them into 18 trillion seconds.
        return (d["timestamp"].astype(np.int64) - np.int64(t0)) / 1e6

    def vec(d, prefix, n=3):
        return np.column_stack([d[f"{prefix}[{i}]"] for i in range(n)])

    sc = topic("sensor_combined")
    imu = np.hstack([vec(sc, "accelerometer_m_s2") @ FRD_TO_FLU.T,
                     vec(sc, "gyro_rad") @ FRD_TO_FLU.T])

    lp = topic("vehicle_local_position")
    ok = lp["xy_valid"].astype(bool) if np.any(lp["xy_valid"]) else np.ones(len(lp["x"]), bool)
    lat0, lon0, alt0 = (float(lp[k][ok][0]) for k in ("ref_lat", "ref_lon", "ref_alt"))
    ref_pos = np.column_stack([lp["x"], lp["y"], lp["z"]]) @ NED_TO_ENU.T
    ref_vel = np.column_stack([lp["vx"], lp["vy"], lp["vz"]]) @ NED_TO_ENU.T

    g = topic("vehicle_gps_position", "sensor_gps")
    if g is None:                               # no GPS in this log (bench test, indoor...)
        g = {"timestamp": np.zeros(0), "lat": np.zeros(0), "lon": np.zeros(0), "alt": np.zeros(0),
             "fix_type": np.zeros(0), "eph": np.zeros(0), "epv": np.zeros(0),
             "s_variance_m_s": np.zeros(0), "vel_e_m_s": np.zeros(0), "vel_n_m_s": np.zeros(0),
             "vel_d_m_s": np.zeros(0)}
    if "latitude_deg" in g:                     # recent PX4 versions
        lat, lon, alt = g["latitude_deg"], g["longitude_deg"], g["altitude_msl_m"]
    else:                                       # older versions: 1e-7 deg and mm
        lat, lon, alt = g["lat"] * 1e-7, g["lon"] * 1e-7, g["alt"] * 1e-3
    fix = g["fix_type"] >= 3
    east = np.radians(lon - lon0) * EARTH_RADIUS * np.cos(np.radians(lat0))
    north = np.radians(lat - lat0) * EARTH_RADIUS
    gps = np.column_stack([east, north, alt - alt0, g["vel_e_m_s"], g["vel_n_m_s"], -g["vel_d_m_s"]])
    gps_std = ((float(np.median(g["eph"][fix])),) * 2 + (float(np.median(g["epv"][fix])),
               float(np.median(g["s_variance_m_s"][fix]))) if fix.any() else (np.nan,) * 4)

    air = topic("vehicle_air_data")
    if air is not None:
        baro_t, baro_alt = secs(air), air["baro_alt_meter"]
    else:                                       # older logs: baro inside sensor_combined
        baro_t, baro_alt = secs(sc), sc["baro_alt_meter"]
    # The barometer has its own unknown offset: align it on the GPS altitude at the first fix.
    if fix.any():
        baro_offset = np.interp(secs(g)[fix][0], baro_t, baro_alt) - (alt[fix][0] - alt0)
    else:
        baro_offset = baro_alt[0]
    baro = (baro_alt - baro_offset)[:, None]

    m = topic("vehicle_magnetometer")
    m = m if m is not None else topic("sensor_mag")
    if m is not None and "magnetometer_ga[0]" in m:
        mag_t, mag = secs(m), vec(m, "magnetometer_ga") @ FRD_TO_FLU.T
    else:
        mag_t, mag = secs(sc), vec(sc, "magnetometer_ga") @ FRD_TO_FLU.T

    va = topic("vehicle_attitude")
    att = np.array([attitude_ned_to_enu(q) for q in vec(va, "q", 4)])

    ld = topic("vehicle_land_detected")
    in_air_t = secs(ld) if ld is not None else np.zeros(0)
    in_air = (ld["landed"] == 0) if ld is not None else np.zeros(0, dtype=bool)

    return FlightLog(secs(sc), imu, secs(g)[fix], gps[fix], baro_t, baro, mag_t, mag,
                     secs(lp), ref_pos, ref_vel, secs(va), att, in_air_t, in_air, gps_std)


# ----- 1. Sensor noise ------------------------------------------------------------------
def still_window(log, min_length=5.0):
    """Longest stretch where the drone is on the ground and not moving (gyro quiet)."""
    still = (np.linalg.norm(log.imu[:, 3:], axis=1) < 0.03) & ~log.flying(log.imu_t)
    best, start = (0.0, 0.0), None
    for i, s in enumerate(np.append(still, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if log.imu_t[i - 1] - log.imu_t[start] > best[1] - best[0]:
                best = (log.imu_t[start], log.imu_t[i - 1])
            start = None
    if best[1] - best[0] < min_length:
        raise ValueError("No still period of at least %.0f s in this log" % min_length)
    return best


def sensor_noise(log, window=None):
    """Noise of each real sensor while the drone sits still.

    For the GPS, two numbers: the spread of the position (std) and the spread of the
    jump between two consecutive fixes. For white noise (our model) they are equal.
    For a real GPS the error wanders slowly, so the jumps are much smaller than the spread.
    """
    a, b = window or still_window(log)
    sel = lambda t: (t >= a) & (t <= b)  # noqa: E731
    imu = log.imu[sel(log.imu_t)]
    gps = log.gps[sel(log.gps_t)]
    baro = log.baro[sel(log.baro_t), 0]
    gps_ok = len(gps) > 2
    return {
        "window_s": (float(a), float(b)),
        "accel_std": float(np.mean(imu[:, :3].std(axis=0))),
        "gyro_std": float(np.mean(imu[:, 3:].std(axis=0))),
        "gyro_bias": imu[:, 3:].mean(axis=0),
        "baro_std": float(baro.std()),
        "gps_pos_std": gps[:, :3].std(axis=0) if gps_ok else None,
        "gps_step_std": np.diff(gps[:, :3], axis=0).std(axis=0) / np.sqrt(2) if gps_ok else None,
        "gps_vel_std": float(np.mean(gps[:, 3:].std(axis=0))) if gps_ok else None,
    }


# ----- 2. Replay our EKF on the real data -------------------------------------------------
def mag_heading(mag_body, roll, pitch, declination=0.0):
    """Yaw (ENU convention: 0 = facing east) from a body-frame magnetometer reading."""
    level = quat_to_rotmat(quat_from_euler(roll, pitch, 0.0)) @ mag_body
    alpha = np.arctan2(level[1], level[0])          # field direction seen from the nose
    return wrap(np.pi / 2 - declination - alpha)     # magnetic north is at +90° in ENU


def replay_ekf(log, params, start=None, end=None, noise=None):
    """Runs NavigationEKF on the real sensors and returns its estimates.

    Initial attitude: roll and pitch from gravity, yaw from the magnetometer (the usual
    "coarse alignment"). One constant is borrowed from PX4: the magnetic declination,
    chosen so that our first heading matches PX4's (PX4 gets it from a world map).
    """
    start = log.gps_t[0] if start is None else start
    end = log.imu_t[-1] if end is None else end
    noise = noise or {}

    # Coarse alignment on the first 0.5 s.
    first = (log.imu_t >= start) & (log.imu_t < start + 0.5)
    f = log.imu[first, :3].mean(axis=0)
    roll, pitch = np.arctan2(f[1], f[2]), np.arctan2(-f[0], np.hypot(f[1], f[2]))
    m0 = log.mag[np.searchsorted(log.mag_t, start)]
    px4_yaw = quat_to_euler(log.att[np.searchsorted(log.att_t, start)])[2]
    declination = wrap(mag_heading(m0, roll, pitch) - px4_yaw)
    yaw = mag_heading(m0, roll, pitch, declination)

    k = np.searchsorted(log.gps_t, start)
    std = log.gps_std
    # Noise measured at rest is a floor: in flight, motor vibrations make it worse. So we
    # never go below the values our simulated sensors use.
    ekf = NavigationEKF(params, p0=log.gps[k, :3], v0=log.gps[k, 3:],
                        q0=quat_from_euler(roll, pitch, yaw),
                        accel_noise=max(noise.get("accel_std", 0.0), 0.05),
                        gyro_noise=max(noise.get("gyro_std", 0.0), 0.005),
                        gps_noise=(std[0], std[1], std[2], std[3], std[3], std[3]),
                        baro_noise=max(noise.get("baro_std", 0.0), 0.1), mag_noise=0.05)

    events = [(t, "imu", i) for i, t in enumerate(log.imu_t) if start <= t <= end]
    events += [(t, "gps", i) for i, t in enumerate(log.gps_t) if start < t <= end]
    events += [(t, "baro", i) for i, t in enumerate(log.baro_t) if start < t <= end]
    events += [(t, "mag", i) for i, t in enumerate(log.mag_t) if start < t <= end]
    events.sort(key=lambda e: e[0])

    out_t, out_p, out_v, out_q = [], [], [], []
    last_imu = None
    for t, name, i in events:
        if name == "imu":
            dt = t - last_imu if last_imu is not None else 0.0
            last_imu = t
            if dt > 0:
                ekf.process("imu", log.imu[i], None, dt)
                out_t.append(t)
                out_p.append(ekf.position)
                out_v.append(ekf.velocity)
                out_q.append(ekf.attitude)
        elif name == "gps":
            ekf.process("gps", log.gps[i], None, 0.0)
        elif name == "baro":
            ekf.process("baro", log.baro[i], None, 0.0)
        else:
            r, p, _ = quat_to_euler(ekf.attitude)
            ekf.process("mag", np.array([mag_heading(log.mag[i], r, p, declination)]), None, 0.0)
    return np.array(out_t), np.array(out_p), np.array(out_v), np.array(out_q), ekf


def compare_with_px4(log, t, pos, vel, quat, flight_only=True):
    """Differences between our estimates and PX4's, sampled at PX4's times.

    flight_only: judge only the time in the air (if the log says when). On the ground,
    PX4 uses its land detector to force the velocity to zero; our EKF does not.
    """
    sel = (log.ref_t >= t[0]) & (log.ref_t <= t[-1])
    if flight_only and log.flying(log.ref_t).any():
        sel &= log.flying(log.ref_t)
    rt = log.ref_t[sel]
    ours_p = np.column_stack([np.interp(rt, t, pos[:, i]) for i in range(3)])
    ours_v = np.column_stack([np.interp(rt, t, vel[:, i]) for i in range(3)])
    dpos = ours_p - log.ref_pos[sel]
    dvel = ours_v - log.ref_vel[sel]
    asel = (log.att_t >= t[0]) & (log.att_t <= t[-1])
    if flight_only and log.flying(log.att_t).any():
        asel &= log.flying(log.att_t)
    idx = np.clip(np.searchsorted(t, log.att_t[asel]), 0, len(t) - 1)
    datt = np.array([quat_to_euler(quat[j]) - quat_to_euler(q) for j, q in zip(idx, log.att[asel])])
    datt[:, 2] = wrap(datt[:, 2])
    return {"t": rt, "dpos": dpos, "dvel": dvel, "att_t": log.att_t[asel], "datt_deg": np.degrees(datt),
            "pos_rms_m": float(np.sqrt(np.mean(np.sum(dpos**2, axis=1)))),
            "vel_rms_m_s": float(np.sqrt(np.mean(np.sum(dvel**2, axis=1)))),
            "att_rms_deg": np.sqrt(np.mean(np.degrees(datt) ** 2, axis=0))}


# ----- 3. Drag ---------------------------------------------------------------------------
def fit_drag(log, smooth=0.2, min_speed=0.3, min_range=1.0):
    """Fits the linear drag model on the flight: horizontal accelerometer = -k × body velocity.

    Thrust is along body z, so body x/y specific force comes from drag only (as in step 1).
    Our model says k = drag_coeff / mass. Returns k for x and y, and R² (1 = perfect fit).
    Without wind information, wind shows up as unexplained scatter.
    Hovering says nothing about drag, so only samples faster than `min_speed` are used,
    and an axis whose velocity spans less than `min_range` m/s is reported as None.
    """
    t = log.imu_t[log.flying(log.imu_t)]
    if len(t) < 100:
        raise ValueError("The drone never flies in this log")
    # Average over `smooth` seconds: we want the slow drag force, not the motor vibrations.
    bins = np.arange(t[0], t[-1], smooth)
    idx = np.digitize(log.imu_t, bins)
    rows = []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() < 5 or not log.flying(np.array([bins[b - 1]]))[0]:
            continue
        tc = bins[b - 1] + smooth / 2
        q = log.att[min(np.searchsorted(log.att_t, tc), len(log.att) - 1)]
        v_world = np.array([np.interp(tc, log.ref_t, log.ref_vel[:, i]) for i in range(3)])
        v_body = quat_to_rotmat(q).T @ v_world
        rows.append(np.concatenate([log.imu[m, :2].mean(axis=0), v_body[:2]]))
    rows = np.array(rows)
    rows = rows[np.hypot(rows[:, 2], rows[:, 3]) > min_speed]
    result = {"samples": len(rows), "f_body": rows[:, :2], "v_body": rows[:, 2:]}
    for i, axis in enumerate("xy"):
        f, v = rows[:, i], rows[:, 2 + i]
        if len(v) < 10 or np.ptp(v) < min_range:
            result[f"k_{axis}"] = result[f"r2_{axis}"] = None    # not enough motion on this axis
            continue
        A = np.column_stack([v, np.ones_like(v)])   # f = -k v + offset (offset: accel bias, wind)
        (slope, offset), *_ = np.linalg.lstsq(A, f, rcond=None)
        pred = A @ np.array([slope, offset])
        result[f"k_{axis}"] = float(-slope)
        result[f"r2_{axis}"] = float(1 - np.sum((f - pred) ** 2) / np.sum((f - f.mean()) ** 2))
    return result


# ----- Validating the validation ---------------------------------------------------------
def from_simulation(res, earth_field=(0.0, 0.2, -0.4)):
    """Packs one of OUR simulated flights as a FlightLog, so every check above can be run
    on data where the right answer is known (e.g. fit_drag must find drag_coeff / mass).
    The simulation must have flown IMU, GPS and barometer."""
    from .dynamics import POS, QUAT, VEL

    imu_t, imu = res.measurements["imu"]
    gps_t, gps = res.measurements["gps"]
    baro_t, baro = res.measurements["baro"]
    rot = [quat_to_rotmat(q) for q in res.states[:, QUAT]]
    mag = np.array([R.T @ np.array(earth_field) for R in rot])   # field seen by the body
    return FlightLog(imu_t, imu, gps_t, gps, baro_t, baro, res.t, mag,
                     res.t, res.states[:, POS], res.states[:, VEL], res.t, res.states[:, QUAT],
                     in_air_t=np.array([0.0]), in_air=np.array([True]))
