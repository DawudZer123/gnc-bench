"""Digital twin: fit our physics and sensor models to one real PX4 flight log.

From a .ulg file we identify what the flight actually shows:
  thrust      how much acceleration a unit of motor command gives, and how it fades as the
              battery voltage drops (the hover throttle creeps up during a flight);
  battery     voltage vs charge drawn, current vs thrust;
  balance     where the centre of gravity is, from which motors work harder in hover;
  drag        horizontal drag coefficient (needs some fast flight);
  sensors     vibration noise on the IMU in flight, barometer drift;
  rotation    roll/pitch/yaw effectiveness (inertia), ONLY if the flight excites it
              (stick doublets, PX4 autotune). A calm mission does not: we then keep a prior
              and say so;
and we copy the drone's own PX4 tuning (rate/attitude/position gains, filters, geometry),
so that PX4 SITL runs with the same configuration as the real drone.

All quantities are in PX4's body frame FRD unless stated otherwise.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .px4bridge import IrisParams, SensorNoise

GRAVITY = 9.81

# PX4 parameters copied from the real drone to SITL: everything that shapes how PX4 flies
# this airframe. Sensor calibrations, failsafes, radio and mission settings are left out.
PX4_PARAM_PREFIXES = ("MC_ROLL", "MC_PITCH", "MC_YAW", "MC_AIRMODE", "MPC_", "IMU_GYRO_CUTOFF",
                      "IMU_DGYRO_CUTOFF", "THR_MDL_FAC", "CA_ROTOR_COUNT")
PX4_PARAM_EXCLUDE = ("MPC_POS_MODE", "MPC_ALT_MODE", "MPC_YAW_MODE")


def px4_params_to_copy(params, rotors=4):
    out = {k: v for k, v in params.items()
           if k.startswith(PX4_PARAM_PREFIXES) and k not in PX4_PARAM_EXCLUDE}
    for i in range(rotors):
        for f in ("PX", "PY", "KM", "CT"):
            out[f"CA_ROTOR{i}_{f}"] = params[f"CA_ROTOR{i}_{f}"]
    return out


def px4_mixer(params, rotors=4):
    """PX4's own control allocation (pseudo-inverse + normalisation), as in ControlAllocator.

    Returns M (rotors x 4) such that motor commands u = M @ [roll, pitch, yaw, thrust]
    for normalised torque/thrust setpoints, before saturation handling.
    """
    E = np.zeros((6, rotors))
    for i in range(rotors):
        p = np.array([params[f"CA_ROTOR{i}_PX"], params[f"CA_ROTOR{i}_PY"], 0.0])
        axis = np.array([0.0, 0.0, -1.0])
        ct, km = params[f"CA_ROTOR{i}_CT"], params[f"CA_ROTOR{i}_KM"]
        E[:3, i] = ct * np.cross(p, axis) - ct * km * axis
        E[3:, i] = ct * axis
    mix = np.linalg.pinv(E)                            # rotors x 6
    nz = lambda c: max(np.sum(np.abs(c) > 1e-3), 1)    # noqa: E731
    rp = max(np.sqrt(np.sum(mix[:, 0] ** 2) / (nz(mix[:, 0]) / 2)),
             np.sqrt(np.sum(mix[:, 1] ** 2) / (nz(mix[:, 1]) / 2)))
    scale = [rp, rp, np.max(mix[:, 2]), 1.0, 1.0, np.sum(np.abs(mix[:, 5])) / nz(mix[:, 5])]
    mix = mix / np.array(scale)
    # PX4's thrust setpoint z is negative (up in FRD): command = -mix_z * thrust_z
    return np.column_stack([mix[:, 0], mix[:, 1], mix[:, 2], -mix[:, 5]])


# ----- reading the log ---------------------------------------------------------------------

@dataclass
class TwinData:
    """The log, resampled on one 50 Hz time grid while flying (PX4 frames, SI units)."""

    t: np.ndarray
    thrust_sp: np.ndarray      # normalised collective thrust command (0..1)
    torque_sp: np.ndarray      # (N, 3) normalised torque commands
    rate: np.ndarray           # (N, 3) rad/s
    rate_dot: np.ndarray       # (N, 3) rad/s² (PX4's filtered derivative)
    accel: np.ndarray          # (N, 3) specific force, averaged over each 20 ms step
    voltage: np.ndarray
    current: np.ndarray
    drawn_mah: np.ndarray
    motors_t: np.ndarray       # motor commands (logged at a lower rate)
    motors: np.ndarray
    imu_t: np.ndarray          # raw IMU (for vibration)
    imu: np.ndarray            # (M, 6) accel + gyro
    baro_bias_t: np.ndarray    # PX4 EKF2's estimate of the barometer bias
    baro_bias: np.ndarray
    params: dict
    airframe: int
    hardware: str


def read_log(path, margin=5.0):
    """Reads what the twin needs from a .ulg file (pip install pyulog)."""
    from pyulog import ULog

    ulog = ULog(str(path))
    t0 = np.int64(ulog.start_timestamp)

    def topic(name):
        for d in ulog.data_list:
            if d.name == name and d.multi_id == 0:
                return (d.data["timestamp"].astype(np.int64) - t0) / 1e6, d.data
        raise KeyError(f"topic {name} not in the log")

    def vec(d, prefix, n=3):
        return np.column_stack([d[f"{prefix}[{i}]"] for i in range(n)])

    tl, land = topic("vehicle_land_detected")
    air = np.flatnonzero(land["landed"] == 0)
    if len(air) == 0:
        raise ValueError("The drone never flies in this log")
    t_up, t_down = tl[air[0]] + margin, tl[min(air[-1] + 1, len(tl) - 1)] - margin
    ta, av = topic("vehicle_angular_velocity")
    keep = (ta > t_up) & (ta < t_down)
    t = ta[keep]
    interp = lambda tt, x: np.interp(t, tt, x)                             # noqa: E731
    tt, th = topic("vehicle_thrust_setpoint")
    tq, tor = topic("vehicle_torque_setpoint")
    ts, sc = topic("sensor_combined")
    imu = np.column_stack([vec(sc, "accelerometer_m_s2"), vec(sc, "gyro_rad")])
    # Average the raw accelerometer over each 20 ms step (removes most of the vibration).
    csum = np.vstack([np.zeros(3), np.cumsum(imu[:, :3], axis=0)])
    dt = np.median(np.diff(t))
    lo, hi = np.searchsorted(ts, t - dt / 2), np.searchsorted(ts, t + dt / 2)
    accel = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)[:, None]
    tb, bat = topic("battery_status")
    tm, mo = topic("actuator_motors")
    rotors = int(ulog.initial_parameters.get("CA_ROTOR_COUNT", 4))
    mkeep = (tm > t_up) & (tm < t_down)
    try:
        tbb, bb = topic("estimator_baro_bias")
        baro_t, baro_bias = tbb, bb["bias"]
    except KeyError:
        baro_t, baro_bias = np.zeros(0), np.zeros(0)
    ikeep = (ts > t_up) & (ts < t_down)
    return TwinData(
        t=t, thrust_sp=-interp(tt, th["xyz[2]"]),
        torque_sp=np.column_stack([interp(tq, tor[f"xyz[{i}]"]) for i in range(3)]),
        rate=vec(av, "xyz")[keep], rate_dot=vec(av, "xyz_derivative")[keep], accel=accel,
        voltage=interp(tb, bat["voltage_v"]), current=interp(tb, bat["current_a"]),
        drawn_mah=interp(tb, bat["discharged_mah"]),
        motors_t=tm[mkeep], motors=vec(mo, "control", rotors)[mkeep],
        imu_t=ts[ikeep], imu=imu[ikeep], baro_bias_t=baro_t, baro_bias=np.asarray(baro_bias),
        params=dict(ulog.initial_parameters), airframe=int(ulog.initial_parameters.get("SYS_AUTOSTART", 0)),
        hardware=ulog.msg_info_dict.get("ver_hw", "?"))


# ----- identification ----------------------------------------------------------------------

def smooth(x, n):
    """Centred moving average over n samples (each column)."""
    k = np.ones(n) / n
    if x.ndim == 1:
        return np.convolve(x, k, mode="same")
    return np.column_stack([np.convolve(c, k, mode="same") for c in x.T])


def first_order_lag(x, tau, dt):
    if tau <= 0:
        return x
    a = np.exp(-dt / tau)
    y = np.empty_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = a * y[k - 1] + (1 - a) * x[k]
    return y


def fit_thrust(d, window=1.0):
    """Thrust gain K (m/s² per unit command) and how it scales with battery voltage.

    Over windows of `window` seconds the vertical accelerations average out, so
    K = (upward specific force) / (thrust command). Fitting log K against log V gives
    thrust ∝ V^alpha.
    """
    n = max(int(window / np.median(np.diff(d.t))), 1)
    up = smooth(-d.accel[:, 2], n)[n:-n]
    cmd = smooth(d.thrust_sp, n)[n:-n]
    v = smooth(d.voltage, n)[n:-n]
    ok = cmd > 0.05
    gain = up[ok] / cmd[ok]
    alpha, log_k_full = np.polyfit(np.log(v[ok] / v[ok].max()), np.log(gain), 1)
    pred = np.exp(log_k_full) * (v[ok] / v[ok].max()) ** alpha
    return {"gain_full": float(np.exp(log_k_full)), "voltage_full": float(v[ok].max()),
            "voltage_exponent": float(alpha), "hover_thrust": float(np.median(d.thrust_sp)),
            "gain_spread": float(np.std(gain - pred) / np.mean(gain))}


def fit_battery(d):
    """Voltage = V0 - a × charge drawn - R × current, and current = i0 + i1 × thrust command."""
    A = np.column_stack([np.ones_like(d.voltage), -d.drawn_mah, -d.current])
    (v0, per_mah, resistance), *_ = np.linalg.lstsq(A, d.voltage, rcond=None)
    (i0, i1), *_ = np.linalg.lstsq(np.column_stack([np.ones_like(d.thrust_sp), d.thrust_sp]),
                                   d.current, rcond=None)
    if per_mah < 0 or resistance < 0:      # collinear data: fall back to charge only
        (v0, per_mah), *_ = np.linalg.lstsq(A[:, :2], d.voltage, rcond=None)
        resistance = 0.0
    return {"v0": float(v0), "volts_per_mah": float(per_mah), "resistance": float(resistance),
            "current_i0": float(i0), "current_i1": float(i1),
            "drawn_mah": float(d.drawn_mah[-1] - d.drawn_mah[0])}


def fit_balance(d, mixer):
    """Centre of gravity from the hover motor commands.

    In steady hover the thrusts balance the weight about the CG, so the CG sits at the
    thrust-weighted centroid of the rotor positions. Returned as a fraction of the rotor
    arm's x/y projection (multiply by it for metres), FRD.
    """
    u = d.motors
    mean = u.mean(axis=0)
    signs = np.sign([[d.params[f"CA_ROTOR{i}_PX"], d.params[f"CA_ROTOR{i}_PY"]]
                     for i in range(len(mean))])
    cg = signs.T @ mean / mean.sum()
    trim = d.torque_sp.mean(axis=0)
    # Yaw: in hover the counter-clockwise and clockwise rotors' reaction torques cancel.
    # If one set works harder, the other set makes more torque per unit thrust (props,
    # motors or a slightly tilted motor): ratio = CCW torque coefficient / CW one.
    ccw = np.array([d.params[f"CA_ROTOR{i}_KM"] > 0 for i in range(len(mean))])
    yaw_ratio = mean[~ccw].sum() / mean[ccw].sum()
    return {"motor_mean": mean.tolist(), "motor_max": u.max(axis=0).tolist(),
            "motor_p99": np.percentile(u, 99, axis=0).tolist(),
            "cg_fraction": cg.tolist(), "yaw_torque_ratio": float(yaw_ratio),
            "torque_trim": trim.tolist()}


def fit_rotation(d, mixer, min_rate_std=0.3, lags=(0.0, 0.02, 0.04, 0.06, 0.08, 0.1)):
    """Angular acceleration per unit torque command (rad/s²) and the motor time constant.

    Needs real rotations: if the rate on an axis barely moves (std below `min_rate_std`
    rad/s), that axis is not identifiable from this flight and is returned as None.
    """
    dt = float(np.median(np.diff(d.t)))
    out = {}
    best_tau = []
    for i, axis in enumerate(("roll", "pitch", "yaw")):
        if np.std(d.rate[:, i]) < min_rate_std:
            out[axis] = None
            continue
        y = smooth(d.rate_dot[:, i], 3)
        fits = []
        for tau in lags:
            x = first_order_lag(d.torque_sp[:, i], tau, dt)
            A = np.column_stack([x, np.ones_like(x)])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - np.var(y - A @ c) / np.var(y)
            fits.append((r2, tau, c[0]))
        r2, tau, gain = max(fits)
        out[axis] = {"gain": float(gain), "r2": float(r2), "motor_tau": float(tau)}
        best_tau.append(tau)
    out["motor_tau"] = float(np.median(best_tau)) if best_tau else None
    return out


def fit_vibration(d, n=16, cutoff=30.0, sim_rate=250.0):
    """IMU vibration in flight (motors spinning).

    `accel`/`gyro`: everything faster than ~10 Hz, as a pilot would quote it. Most of it is at
    the propeller frequency (~90 Hz here), which PX4's low-pass filters (IMU_*_CUTOFF, `cutoff`)
    remove. What reaches the controllers and EKF2 is the 10 Hz - cutoff band, so the twin
    replays THAT: `accel_sim`/`gyro_sim` are the white-noise levels at the simulator's
    `sim_rate` with the same energy in that band.
    """
    from scipy.signal import butter, sosfiltfilt

    hf = d.imu - smooth(d.imu, n)
    std = hf[n:-n].std(axis=0)
    fs = 1.0 / np.median(np.diff(d.imu_t))
    band = sosfiltfilt(butter(4, [10.0, cutoff], btype="band", fs=fs, output="sos"), d.imu, axis=0)
    in_band = band[n:-n].std(axis=0) * np.sqrt((sim_rate / 2) / (cutoff - 10.0))
    return {"accel": std[:3].tolist(), "gyro": std[3:].tolist(),
            "accel_sim": in_band[:3].tolist(), "gyro_sim": in_band[3:].tolist()}


def fit_baro(d):
    """Barometer drift: how far PX4's own baro-bias estimate wanders during the flight."""
    if len(d.baro_bias) < 10:
        return {"walk": None, "total_drift_m": None}
    m = (d.baro_bias_t > d.t[0]) & (d.baro_bias_t < d.t[-1])
    b, tb = d.baro_bias[m], d.baro_bias_t[m]
    drift = float(b[-1] - b[0])
    return {"walk": float(abs(drift) / np.sqrt(tb[-1] - tb[0])), "total_drift_m": drift}


# ----- the twin ----------------------------------------------------------------------------

@dataclass
class DigitalTwin:
    """Everything identified from one flight, plus the PX4 configuration to fly it with."""

    source: str
    hardware: str
    airframe: int
    thrust: dict
    battery: dict
    balance: dict
    rotation: dict
    drag: dict
    vibration: dict
    baro: dict
    px4_params: dict = field(default_factory=dict)

    # Priors for what one calm flight cannot show (S500-class quad, 4S).
    PRIOR_INERTIA_PER_KG = (0.017, 0.017, 0.03)    # kg·m² per kg of mass
    PRIOR_MOTOR_TAU = 0.04
    PRIOR_DRAG_K = 0.2

    def quad_params(self, mass=1.8, arm=0.24, torque_coeff=0.016):
        """Our physics for this drone. Only ratios are observable, so `mass` (kg) and `arm`
        (centre to rotor, m) set the scale: change them if you know the real values.
        """
        d = arm / np.sqrt(2)
        max_thrust = self.thrust["gain_full"] * mass / 4          # N per rotor, full battery
        signs = np.sign([[self.px4_params[f"CA_ROTOR{i}_PX"], self.px4_params[f"CA_ROTOR{i}_PY"]]
                         for i in range(4)])
        cg_frd = np.asarray(self.balance["cg_fraction"]) * d
        rotors_flu = np.column_stack([signs[:, 0] * d - cg_frd[0],
                                      -(signs[:, 1] * d - cg_frd[1]), np.zeros(4)])
        spins = np.sign([self.px4_params[f"CA_ROTOR{i}_KM"] for i in range(4)])  # KM > 0: CCW
        spins = np.where(spins > 0, spins * self.balance.get("yaw_torque_ratio", 1.0), spins)
        tau = self.rotation.get("motor_tau") or self.PRIOR_MOTOR_TAU
        params = TwinParams(mass=mass, inertia=np.diag(np.array(self.PRIOR_INERTIA_PER_KG) * mass),
                            max_thrust=max_thrust, torque_coeff=torque_coeff,
                            drag_coeff=self.drag_k() * mass, motor_tau=tau,
                            rotors=rotors_flu, spins=spins)
        # Identified axes: inertia = torque from one unit of PX4 torque command / measured gain.
        mix = px4_mixer(self.px4_params)
        inertia = np.diag(params.inertia).copy()
        for i, axis in enumerate(("roll", "pitch", "yaw")):
            fit = self.rotation.get(axis)
            if fit is not None and fit["gain"] > 0:
                torque = params.mixer()[1:] @ (mix[:, i] * max_thrust)
                inertia[i] = abs(torque[i]) / fit["gain"]
        params.inertia = np.diag(inertia)
        return params

    def drag_k(self):
        ks = [self.drag.get(f"k_{a}") for a in "xy"
              if self.drag.get(f"k_{a}") is not None and (self.drag.get(f"r2_{a}") or 0) > 0.5]
        return float(np.mean(ks)) if ks else self.PRIOR_DRAG_K

    def sensor_noise(self):
        """Sensor models with the vibration and baro drift seen in flight."""
        v = self.vibration
        return SensorNoise(accel_flight=float(np.mean(v.get("accel_sim", v["accel"]))),
                           gyro_flight=float(np.mean(v.get("gyro_sim", v["gyro"]))),
                           baro_walk=self.baro.get("walk") or 0.0)

    def battery_model(self):
        return Battery(v_full=self.thrust["voltage_full"], exponent=self.thrust["voltage_exponent"],
                       volts_per_mah=self.battery["volts_per_mah"],
                       resistance=self.battery["resistance"], i0=self.battery["current_i0"],
                       i1=self.battery["current_i1"], v0=self.battery["v0"])

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=float))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class TwinParams(IrisParams):
    """Our quad model with an explicit rotor layout (relative to the CG) and spin directions."""

    rotors: np.ndarray = None
    spins: np.ndarray = None

    def rotor_positions(self):
        return self.rotors

    def rotor_spins(self):
        return self.spins


@dataclass
class Battery:
    """Battery seen by the motors: thrust ∝ (V / V_full)^exponent.

    V = v0 - volts_per_mah × charge drawn - resistance × current, current = i0 + i1 × command.
    """

    v_full: float
    exponent: float
    volts_per_mah: float
    resistance: float
    i0: float
    i1: float
    v0: float
    drawn_mah: float = 0.0
    voltage: float = None

    def step(self, command, dt):
        current = max(self.i0 + self.i1 * command, 0.0)
        self.drawn_mah += current * dt / 3.6
        self.voltage = self.v0 - self.volts_per_mah * self.drawn_mah - self.resistance * current
        return self.thrust_scale()

    def thrust_scale(self):
        v = self.voltage if self.voltage is not None else self.v_full
        return (v / self.v_full) ** self.exponent


def build_twin(path):
    """Identify everything this log allows. Returns a DigitalTwin."""
    from .flightlog import fit_drag, load_px4_log

    d = read_log(path)
    mixer = px4_mixer(d.params)
    try:
        drag = fit_drag(load_px4_log(path))
        drag = {k: drag[k] for k in ("k_x", "r2_x", "k_y", "r2_y", "samples")}
    except ValueError:
        drag = {"k_x": None, "r2_x": None, "k_y": None, "r2_y": None, "samples": 0}
    return DigitalTwin(
        source=Path(path).name, hardware=d.hardware, airframe=d.airframe,
        thrust=fit_thrust(d), battery=fit_battery(d), balance=fit_balance(d, mixer),
        rotation=fit_rotation(d, mixer), drag=drag, vibration=fit_vibration(d, cutoff=d.params.get("IMU_GYRO_CUTOFF", 30.0)), baro=fit_baro(d),
        px4_params=px4_params_to_copy(d.params))


# ----- replaying the real mission on PX4 SITL ----------------------------------------------

@dataclass
class RealFlight:
    """What the real drone did, in the bench frames, times from the first setpoint (s)."""

    t: np.ndarray
    position: np.ndarray       # ENU, relative to the take-off point
    tilt_deg: np.ndarray
    thrust: np.ndarray         # collective command, normalised
    motors_t: np.ndarray
    motors: np.ndarray
    setpoint_t: np.ndarray
    setpoint_pos: np.ndarray   # ENU, relative to the take-off point
    setpoint_vel: np.ndarray


def read_real_flight(path):
    from pyulog import ULog

    from .flightlog import NED_TO_ENU

    ulog = ULog(str(path))
    t0 = np.int64(ulog.start_timestamp)

    def topic(name):
        d = next(d for d in ulog.data_list if d.name == name and d.multi_id == 0).data
        return (d["timestamp"].astype(np.int64) - t0) / 1e6, d

    def fill(x):
        """Replace NaN by the previous valid value (or the next one at the start)."""
        x = x.copy()
        for c in range(x.shape[1]):
            ok = np.flatnonzero(np.isfinite(x[:, c]))
            if len(ok) == 0:
                x[:, c] = 0.0
                continue
            idx = np.maximum.accumulate(np.where(np.isfinite(x[:, c]), np.arange(len(x)), 0))
            idx[:ok[0]] = ok[0]
            x[:, c] = x[idx, c]
        return x

    ts, sp = topic("trajectory_setpoint")
    start = ts[0]
    tl, lp = topic("vehicle_local_position")
    origin = np.array([lp["x"][0], lp["y"][0], lp["z"][0]])
    pos = (np.column_stack([lp["x"], lp["y"], lp["z"]]) - origin) @ NED_TO_ENU.T
    sp_pos = (fill(np.column_stack([sp[f"position[{i}]"] for i in range(3)])) - origin) @ NED_TO_ENU.T
    sp_vel = np.nan_to_num(np.column_stack([sp[f"velocity[{i}]"] for i in range(3)])) @ NED_TO_ENU.T
    tq, att = topic("vehicle_attitude")
    q = np.column_stack([att[f"q[{i}]"] for i in range(4)])
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
    tt, th = topic("vehicle_thrust_setpoint")
    tm, mo = topic("actuator_motors")
    return RealFlight(
        t=tl - start, position=pos, tilt_deg=np.interp(tl, tq, tilt),
        thrust=np.interp(tl, tt, -th["xyz[2]"]), motors_t=tm - start,
        motors=np.column_stack([mo[f"control[{i}]"] for i in range(4)]),
        setpoint_t=ts - start, setpoint_pos=sp_pos, setpoint_vel=sp_vel)


def logged_mission(real):
    """The real flight's setpoints as a PX4 SITL mission: (function t -> pos, vel, acc; length)."""
    def reference(t):
        pos = np.array([np.interp(t, real.setpoint_t, real.setpoint_pos[:, i]) for i in range(3)])
        vel = np.array([np.interp(t, real.setpoint_t, real.setpoint_vel[:, i]) for i in range(3)])
        return pos, vel, np.zeros(3)
    return reference, float(real.setpoint_t[-1])
