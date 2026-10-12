"""Fly the real PX4 autopilot on our physics (software-in-the-loop, lockstep).

PX4 SITL is the real flight code compiled for a computer. Instead of reading real sensors,
it connects over TCP to a simulator and exchanges MAVLink messages:

    our simulator  --HIL_SENSOR (IMU, mag, baro), HIL_GPS-->  PX4
    our simulator  <--HIL_ACTUATOR_CONTROLS (4 motor commands)--  PX4

In lockstep mode PX4 has no clock of its own: its time is the timestamp of our HIL_SENSOR
messages. Each sensor message lets PX4 run one step, and it answers with the motor commands
for that step. The flight is therefore exactly reproducible, and can run faster or slower
than real time without changing anything PX4 sees.

Frames: our simulator is ENU / FLU (z up), PX4 is NED / FRD (z down).
"""

import os
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dynamics import MOTORS, OMEGA, POS, QUAT, VEL, initial_state, state_derivative
from .geo import Home, local_to_lla, magnetic_field, pressure_at
from .integrators import rk4_step
from .math3d import quat_to_euler, quat_to_rotmat
from .params import QuadParams

NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # its own inverse
FRD_TO_FLU = np.diag([1.0, -1.0, -1.0])                                         # its own inverse

# HIL_SENSOR.fields_updated bits (PX4 SimulatorMavlink.hpp)
ACCEL, GYRO, MAG, BARO = 0b111, 0b111000, 0b111000000, 0b1101000000000

SIM_PORT, OFFBOARD_PORT = 4560, 14540    # + instance number


@dataclass
class IrisParams(QuadParams):
    """The 'Iris' quadcopter that PX4's none_iris airframe is tuned for.

    PX4's gains are tuned for this mass, inertia and rotor layout. Rotors in PX4 order:
    0 front-right (CCW), 1 rear-left (CCW), 2 front-left (CW), 3 rear-right (CW), same as ours.
    """

    mass: float = 1.5
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.029, 0.029, 0.055]))
    max_thrust: float = 7.0          # N per rotor (thrust-to-weight about 1.9)
    torque_coeff: float = 0.016
    drag_coeff: float = 0.3
    motor_tau: float = 0.0125

    def rotor_positions(self):
        # PX4 CA_ROTORi_PX/PY are in FRD; FLU flips y.
        return np.array([[0.1515, -0.245, 0], [-0.1515, 0.1875, 0],
                         [0.1515, 0.245, 0], [-0.1515, -0.1875, 0]])


@dataclass
class SensorNoise:
    """Noise of the simulated sensors sent to PX4 (1 sigma, per sample)."""

    accel: float = 0.02      # m/s²   (real boards measured at step 4: 0.014-0.016)
    gyro: float = 0.002      # rad/s  (real boards: 0.001-0.002)
    mag: float = 0.002       # gauss
    baro: float = 0.1        # m, converted to pressure
    gps_pos: tuple = (0.3, 0.3, 0.6)   # m, slowly wandering error (atmosphere, multipath)
    gps_tau: float = 60.0    # s, correlation time of that error
    gps_white: float = 0.05  # m, fast receiver noise on top
    flow: float = 0.02       # rad/s, optical flow angular rate
    range: float = 0.02      # m, downward rangefinder
    baro_walk: float = 0.0   # m/√s, barometer drift in flight (random walk; prop wash, heating)
    accel_flight: float = 0.0  # m/s², extra vibration while the motors spin
    gyro_flight: float = 0.0   # rad/s
    gps_vel: float = 0.05    # m/s


@dataclass
class Px4Flight:
    """Everything recorded during one PX4 flight. Times are simulation seconds."""

    t: np.ndarray
    states: np.ndarray        # true state (N, 17)
    commands: np.ndarray      # motor commands from PX4, 0..1 (N, 4)
    wind: np.ndarray          # (N, 3)
    reference: np.ndarray     # position setpoint sent to PX4, ENU (N, 3), NaN before take-off
    px4_t: np.ndarray         # PX4's own position estimate (LOCAL_POSITION_NED), ENU
    px4_pos: np.ndarray
    px4_vel: np.ndarray
    messages: list            # PX4 STATUSTEXT messages: (t, text)
    modes: list               # PX4 flight mode changes: (t, main mode, sub mode, armed)
    armed_at: float = np.nan
    missed_steps: int = 0     # lockstep steps PX4 did not answer in time


def ground_contact(x, params):
    """A flat, rigid ground at z = 0: the drone sits on it until thrust exceeds weight."""
    if x[POS][2] > 0.0:
        return x, False
    R = quat_to_rotmat(x[QUAT])
    lift = (R @ np.array([0.0, 0.0, np.sum(x[MOTORS])]))[2]
    if x[POS][2] < 0.0 or lift <= params.mass * params.gravity:
        yaw = quat_to_euler(x[QUAT])[2]
        x = x.copy()
        x[POS][2] = 0.0
        x[VEL] = 0.0
        x[OMEGA] = 0.0
        x[QUAT] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        return x, True
    return x, False


class Px4Bridge:
    """Our physics + sensor models, served to PX4 over its simulator link.

    wind:   a Wind object (optional)
    faults: SensorFault objects on "gps" (measurement = ENU [p, v]), as in the rest of the bench
    """

    def __init__(self, params=None, home=Home(), seed=0, instance=0, wind=None, faults=(),
                 noise=SensorNoise(), yaw0=0.0, flow=False, battery=None):
        from pymavlink.dialects.v20 import common as mavlink   # pip install pymavlink

        self.mavlink = mavlink
        self.params = params or IrisParams()
        self.home = home
        self.rng = np.random.default_rng(seed)
        self.port = SIM_PORT + instance
        self.wind = wind
        self.faults = list(faults)
        self.noise = noise
        self.flow = flow           # also send a downward optical flow camera + rangefinder
        self.battery = battery     # optional Battery: thrust fades as the voltage drops
        self.baro_bias = 0.0
        self.x = initial_state(quat=(np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)))
        self.u = np.zeros(4)
        self.t = 0.0
        self.dt = 0.004            # PX4 SITL expects the IMU at 250 Hz
        self.substeps = 2          # physics at 500 Hz, as in sim.py
        self.on_ground = True
        self.last_gps = None
        self.mission_start = np.inf   # set when the mission begins; faults are timed from it
        self.field_enu, self.declination, _ = magnetic_field(home.lat, home.lon)
        self.accel_bias = self.rng.normal(0.0, 0.02, 3)
        self.gyro_bias = self.rng.normal(0.0, 0.002, 3)
        self.gps_error = self.rng.normal(0.0, 1.0, 3) * np.asarray(noise.gps_pos)
        self.sock = None
        self.link = None
        self.lockstep = False
        self._pending = []         # messages already received, not yet read
        self.missed_steps = 0      # steps PX4 did not answer (we then keep the last commands)
        self.consecutive_missed = 0

    # ---- connection ---------------------------------------------------------------------------

    def listen(self):
        """Open the TCP port PX4 connects to (call before starting PX4)."""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", self.port))
        self.server.listen(1)

    def accept(self, timeout=60.0):
        self.server.settimeout(timeout)
        self.sock, _ = self.server.accept()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(5.0)
        self.link = self.mavlink.MAVLink(_SocketWriter(self.sock), srcSystem=1, srcComponent=51)
        self.link.robust_parsing = True

    def close(self):
        for s in (self.sock, getattr(self, "server", None)):
            if s is not None:
                s.close()

    # ---- one lockstep step --------------------------------------------------------------------

    def step(self):
        """Send this step's sensors, wait for PX4's motor commands, advance the physics."""
        self.t += self.dt
        k = int(round(self.t / self.dt))
        self._send_sensors(k)
        self._receive_controls()
        self._integrate()

    def _integrate(self):
        p = self.params
        cmd = p.max_thrust * np.clip(self.u, 0.0, 1.0)      # linear thrust curve, as PX4 assumes
        if self.battery is not None:
            cmd = cmd * self.battery.step(float(np.mean(np.clip(self.u, 0, 1))), self.dt)
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            w = self.wind.step(h) if self.wind is not None else np.zeros(3)
            self.current_wind = w
            f_wind = p.drag_coeff * w
            self.x = rk4_step(lambda x: state_derivative(x, cmd, p, f_wind), self.x, h)
            self.x[QUAT] /= np.linalg.norm(self.x[QUAT])
            self.x, self.on_ground = ground_contact(self.x, p)
        self.f_wind = f_wind

    def _specific_force_flu(self):
        p = self.params
        R = quat_to_rotmat(self.x[QUAT])
        if self.on_ground:
            accel = np.zeros(3)
        else:
            cmd = p.max_thrust * np.clip(self.u, 0.0, 1.0)
            accel = state_derivative(self.x, cmd, p, getattr(self, "f_wind", np.zeros(3)))[VEL]
        return R.T @ (accel + np.array([0.0, 0.0, p.gravity]))

    def _send_sensors(self, k):
        n, rng = self.noise, self.rng
        R = quat_to_rotmat(self.x[QUAT])
        spinning = float(np.mean(np.nan_to_num(self.u))) > 0.05     # vibration and baro drift need the motors on
        sa = np.hypot(n.accel, n.accel_flight) if spinning else n.accel
        sg = np.hypot(n.gyro, n.gyro_flight) if spinning else n.gyro
        accel = FRD_TO_FLU @ (self._specific_force_flu() + self.accel_bias + rng.normal(0, sa, 3))
        gyro = FRD_TO_FLU @ (self.x[OMEGA] + self.gyro_bias + rng.normal(0, sg, 3))
        mag = FRD_TO_FLU @ (R.T @ self.field_enu + rng.normal(0, n.mag, 3))
        if n.baro_walk and spinning and k % 5 == 0:
            self.baro_bias += rng.normal(0, n.baro_walk * np.sqrt(5 * self.dt))
        altitude = self.home.alt + self.x[POS][2] + self.baro_bias + rng.normal(0, n.baro)
        pressure, temperature = pressure_at(altitude)
        fields = ACCEL | GYRO
        if k % 2 == 0:
            fields |= MAG                 # 125 Hz
        if k % 5 == 0:
            fields |= BARO                # 50 Hz
        t_us = int(round(self.t * 1e6))
        self.link.hil_sensor_send(t_us, *accel, *gyro, *mag, pressure, 0.0, altitude, temperature,
                                  fields, 0)
        if k % 25 == 0:                   # GPS at 10 Hz
            self._send_gps(t_us)
        if self.flow and k % 5 == 0:      # flow + rangefinder at 50 Hz
            self._send_flow(t_us, 5 * self.dt)

    def _send_gps(self, t_us):
        n = self.noise
        # A real GPS error is not white noise: it wanders slowly (Gauss-Markov, 0.1 s steps).
        a = np.exp(-0.1 / n.gps_tau)
        sigma = np.asarray(n.gps_pos)
        self.gps_error = a * self.gps_error + np.sqrt(1 - a * a) * sigma * self.rng.normal(0, 1, 3)
        z = np.concatenate([self.x[POS] + self.gps_error + self.rng.normal(0, n.gps_white, 3),
                            self.x[VEL] + self.rng.normal(0, n.gps_vel, 3)])
        tm = self.t - self.mission_start         # fault times count from the start of the mission
        active = [f for f in self.faults if f.sensor == "gps" and f.is_active(tm)]
        if active:
            z = active[0].apply(z, self.last_gps, tm)
        else:
            self.last_gps = z
        if z is None or np.any(np.isnan(z)):
            return                        # outage: PX4 simply receives nothing
        lat, lon, alt = local_to_lla(z[:3], self.home)
        v_ned = NED_TO_ENU @ z[3:]
        speed = np.hypot(v_ned[0], v_ned[1])
        cog = int(np.degrees(np.arctan2(v_ned[1], v_ned[0])) % 360 * 100) if speed > 0.2 else 65535
        self.link.hil_gps_send(t_us, 3, int(lat * 1e7), int(lon * 1e7), int(alt * 1000),
                               int(n.gps_pos[0] * 100), int(n.gps_pos[2] * 100), int(speed * 100),
                               *(int(v * 100) for v in v_ned), cog, 12, 0, 0)

    def _send_flow(self, t_us, dt):
        """Downward camera: angular rate of the ground texture, the way PX4 expects it.

        PX4 (EKF2) models the gyro-compensated flow as [v_y, -v_x] / range in the FRD body frame;
        the raw flow also contains the body rotation. The sensor sits 10 cm above the ground
        when landed. We send no integrated gyro (NaN), so PX4 uses its own.
        """
        n, rng = self.noise, self.rng
        R = quat_to_rotmat(self.x[QUAT])
        v_frd = FRD_TO_FLU @ (R.T @ self.x[VEL])
        w_frd = FRD_TO_FLU @ self.x[OMEGA]
        height = self.x[POS][2] + 0.1
        distance = height / max(R[2, 2], 0.1)          # along the body z axis
        los = np.array([v_frd[1], -v_frd[0]]) / distance + rng.normal(0, n.flow, 2)
        pixel = dt * (w_frd[:2] - los)
        measured = distance + rng.normal(0, n.range)
        nan = float("nan")
        self.link.hil_optical_flow_send(t_us, 0, int(dt * 1e6), *pixel, nan, nan, nan, 20, 255,
                                        0, measured)
        self.link.distance_sensor_send(int(t_us / 1000), 5, 1000, int(measured * 100), 0, 0,
                                       25, 4)          # 25 = facing down; covariance in cm²

    def _receive_controls(self):
        """In lockstep, wait until PX4 has answered this step.

        PX4 occasionally skips an answer. Waiting forever would then freeze both sides (each
        waits for the other), so after a short wall-clock timeout we keep the previous motor
        commands and send the next step, which wakes PX4 up again.
        """
        timeout = 1.0 if self.lockstep else 0.0
        while True:
            msg = self._read_message(timeout)
            if msg is None:
                if self.lockstep:
                    self.missed_steps += 1
                    self.consecutive_missed += 1
                    if self.consecutive_missed > 20:
                        raise TimeoutError("PX4 stopped answering")
                return
            if msg.get_type() == "HIL_ACTUATOR_CONTROLS":
                self.u = np.array(msg.controls[:4]) if msg.mode & 128 else np.zeros(4)  # 128 = armed
                self.lockstep = True
                self.consecutive_missed = 0
                return

    def _read_message(self, timeout):
        """Next MAVLink message from PX4, or None if nothing arrives within `timeout` seconds."""
        if self._pending:
            return self._pending.pop(0)
        self.sock.settimeout(timeout if timeout > 0 else None)
        if timeout == 0:
            self.sock.setblocking(False)
        try:
            data = self.sock.recv(4096)
        except (BlockingIOError, socket.timeout):
            return None
        if not data:
            raise ConnectionError("PX4 closed the simulator link")
        self._pending = self.link.parse_buffer(data) or []
        return self._pending.pop(0) if self._pending else None


class _SocketWriter:
    def __init__(self, sock):
        self.sock = sock

    def write(self, data):
        self.sock.sendall(data)


class OffboardLink:
    """Talks to PX4 like a companion computer would: parameters, arming, position setpoints.

    Uses PX4's offboard MAVLink port (14540 + instance), the one MAVSDK and ROS use.
    """

    def __init__(self, instance=0):
        from pymavlink import mavutil

        self.mavutil = mavutil
        self.conn = mavutil.mavlink_connection(f"udpin:127.0.0.1:{OFFBOARD_PORT + instance}",
                                               source_system=255, source_component=190)
        self.target = (instance + 1, 1)
        self.connected = False
        self.armed = False
        self.mode = None
        self.modes = []           # (t, PX4 main mode, sub mode, armed) at each change
        self.positions = []       # (t, pos ENU, vel ENU) from PX4's estimator
        self.messages = []

    def poll(self, t):
        """Read everything PX4 sent since last call (non-blocking)."""
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            if kind == "HEARTBEAT" and msg.get_srcComponent() == 1:
                self.connected = True
                self.armed = bool(msg.base_mode & 128)
                mode = ((msg.custom_mode >> 16) & 0xFF, (msg.custom_mode >> 24) & 0xFF)
                if not self.modes or self.modes[-1][1:] != (*mode, self.armed):
                    self.modes.append((t, *mode, self.armed))
                self.mode = mode[0]                               # PX4 main mode (6 = offboard)
            elif kind == "LOCAL_POSITION_NED":
                self.positions.append((msg.time_boot_ms / 1000.0,
                                       NED_TO_ENU @ [msg.x, msg.y, msg.z],
                                       NED_TO_ENU @ [msg.vx, msg.vy, msg.vz]))
            elif kind == "STATUSTEXT":
                self.messages.append((t, msg.text))

    def close(self):
        self.conn.close()        # frees the UDP port for the next flight on this instance

    def set_param(self, name, value):
        kind = (self.mavutil.mavlink.MAV_PARAM_TYPE_INT32 if isinstance(value, int)
                else self.mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        if isinstance(value, int):          # PX4 sends integers bit-cast into the float field
            value = np.frombuffer(np.int32(value).tobytes(), dtype=np.float32)[0]
        self.conn.mav.param_set_send(*self.target, name.encode(), float(value), kind)

    def command(self, command, *params):
        p = list(params) + [0.0] * (7 - len(params))
        self.conn.mav.command_long_send(*self.target, command, 0, *p)

    def arm(self):
        self.command(self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)

    def set_offboard(self):
        # base mode: custom; PX4 main mode 6 = OFFBOARD
        self.command(self.mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1.0, 6.0)

    def land(self):
        self.command(self.mavutil.mavlink.MAV_CMD_NAV_LAND)

    def setpoint(self, t, pos_enu, vel_enu=(0, 0, 0), acc_enu=(0, 0, 0), yaw=None):
        """Position setpoint with velocity and acceleration feedforward, sent in NED."""
        p = NED_TO_ENU @ np.asarray(pos_enu, float)
        v = NED_TO_ENU @ np.asarray(vel_enu, float)
        a = NED_TO_ENU @ np.asarray(acc_enu, float)
        ignore = 0x800                         # type_mask: ignore yaw rate
        if yaw is None:
            ignore |= 0x400                    # ... and yaw
        self.conn.mav.set_position_target_local_ned_send(
            int(t * 1000), *self.target, self.mavutil.mavlink.MAV_FRAME_LOCAL_NED, ignore,
            *p, *v, *a, 0.0 if yaw is None else np.pi / 2 - yaw, 0)


def find_px4(px4_dir=None):
    """Locate the PX4 SITL build: argument, $PX4_DIR, or ~/PX4-Autopilot."""
    candidates = [px4_dir, os.environ.get("PX4_DIR"), Path.home() / "PX4-Autopilot"]
    for c in candidates:
        if c and (Path(c) / "build/px4_sitl_default/bin/px4").exists():
            return Path(c) / "build/px4_sitl_default"
    raise FileNotFoundError(
        "PX4 SITL not found. Build it (see README, section PX4) and set PX4_DIR to the "
        "PX4-Autopilot folder.")


class Px4Process:
    """Start PX4 SITL (airframe none_iris) in a fresh working directory, without its own simulator."""

    def __init__(self, instance=0, px4_dir=None, log_dir=None):
        self.build = find_px4(px4_dir)
        self.instance = instance
        self.workdir = Path(tempfile.mkdtemp(prefix=f"px4_{instance}_"))
        self.log_dir = log_dir
        env = dict(os.environ, PX4_SIM_MODEL="none_iris")
        self.output = open(self.workdir / "px4.log", "w")
        self.proc = subprocess.Popen(
            [str(self.build / "bin/px4"), "-d", "-i", str(instance), str(self.build / "etc")],
            cwd=self.workdir, env=env, stdout=self.output, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL)

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.output.close()
        if self.log_dir is not None:          # keep PX4's own .ulg log of the flight
            logs = sorted(self.workdir.glob("log/**/*.ulg"))
            if logs:
                Path(self.log_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy(logs[-1], Path(self.log_dir) / f"px4_{self.instance}.ulg")
        shutil.rmtree(self.workdir, ignore_errors=True)
