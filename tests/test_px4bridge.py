"""PX4 bridge: Earth models, frames and the lockstep protocol, checked against a fake PX4.

These tests do not need PX4 itself; they need pymavlink (pip install pymavlink).
"""

import socket
import threading

import numpy as np
import pytest

from gncbench.dynamics import POS, initial_state
from gncbench.geo import Home, lla_to_local, local_to_lla, magnetic_field, pressure_at
from gncbench.px4bridge import FRD_TO_FLU, NED_TO_ENU, IrisParams, ground_contact
from gncbench.px4flight import mission_reference
from gncbench.trajectories import figure_eight

mavlink = pytest.importorskip("pymavlink.dialects.v20.common")


def test_coordinates_round_trip():
    home = Home()
    p = np.array([120.0, -45.0, 7.5])
    lat, lon, alt = local_to_lla(p, home)
    assert np.allclose(lla_to_local(lat, lon, alt, home), p, atol=1e-6)
    assert lat < home.lat and lon > home.lon          # 45 m south, 120 m east


def test_earth_models():
    # Montreal: field about 0.52 gauss, dipping about 68°, declination about -14° (WMM)
    field, decl, incl = magnetic_field(45.5, -73.6)
    assert 0.45 < np.linalg.norm(field) < 0.6
    assert -16 < np.degrees(decl) < -12 and 65 < np.degrees(incl) < 72
    assert field[2] < 0                                  # points down in the northern hemisphere
    assert pressure_at(0.0)[0] == pytest.approx(1013.25)
    assert pressure_at(100.0)[0] == pytest.approx(1001.3, abs=0.2)   # about 12 Pa per metre


def test_frame_conversions_are_their_own_inverse():
    assert np.allclose(NED_TO_ENU @ NED_TO_ENU, np.eye(3))
    assert np.allclose(FRD_TO_FLU @ FRD_TO_FLU, np.eye(3))
    assert np.allclose(NED_TO_ENU @ [1, 0, 0], [0, 1, 0])   # north is ENU y


def test_ground_contact():
    p = IrisParams()
    x = initial_state(position=(0, 0, -0.01))
    x[3:6] = [0.5, 0, -1.0]
    x, on_ground = ground_contact(x, p)
    assert on_ground and x[POS][2] == 0 and np.all(x[3:6] == 0)
    x = initial_state(position=(0, 0, 0.0))
    x[13:17] = p.max_thrust                             # full thrust: lift-off allowed
    assert not ground_contact(x, p)[1]


def test_mission_reference_is_continuous():
    t = np.arange(0, 60, 0.01)
    pos = np.array([mission_reference(s, figure_eight, 30.0)[0] for s in t])
    assert np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)) < 0.05


def fake_px4(port, replies, thrust):
    """Connects like PX4, answers every HIL_SENSOR with HIL_ACTUATOR_CONTROLS."""
    sock = socket.create_connection(("127.0.0.1", port))
    link = mavlink.MAVLink(sock.makefile("wb", buffering=0), srcSystem=1, srcComponent=1)
    link.robust_parsing = True
    while len(replies) < 400:
        try:
            data = sock.recv(4096)
        except ConnectionResetError:          # the bridge closed the link: end of test
            break
        if not data:
            break
        for msg in link.parse_buffer(data) or []:
            replies.append(msg)
            if msg.get_type() == "HIL_SENSOR":
                link.hil_actuator_controls_send(msg.time_usec, [thrust] * 4 + [0.0] * 12,
                                                128, 1)
    sock.close()


@pytest.mark.parametrize("thrust", [0.0, 0.8])
def test_lockstep_with_fake_px4(thrust):
    from gncbench.px4bridge import Px4Bridge

    bridge = Px4Bridge(instance=37, seed=3)
    bridge.listen()
    received = []
    px4 = threading.Thread(target=fake_px4, args=(bridge.port, received, thrust))
    px4.start()
    bridge.accept(timeout=5)
    for _ in range(300):                                # 1.2 s
        bridge.step()
    bridge.close()
    px4.join(5)

    sensors = [m for m in received if m.get_type() == "HIL_SENSOR"]
    gps = [m for m in received if m.get_type() == "HIL_GPS"]
    assert sensors[0].time_usec == 4000                 # PX4's clock is our clock
    assert np.all(np.diff([m.time_usec for m in sensors]) == 4000)
    first = sensors[0]
    # On the ground, level: the accelerometer reads 1 g upward, i.e. -9.81 on the FRD z axis
    assert first.zacc == pytest.approx(-9.81, abs=0.2) and abs(first.xacc) < 0.2
    assert first.zmag > 0                               # field points down = +z in FRD
    assert len(gps) >= 10 and gps[0].fix_type == 3
    lat, lon, _ = local_to_lla(np.zeros(3), Home())
    assert gps[0].lat / 1e7 == pytest.approx(lat, abs=2e-5)
    if thrust == 0.0:
        assert bridge.x[POS][2] == 0.0                  # stays on the ground
    else:
        assert bridge.x[POS][2] > 0.5                   # 0.8 x 28 N > 14.7 N weight: climbs
