"""Earth models needed to talk to a real autopilot: coordinates, magnetic field, air pressure.

Our simulator works in a flat local frame (ENU, metres). PX4 expects GPS latitude/longitude,
a magnetometer that agrees with the World Magnetic Model at that place, and a barometric
pressure. These functions convert between the two.
"""

from dataclasses import dataclass

import numpy as np

from .wmm_tables import DECLINATION, INCLINATION, STRENGTH

EARTH_RADIUS = 6_378_137.0   # m (WGS-84 equatorial radius)


@dataclass(frozen=True)
class Home:
    """Origin of the local frame. Default: McGill campus, Montreal."""

    lat: float = 45.5048     # deg
    lon: float = -73.5772    # deg
    alt: float = 60.0        # m above mean sea level


def local_to_lla(p_enu, home: Home):
    """Local ENU position (m) -> latitude (deg), longitude (deg), altitude MSL (m).

    Flat-Earth approximation: accurate to millimetres over a few kilometres.
    """
    east, north, up = p_enu
    lat = home.lat + np.degrees(north / EARTH_RADIUS)
    lon = home.lon + np.degrees(east / (EARTH_RADIUS * np.cos(np.radians(home.lat))))
    return lat, lon, home.alt + up


def lla_to_local(lat, lon, alt, home: Home):
    """Inverse of local_to_lla."""
    north = np.radians(lat - home.lat) * EARTH_RADIUS
    east = np.radians(lon - home.lon) * EARTH_RADIUS * np.cos(np.radians(home.lat))
    return np.array([east, north, alt - home.alt])


def _table_lookup(table, lat, lon):
    """Bilinear interpolation on the 10° grid, exactly as PX4 does it."""
    lat = float(np.clip(lat, -90.0, 90.0))
    lon = (lon + 180.0) % 360.0 - 180.0
    i = min(int((np.floor(lat / 10.0) * 10.0 + 90.0) / 10.0), 17)
    j = min(int((np.floor(lon / 10.0) * 10.0 + 180.0) / 10.0), 35)
    lat0, lon0 = -90.0 + 10.0 * i, -180.0 + 10.0 * j
    a = np.clip((lat - lat0) / 10.0, 0.0, 1.0)
    b = np.clip((lon - lon0) / 10.0, 0.0, 1.0)
    south = (1 - b) * table[i][j] + b * table[i][j + 1]
    north = (1 - b) * table[i + 1][j] + b * table[i + 1][j + 1]
    return (1 - a) * south + a * north


def magnetic_field(lat, lon):
    """Earth's magnetic field at (lat, lon), in the ENU frame, in gauss.

    Returns (field_enu, declination_rad, inclination_rad).
    """
    declination = _table_lookup(DECLINATION, lat, lon) * 1e-4
    inclination = _table_lookup(INCLINATION, lat, lon) * 1e-4
    strength = _table_lookup(STRENGTH, lat, lon) * 1e-4
    horizontal = strength * np.cos(inclination)
    field = np.array([
        horizontal * np.sin(declination),    # east
        horizontal * np.cos(declination),    # north
        -strength * np.sin(inclination),     # up (points down in the northern hemisphere)
    ])
    return field, declination, inclination


def pressure_at(altitude_msl):
    """Standard atmosphere (ISA, troposphere): pressure (hPa) and temperature (°C)."""
    temperature_k = 288.15 - 0.0065 * altitude_msl
    pressure = 1013.25 * (temperature_k / 288.15) ** 5.25588
    return pressure, temperature_k - 273.15
