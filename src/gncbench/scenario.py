"""Test scenarios: everything that describes one flight, apart from the controller.

A Scenario fixes the wind, the GPS outage, the model error and the random seed.
`Scenario.random(seed)` draws a new one: this is what a Monte Carlo campaign repeats.
"""

from dataclasses import dataclass

import numpy as np

from .controllers import HoverController, WindRejectionController
from .dynamics import MOTORS, POS, initial_state
from .ekf import NavigationEKF
from .estimator import NavigationFilter
from .faults import SensorFault
from .metrics import tracking_report
from .params import QuadParams
from .sensors import GPS, IMU, Barometer, Magnetometer, OpticalFlow
from .sim import simulate
from .trajectories import figure_eight
from .wind import Wind

SENSOR_TYPES = {"imu": IMU, "gps": GPS, "baro": Barometer, "flow": OpticalFlow,
                "mag": Magnetometer}
NAVIGATION = {"kf": NavigationFilter,   # attitude borrowed from the simulation (weeks 3-4)
              "ekf": NavigationEKF}     # attitude estimated too, like a real autopilot


def default_controller(params, dt):
    """The controller used when none is given: our cascaded PID + geometric attitude."""
    return HoverController(params, dt, ki=1.0)


def wind_rejection_controller(params, dt):
    """The same controller + wind compensation from the navigation filter."""
    return WindRejectionController(params, dt, ki=1.0)


@dataclass
class Scenario:
    seed: int = 1                # sensor noise, IMU bias and gusts all come from this seed
    duration: float = 40.0       # s
    wind_speed: float = 3.0      # m/s, mean wind
    wind_heading: float = 0.0    # rad, direction the wind blows towards (0 = +x)
    gust_std: float = 1.5        # m/s
    outage_start: float = 10.0   # s, GPS lost at this time...
    outage_length: float = 20.0  # s, ...for this long
    mass_error: float = 0.0      # true mass = nominal mass × (1 + mass_error)

    @classmethod
    def random(cls, seed, duration=30.0):
        """Draws a random scenario. Same seed -> same scenario, so any flight can be replayed.

        The outage scales with the flight: for 30 s, it starts between 6 and 10 s
        and lasts 4.5 to 15 s.
        """
        rng = np.random.default_rng(seed)
        return cls(
            seed=seed,
            duration=duration,
            wind_speed=rng.uniform(0.0, 5.0),
            wind_heading=rng.uniform(0.0, 2 * np.pi),
            gust_std=rng.uniform(0.5, 2.0),
            outage_start=rng.uniform(0.2, 1 / 3) * duration,
            outage_length=rng.uniform(0.15, 0.5) * duration,
            mass_error=rng.uniform(-0.1, 0.1),
        )

    @property
    def outage(self):
        return (self.outage_start, self.outage_start + self.outage_length)

    def fly(self, controller=default_controller, sensors=("imu", "gps", "baro", "flow"),
            corridor=0.5, trajectory=figure_eight, navigation="kf"):
        """Flies this scenario and returns (SimResult, report).

        controller: any function (params, dt) -> Controller. A Controller class works too.
                    This is the plug-in point of the bench.
        sensors:    names of the sensors on board (see SENSOR_TYPES).
        navigation: "kf" or "ekf" (see NAVIGATION). The EKF needs a compass for the yaw,
                    so "mag" is added to the sensors if it is missing.
        The report only judges the GPS outage window.
        """
        nominal = QuadParams()                                   # what the controller believes
        true = QuadParams(mass=nominal.mass * (1 + self.mass_error))  # what actually flies
        rng = np.random.default_rng(self.seed)
        if navigation == "ekf" and "mag" not in sensors:
            sensors = (*sensors, "mag")
        x0 = initial_state(position=trajectory(0)["position"])
        x0[MOTORS] = true.mass * true.gravity / 4
        mean_wind = self.wind_speed * np.array([np.cos(self.wind_heading),
                                                np.sin(self.wind_heading), 0.0])
        res = simulate(
            x0, controller(nominal, 0.004), trajectory, true, duration=self.duration,
            sensors=[SENSOR_TYPES[name](rng) for name in sensors],
            faults=[SensorFault("gps", start=self.outage[0], end=self.outage[1])],
            estimator=NAVIGATION[navigation](nominal, p0=x0[POS]), use_estimate=True,
            wind=Wind(np.random.default_rng(self.seed + 10), mean=mean_wind, gust_std=self.gust_std),
        )
        return res, tracking_report(res, trajectory, corridor, window=self.outage, params=true)
