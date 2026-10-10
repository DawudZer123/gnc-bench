import numpy as np

from gncbench.controllers import HoverController
from gncbench.dynamics import MOTORS, POS, initial_state, state_derivative
from gncbench.params import QuadParams
from gncbench.sensors import GPS, IMU, Barometer
from gncbench.sim import simulate

P = QuadParams()


def hover_state():
    x = initial_state(position=(0, 0, 1))
    x[MOTORS] = P.mass * P.gravity / 4
    return x


def test_imu_reads_gravity_at_rest():
    imu = IMU(np.random.default_rng(0), accel_noise=0, gyro_noise=0, accel_bias=0, gyro_bias=0)
    x = hover_state()
    m = imu.measure(0.0, x, state_derivative(x, x[MOTORS], P), P)
    assert np.allclose(m[:3], [0, 0, P.gravity])   # +g vers le haut
    assert np.allclose(m[3:], 0)


def test_gps_noise_has_expected_spread():
    gps = GPS(np.random.default_rng(1))
    x = hover_state()
    samples = np.array([gps.measure(0.0, x, None, P) for _ in range(5000)])
    assert np.allclose(samples[:, :3].std(axis=0), gps.pos_noise, rtol=0.05)
    assert np.allclose(samples[:, :3].mean(axis=0), x[POS], atol=0.05)


def test_sensor_rates_in_simulation():
    rng = np.random.default_rng(2)
    res = simulate(hover_state(), HoverController(P, dt=0.004), {"position": np.array([0, 0, 1.0])},
                   P, duration=2.0, sensors=[IMU(rng), GPS(rng)])
    imu_t, _ = res.measurements["imu"]
    gps_t, gps = res.measurements["gps"]
    assert len(imu_t) == 2.0 * 250 + 1
    assert len(gps_t) == 2.0 * 10 + 1
    assert gps.shape == (21, 6)

def test_baro_noise_has_expected_spread():
    baro = Barometer(np.random.default_rng(3))
    x = hover_state()                    # drone à 1 m de haut
    samples = np.array([baro.measure(0.0, x, None, P)[0] for _ in range(5000)])
    assert np.isclose(samples.std(), baro.alt_noise, rtol=0.05)     
    assert np.isclose(samples.mean(), x[POS][2], atol=0.01)   
