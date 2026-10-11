# GNC Bench: a GNC-in-the-loop test bench (MEC McGill)

A 6-degree-of-freedom quadcopter simulator into which you plug a controller to test it automatically. It includes noisy sensors, wind, sensor faults and a Kalman navigation filter. The headline demo is a figure-8 flight in the wind with a 20 s GPS outage.

---

## 1. Install the software (once)

Everything runs in **VS Code** with **Python**. No MATLAB needed.

| Software | Where to get it | Note |
| --- | --- | --- |
| VS Code | already installed | |
| Python 3.12 | [python.org/downloads](https://www.python.org/downloads/) | Windows: tick **"Add python.exe to PATH"** during installation |
| Git | [git-scm.com](https://git-scm.com/downloads) | Mac: already there, or `xcode-select --install` |
| GitHub account | [github.com](https://github.com) | to back up the code and show it to the judges |

Check in a terminal:

```bash
python --version     # Mac/Linux: python3 --version
git --version
```

## 2. Open the project in VS Code

1. Clone or unzip the project into a simple folder, for example `Documents/gnc-bench`.
2. In VS Code: **File > Open Folder…** and pick `gnc-bench`.
3. VS Code offers to install the recommended extensions (Python, Pylance, Ruff, Jupyter). Click **Install All**.

## 3. Create the Python environment

A virtual environment (`.venv`) isolates this project's libraries. Open the integrated terminal (**Terminal > New Terminal**), then:

**Mac / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If PowerShell refuses to activate: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

Then, in VS Code: `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`) > **Python: Select Interpreter** > pick the one inside `.venv`. The `(.venv)` prefix should appear in the terminal.

## 4. Check that everything works

```bash
python -m pytest
```

Expected result: `50 passed`. You can also open the **Testing** tab (flask icon) in VS Code and click ▶.

Then run the demos (press **F5** for the "Hover demo" configuration, or use the terminal):

| Script | What it shows |
| --- | --- |
| `scripts/hover_demo.py` | Take-off from a tilted attitude to (1, -0.5, 2) m; final error about 0.6 cm |
| `scripts/sensors_demo.py` | IMU, GPS and barometer readings compared with the truth |
| `scripts/wind_demo.py` | Holding position in a 5 m/s wind, with and without integral action |
| `scripts/faults_demo.py` | 4 s GPS outage: dead reckoning drifts about 2.5 m |
| `scripts/navigation_demo.py` | Kalman filter vs dead reckoning during the same outage |
| `scripts/final_demo.py` | Figure-8 in the wind, GPS lost for 20 s: IMU + baro FAIL, + optical flow PASS |
| `scripts/campaign_demo.py` | Monte Carlo campaign: 100 random flights per version, writes `outputs/campaign_report.html` |
| `scripts/controller_comparison.py` | Plug-in demo: baseline PID vs wind-rejection controller on the same 100 flights |
| `scripts/ekf_demo.py` | Full EKF: the drone flies on its estimated attitude (error < 0.5°) |
| `scripts/ekf_campaign.py` | KF (true attitude) vs full EKF over 100 random flights |
| `scripts/fault_detection_demo.py` | GPS frozen, jumping or drifting: EKF with and without fault detection |
| `scripts/validate_flight_log.py` | Checks the models against a real PX4 log (`.ulg`); without a file, self-check on our simulator |
| `scripts/flight3d_demo.py` | 3D view of a flight; `--animate` shows the drone flying, `--gif` saves it for the slides |

Plots are saved in `outputs/`.

## 5. What the code contains

```
gnc-bench/
├── src/gncbench/
│   ├── params.py        physical parameters (mass, inertia, arms, rotors) and mixer matrix
│   ├── math3d.py        quaternions, rotation matrices, Euler angles
│   ├── dynamics.py      6-DOF equations of motion + motor lag: dx/dt = f(x, u)
│   ├── integrators.py   Runge-Kutta 4 step
│   ├── controllers.py   Controller interface, cascaded PID controller, wind-rejection controller
│   ├── sensors.py       IMU, GPS, barometer, optical flow, magnetometer
│   ├── wind.py          mean wind + Gauss-Markov gusts
│   ├── faults.py        sensor faults (dropout, freeze, offset, drift)
│   ├── estimator.py     navigation Kalman filter, attitude known (position, velocity, accel bias)
│   ├── ekf.py           full error-state EKF (attitude, gyro bias) with fault detection
│   ├── metrics.py       tracking report (corridor, PASS/FAIL, control effort, saturation)
│   ├── trajectories.py  reference trajectories (figure-8)
│   ├── scenario.py      one flight's conditions; Scenario.random(seed); plug-in point for controllers
│   ├── campaign.py      Monte Carlo campaigns in parallel + summary
│   ├── flightlog.py     reads PX4 logs (NED/FRD -> ENU/FLU), sensor noise, EKF replay, drag fit
│   ├── report.py        self-contained HTML campaign report
│   ├── viz.py           3D flight view and drone animation
│   └── sim.py           simulation loop (physics 500 Hz, controller 250 Hz)
├── scripts/             demos + plots
├── tests/               50 validation tests
└── .vscode/             VS Code settings (interpreter, tests, F5)
```

**The model.** The state has 17 components: position and velocity in the world frame (z up), attitude as a quaternion, angular rates in the body frame, and the actual thrust of each rotor. The equations:

- ṗ = v
- m v̇ = R(q) [0, 0, T] − m g e₃ − c v + c w_wind
- q̇ = ½ q ⊗ [0, ω]
- J ω̇ = τ − ω × J ω
- Ṫᵢ = (Tᵢ,cmd − Tᵢ) / τ_motor

**The controller.** An outer PID loop (with velocity and acceleration feedforward) computes the desired acceleration. From it we derive the desired orientation of the drone, then a geometric attitude controller on SO(3) (Lee, Leok and McClamroch, 2010) computes the torques. A mixer matrix converts them into a thrust for each of the 4 rotors.

**Navigation.** A linear Kalman filter predicts with the IMU and corrects with the GPS, the barometer and optical flow. Failed sensors (NaN) are ignored. With `use_estimate=True`, the controller flies on the estimate instead of the truth, like a real drone.

**Full EKF.** `NavigationEKF` is an error-state EKF with 15 error states (position, velocity, small attitude rotation, accelerometer and gyro biases). It predicts with the IMU and corrects with GPS, barometer, optical flow and magnetometer. Roll and pitch are observable through gravity, yaw through the magnetometer. Choose it with `scenario.fly(navigation="ekf")`: the controller then flies on the estimated attitude and angular rate too.

**Fault detection.** Before using a measurement, the EKF rejects exact repeats (a frozen sensor) and runs a χ² test on the innovation: if the measurement is too far from the prediction given the filter's own uncertainty (99.9 % gate), it is dropped. On normal flights this rejects about 0.1 % of measurements, as the theory predicts. It catches a frozen GPS and sudden jumps; a slow drift (spoofing) stays below the gate, which is the honest limit of this test. Switch it off with `navigation="ekf_naive"`.

**Validation on real data.** `flightlog.py` reads a PX4 log and runs three checks: the noise of the real sensors at rest against our sensor models, our EKF replayed on the real sensor data against PX4's own EKF2, and the drag coefficient seen in flight against our dynamics model. The same checks run on our own simulator first (`from_simulation`), where the right answers are known. Reading `.ulg` files needs `pip install pyulog`.

**Wind rejection.** Thrust always points along the body z axis, so the accelerometer's body x and y readings can only come from aerodynamic forces. The filter turns them into a horizontal disturbance estimate, and `WindRejectionController` subtracts it from the commanded acceleration, cancelling gusts before a position error builds up.

**The bench.** To test your own controller, write a class that inherits from `Controller` and implements `update(t, state, reference)`, which returns the 4 rotor thrusts. Then run `run_campaign(100, controller=MyController)`: the controller is built as `MyController(params, dt)`. The campaign draws random wind, gusts, GPS outages and mass errors, and the report gives the pass rate, the worst flight and its seed so you can replay it.

**The tests** check the physics, not just that the code runs: free fall against the analytic solution, exact equilibrium at hover thrust, conservation of energy and angular momentum in free rotation, motor step response, sensor noise statistics, wind statistics and steady-state offset, fault injection, Kalman filter accuracy and uncertainty growth, and the corridor report.

## 6. Commit your work

Commit after every step that works:

```bash
git add .
git commit -m "Short description of the change"
git push
```

VS Code can do this in one click from the **Source Control** tab.
