# GNC Bench: a GNC-in-the-loop test bench for quadcopters

A 6-degree-of-freedom quadcopter simulator into which you plug a controller to test it automatically. It includes noisy sensors, wind, sensor faults and a Kalman navigation filter. The headline demo is a figure-8 flight in the wind with a 20 s GPS outage.

It can also fly the **real PX4 autopilot** on its physics (software-in-the-loop, section 7), so the code that flies real drones goes through the same Monte Carlo campaigns and fault injection.

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

Expected result: `63 passed` (the 7 PX4 bridge tests need `pip install -e ".[px4]"`; without it they are skipped). You can also open the **Testing** tab (flask icon) in VS Code and click ▶.

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
| `scripts/px4_demo.py` | The real PX4 autopilot on our physics, GPS lost 20 s: GPS only vs + optical flow (section 7) |
| `scripts/px4_campaign.py` | 100 random PX4 flights per version, in parallel PX4 instances (section 7) |
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
│   ├── geo.py           latitude/longitude, World Magnetic Model, standard atmosphere
│   ├── wmm_tables.py    magnetic field tables (copied from PX4, BSD-3)
│   ├── px4bridge.py     PX4 SITL link: HIL sensors out, motor commands in (lockstep), offboard link
│   ├── px4flight.py     one PX4 flight: start PX4, take off, fly the mission, record
│   ├── px4campaign.py   Monte Carlo on PX4 in parallel instances
│   ├── twin.py          digital twin: identify thrust, battery, balance, drag, sensors from a .ulg log
│   ├── report.py        self-contained HTML campaign report
│   ├── viz.py           3D flight view and drone animation
│   └── sim.py           simulation loop (physics 500 Hz, controller 250 Hz)
├── scripts/             demos + plots
├── tests/               63 validation tests
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

## 7. Fly the real PX4 autopilot on our physics

Up to here the controller and the estimator under test were our own Python code. Real drones fly with **PX4**. PX4 can be compiled for a computer (SITL, software-in-the-loop): it then reads its sensors from a simulator and sends its motor commands back. `px4bridge.py` makes our simulator that simulator.

```
our physics (Python)  --HIL_SENSOR: IMU, magnetometer, barometer (250 Hz)-->  PX4 SITL (real C++ flight code)
                      --HIL_GPS (10 Hz), optical flow + rangefinder (50 Hz)-->
                      <--HIL_ACTUATOR_CONTROLS: 4 motor commands------------
our mission (Python)  --position setpoints over MAVLink (offboard, 20 Hz)-->
```

**Lockstep.** PX4 has no clock of its own in this mode: its time is the timestamp of our sensor messages, and it answers each one with the motor commands for that step. A flight is reproducible from its seed and runs as fast as the computer allows (about 3× real time per instance).

**What we simulate for PX4.** The Iris airframe PX4 is tuned for (1.5 kg, `IrisParams`); a GPS whose error wanders slowly (Gauss-Markov, like a real receiver) instead of white noise; the magnetometer from the World Magnetic Model at the flight location (`geo.py`, McGill campus by default); the barometer from the standard atmosphere; a rigid ground for take-off and landing; optionally a downward optical flow camera and rangefinder, in the exact sign convention of PX4's EKF2.

### Build PX4 (once, about 20 minutes)

On a Mac, with the project's virtual environment active (it provides CMake and PX4's Python tools, no Homebrew needed):

```bash
cd ~/Documents/gnc-bench && source .venv/bin/activate
pip install "cmake<4" ninja empy==3.3.4 pyros-genmsg kconfiglib jinja2 jsonschema pyyaml future packaging toml pyserial psutil
git clone --recursive --branch v1.15.4 https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot
cd ~/PX4-Autopilot
make px4_sitl_default
```

PX4 v1.15 needs CMake 3 (CMake 4 refuses its older build files). The compiler is Apple's, from the Command Line Tools (`xcode-select --install`). The build ends with `Linking ... dyn_hello.px4mod`. If it complains about too many open files, run `ulimit -S -n 2048` first.

Then, in the project's terminal (`.venv` active): `pip install -e ".[px4]"`. The scripts look for PX4 in `~/PX4-Autopilot`, or wherever `PX4_DIR` points.

### Run

```bash
python scripts/px4_demo.py           # one flight per version, GPS lost 20 s
python scripts/px4_campaign.py 100 4 # 100 random flights per version, 4 PX4 instances in parallel
```

If you stop a campaign with Ctrl+C, some PX4 instances may keep running in the background and block the next run: clean them with `pkill -x px4`.

### What PX4 does (100 random flights per version)

Figure-8 for 30 s, random wind (0–6 m/s) and gusts, the GPS lost for 4.5–15 s at a random moment. PASS = mission completed and never more than 1 m away from it.

| | pass | mission completed | median worst error |
| --- | --- | --- | --- |
| A: PX4, GPS only | 3 % | 3 % | 4.3 m |
| B: PX4 + optical flow | 98 % | 99 % | 0.72 m |

- **GPS only, PX4 gives up after about 5 s.** Its EKF2 dead-reckons on the IMU for 5.1–6.2 s, then declares the position lost and switches to Altitude mode: it descends and lands where it is, 3 m on average from where it gave up. The only flights that passed had outages under 5 s. This is PX4's designed behaviour, and the bench measures it.
- **With optical flow, PX4 flies through outages of 15 s.** One flight failed in strong wind (5.7 m/s, 15 s outage): it completed the mission but drifted 4.6 m.
- **Known limit: not perfectly reproducible.** The physics and PX4 run in lockstep, but the mission's setpoints reach PX4 over UDP, outside the lockstep. Repeating a flight gives worst errors within a few centimetres of each other, and rarely a different outcome: one flight (seed 39) triggered a failsafe once in the campaign, then passed 9 times out of 9 when repeated.

## 8. Digital twin of your own drone

A flight log contains much more than the trajectory: the motor commands, the battery voltage and current, the raw sensors and PX4's whole configuration. `twin.py` turns one log into a model of **that** drone, then real PX4 flies it with **your** tuning (the 105 control, filter and geometry parameters are copied from the log).

```bash
pip install -e ".[px4,logs]"
python scripts/build_twin.py my_flight.ulg --mass 1.8 --arm 0.24   # writes outputs/twin.json, twin.png
python scripts/twin_replay.py my_flight.ulg                        # writes outputs/twin_replay.png
```

`build_twin.py` prints what it found and what it could not find. `twin_replay.py` flies the mission recorded in the log twice on PX4 SITL, at the same time: once on the twin, once on the generic Iris, and compares both with the real flight.

**What one flight identifies.**

- Thrust per unit throttle and how it fades with battery voltage (thrust ∝ V^α).
- A battery model: open-circuit voltage, capacity slope, internal resistance, current versus throttle.
- The centre of gravity, from which motors work harder in hover.
- The yaw torque imbalance between clockwise and counter-clockwise propellers.
- Horizontal drag.
- Accelerometer and gyro vibration in flight, which is 50 to 100 times the noise at rest.
- Barometer drift.

The thrust gain only gives ratios. `--mass` and `--arm` set the scale, so give the real values if you know them.

**What it does not identify: inertia and motor lag.** A calm mission barely rotates the drone (rates of a few hundredths of rad/s), and the attitude controller is in the loop, so a fit of angular acceleration against commanded torque is meaningless. `fit_rotation` checks the excitation and refuses below 0.3 rad/s of rate variation; the twin then uses priors. To identify them, fly a log with excitation: PX4's autotune, or sharp roll/pitch/yaw stick doublets.

**Results on a real S500 log (Pixhawk 6C, 224 s flight).**

- Hover throttle is 0.72, so thrust/weight at full throttle is 1.39.
- Thrust ∝ V^1.19: after 1272 mAh, the same throttle lifts 9 % less.
- Motor commands are [0.74 0.66 0.80 0.68]: the CG is 12 mm forward and 2 mm left, and the CCW props make 5.6 % more yaw torque.
- Drag k = 0.14 1/s.
- Vibration is about 1 m/s² on the accelerometer and 0.02–0.05 rad/s on the gyro.
- The barometer drifts 3.7 m over the flight.

| Replay of the real mission on PX4 | throttle error (RMS) | motor balance error (RMS) |
| --- | --- | --- |
| Generic Iris | 0.199 | 0.198 |
| Twin | 0.013 | 0.004 |

The twin reproduces the throttle rising from 0.70 to 0.75 as the battery drains, and the uneven motors.

**Honest limit.** This comparison is in-sample: the twin was identified from the same flight it replays. The real test is a second log of the same drone, which the twin has never seen.

