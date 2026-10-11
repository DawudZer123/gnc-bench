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

Expected result: `34 passed`. You can also open the **Testing** tab (flask icon) in VS Code and click ▶.

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
│   ├── controllers.py   Controller interface + cascaded hover/tracking controller
│   ├── sensors.py       IMU, GPS, barometer, optical flow
│   ├── wind.py          mean wind + Gauss-Markov gusts
│   ├── faults.py        sensor faults (dropout, freeze)
│   ├── estimator.py     navigation Kalman filter (position, velocity, accel bias)
│   ├── metrics.py       tracking report (corridor, PASS/FAIL, control effort, saturation)
│   ├── trajectories.py  reference trajectories (figure-8)
│   ├── scenario.py      one flight's conditions; Scenario.random(seed); plug-in point for controllers
│   ├── campaign.py      Monte Carlo campaigns in parallel + summary
│   ├── report.py        self-contained HTML campaign report
│   ├── viz.py           3D flight view and drone animation
│   └── sim.py           simulation loop (physics 500 Hz, controller 250 Hz)
├── scripts/             demos + plots
├── tests/               34 validation tests
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
