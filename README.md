# OpenArm Controller UI

A PyQt5 desktop control surface for the **bimanual OpenArm**. Drives the
arms directly through LeRobot's `BiOpenArmFollower` — no separate
control server, no HTTP layer in between. Useful for bring-up,
calibration, joint-by-joint exploration, cartesian jogging, and VR
teleoperation.

## Features

- **Joint control** — keyboard and slider jogging, per-joint torque
  toggles, go-to-zero, e-stop. Reloadable key bindings.
- **Cartesian control** — task-space jogging in the TCP frame, driven
  by an iterative damped-least-squares IK solver (Pinocchio) with a
  rotation-priority workspace-edge fallback so the arm extends toward
  unreachable targets instead of freezing.
- **VR teleoperation** — Pico 4 / XRoboToolkit controllers stream
  pose at 90 Hz over the PXREASDK broker; per-arm grip-clutch,
  pose smoothing, hysteretic dead-band, and configurable position /
  rotation scales.
- **Gravity compensation** — torque feed-forward computed from the
  arm's URDF so the arm holds its pose with motors-on but no command.
- **Camera feeds** — base + per-wrist cameras with diagnostic
  snapshot dump (helps confirm RGB vs BGR delivery).
- **Per-tick performance log** — CSV with stage-level timings (state
  read, IK, action build, send) plus periodic median/p95/max summaries
  on the standard logger. Used to monitor 30 Hz control loop health.

## Architecture

The UI runs Qt's main loop at 5 Hz (camera-only polling). All motor
control is driven by a separate `MotionWorker` QThread at 30 Hz that
owns:

- a `JointTrajectory` per joint (linear time-based ramp at the
  configured `max_speed_deg_per_sec`),
- a **lead cap** that pauses trajectory advance when the setpoint
  would run more than 10° ahead of the motor's current position,
- the IK solver, gravity-comp, and VR-pose pipeline.

UI events post commands into the worker's queue; the worker emits
`state_updated` after each tick and the UI repaints. This keeps motor
timing independent of UI thread jitter and means re-enabling torque is
lurch-free (torque-OFF arms have their trajectories continuously reset
to `start = target = current`).

## Tabs

- **Controller (movej)** — joint-space jogging, torque toggles,
  per-joint speed cap, calibration shortcuts.
- **Cartesian (movel)** — task-space jogging in TCP frame.
- **VR Info** — live readout of the VR receiver: packet rate,
  controller poses, button states. Diagnostic only.
- **VR Control** — enable/disable per-arm VR control, position and
  rotation scales, absolute-mode toggle.
- **System** — calibration, gravity-comp scale, max joint speed,
  camera snapshots, motor configuration display.

## Prerequisites

- CAN interfaces (`can0`, `can1`) brought up with the appropriate
  bitrate. The OpenArm CAN-configure tool from the parent project
  handles this; see your distribution's setup script.
- Both arms powered and physically in a safe rest pose. Connecting
  briefly engages torque before the UI disables it; keep hands clear
  during startup.
- Python 3.10+, `PyQt5`, `numpy`, `pinocchio`, `Pillow`,
  `lerobot >= 0.5.1`. (URDF for IK / gravity-comp ships in `urdf/`.)
- For VR teleop: a Pico 4 / Pico 4 Ultra running XRoboToolkit on the
  same network, broadcasting controller poses to this machine.
- No other process holding the CAN bus or cameras — the UI owns the
  robot for its lifetime.

## Running

From the parent directory containing this package:

```bash
python -m openarm_controller_ui_lerobot.app
```

The module form is required so relative imports resolve.

## Calibration

The UI uses its own LeRobot calibration profile
(`id="openarm_controller_ui"`). On first run, the arms report **NOT
calibrated**. Position each arm in the canonical zero pose (hanging
straight down, gripper closed) and click **Calibrate LEFT arm** /
**Calibrate RIGHT arm** in the System tab. The calibration is loaded
automatically on every subsequent launch.

## Configuration

- **`~/.openarm_ui_config/motion_settings.json`** — persisted runtime
  settings: `max_speed_deg_per_sec`, gravity-comp scale, VR position
  / rotation scales, VR receiver backend.
- **`key_bindings.json`** — joint-jog key map (reloadable from the UI).
- **`config.py`** — hard-coded defaults: motion tick rate, lead cap,
  IK tolerances, dead-band thresholds, VR axis remap.
- **`local_data/`** — diagnostic artefacts (perf logs, VR recordings,
  camera snapshots). Gitignored; safe to delete.

## Layout

```
openarm_controller_ui_lerobot/
    app.py                 # entry point, main window, robot lifecycle
    config.py              # hardware constants, paths, UI defaults
    motion_worker.py       # 30 Hz QThread driving all motor control
    robot_service.py       # thread-safe BiOpenArmFollower wrapper
    runtime_state.py       # mutable settings shared with the worker
    session_config.py      # persisted session settings (load / save)
    ik_solver.py           # damped-least-squares Pinocchio IK
    gravity_comp.py        # torque feed-forward from URDF
    motion_perf_log.py     # per-tick CSV performance recorder
    vr_input.py            # legacy UDP VR receiver
    vr_input_pxreasdk.py   # XRoboToolkit (PXREASDK) VR receiver
    vr_absolute_tracker.py # alternative absolute-mode VR mapping
    key_bindings.py        # loader for key_bindings.json
    tab_controller.py      # Controller tab (joint-space jogging)
    tab_cartesian.py       # Cartesian tab (TCP-space jogging)
    tab_vr.py              # VR Info tab
    tab_vr_control.py      # VR Control tab
    tab_system.py          # System tab (calibration, settings, snapshots)
    urdf/                  # bimanual OpenArm description for IK / gravity comp
    tools/                 # offline replay + log analysis utilities
    docs/                  # design notes
```

## License

[MIT-0](LICENSE) — MIT No Attribution. Use this however you like.
