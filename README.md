# OpenArm Controller UI (LeRobot direct)

A PyQt5 desktop UI that talks to the bimanual OpenArm **directly** through
`BiOpenArmFollower` — no HTTP layer, no control server in between. The goal is
to learn the official LeRobot motor API while having a live control surface
for bring-up, calibration, and exploration.

## Status

**M2 — Controller tab populated (read-only commanding).**
- Camera strip at the top: LEFT wrist / BASE / RIGHT wrist, updated at 5 Hz.
- 16 sliders across two columns (RIGHT arm, LEFT arm), each showing the joint's
  safe range, a target readout, and a live "current" readout from the motor.
- Per-arm **Torque ON/OFF** toggle above each slider column.
- **EMERGENCY STOP** button disables torque on both arms.
- On startup sliders show 0°; on the first poll tick they snap to the current
  motor position so the later command path won't jump when torque is enabled.

**M2 limitation — sliders are read-only.** Moving a slider updates the target
readout but does NOT yet call `send_action`. Command dispatch lands in M3.

Upcoming milestones: slider send_action (M3), keyboard shortcuts + reloadable
bindings (M4), motor info display + editable kp/kd (M5 / v3).

## Prerequisites

- CAN interfaces up (`can0`, `can1`) — see `start_arm.sh --stop` followed by `openarm-can-configure-socketcan` as done elsewhere in the repo.
- Arms powered and physically in a safe rest pose (torque will briefly engage during `connect()` before we disable it).
- `PyQt5` installed (already present on this machine).
- **No other process holding the CAN bus or cameras** — the UI owns the robot for its lifetime.

## Running

From the repo root (`/home/damon/workspace/openarm_space`):

```bash
python -m openarm_controller_ui_lerobot.app
```

Module form is required so relative imports resolve (`from . import config`).

## Layout

```
openarm_controller_ui_lerobot/
    __init__.py
    app.py              # entry point, main window, robot lifecycle
    config.py           # hardware constants, UI defaults
    robot_service.py    # thread-safe BiOpenArmFollower wrapper
    key_bindings.py     # loader for key_bindings.json
    key_bindings.json   # editable key→joint mapping (reloadable in UI later)
    tab_controller.py   # Controller tab (stub in M1)
    tab_system.py       # System tab (stub in M1)
    README.md
```

## Key bindings (editable)

Live in `key_bindings.json`. Each row maps a single character to
`(arm, joint, direction)`. Direction is `+1` or `-1` and gets multiplied by the
active delta at keypress time (default 1°, Shift 3°, Ctrl 0.2°).

M1 ships 8 bindings — the first two joints of each arm. You can edit the JSON
now; the UI will reload on demand starting in M4.

## First-run note

The UI connects with `id="openarm_controller_ui"`, which gives it its own
calibration files separate from the other tools in this repo. On first run,
the arms will be reported as **NOT calibrated** (the log line at startup says
so). Open the System tab, position each arm in the "hanging straight down,
gripper closed" pose, and click **Calibrate LEFT arm** / **Calibrate RIGHT
arm** one at a time. After that, the UI will load the saved calibration on
every subsequent launch with no prompt.

## Known limitations

- Controller tab is a placeholder (M2 will populate it).
- No motor info display yet (M5).
- Connecting briefly enables torque (inside `BiOpenArmFollower.connect()`) before the UI disables it. Keep hands clear during startup.
- Startup freezes briefly during `connect()` — no worker thread for connection yet; only calibration is off the UI thread.
- If connection fails the app exits with code 2 after showing an error dialog.
