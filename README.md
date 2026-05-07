# OpenArm Controller UI (LeRobot direct)

A PyQt5 desktop UI that talks to the bimanual OpenArm **directly** through
`BiOpenArmFollower` — no HTTP layer, no control server in between. The goal is
to learn the official LeRobot motor API while having a live control surface
for bring-up, calibration, and exploration.

## Status

**M3 — sliders command the arm (ramped control loop, in testing).**

Each poll tick (5 Hz):

1. Read observation; update every joint's `current`.
2. For each joint on a torque-ON arm, step its `commanded` toward its
   `target` (= slider position) by at most `max_speed / 5` degrees.
3. Send one `send_action(dict)` with the new commanded values for every
   torque-ON joint.
4. For torque-OFF arms, keep target and commanded synced with current so
   enabling torque later won't lurch.

**Three values per joint** — slider (target, user), commanded (what we
send each tick), current (motor reads back). Each slider paints an amber
tick mark at the current position so you can watch it chase the thumb.

- **Max commanded speed** is a global setting on the System tab (default
  5 °/s; range 0.1–120 °/s). Changes take effect on the next poll tick.
- **Emergency stop** now also resets every target to its current position,
  so toggling torque back on doesn't resume an interrupted motion.
- **Enabling torque** on an arm first aligns target & commanded with the
  last observed current, so the arm doesn't jump when torque engages.

Upcoming: keyboard shortcuts + reloadable bindings (M4), motor info
display + editable kp/kd (M5 / v3).

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
