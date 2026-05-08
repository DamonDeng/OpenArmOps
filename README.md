# OpenArm Controller UI (LeRobot direct)

A PyQt5 desktop UI that talks to the bimanual OpenArm **directly** through
`BiOpenArmFollower` — no HTTP layer, no control server in between. The goal is
to learn the official LeRobot motor API while having a live control surface
for bring-up, calibration, and exploration.

## Status

**M3-v2 — two-thread motion control (in testing).**

After M3's first version, we hit two problems: the closed-loop ramp stalled
on loaded joints (gripper under load was commanded from current, so error
stayed tiny and torque didn't grow), and the open-loop version tripped
LeRobot's `max_relative_target` safety cap.

The new architecture cleanly separates time from motor tracking:

- **MotionWorker** — dedicated QThread at 30 Hz. Owns a `JointTrajectory`
  per joint. Each tick: drain command queue, read motor state, advance
  every trajectory one tick (`setpoint = start + elapsed * deg_per_tick`),
  send one MIT batch per arm.
- **UI thread** — 5 Hz camera-only polling. Slider / keyboard / button
  events post commands into the worker's queue. Worker emits
  `state_updated` every tick; the tab updates "cur:" labels and amber
  markers in that slot.

Key mechanics:

- Setpoint grows linearly in wall-clock time at `max_speed_deg_per_sec`
  regardless of whether the motor is keeping up. If the motor lags, the
  MIT error (`setpoint - current`) grows → torque grows → motor moves.
  Naturally handles loaded joints without stalling.
- **Lead cap** (`LEAD_CAP_DEG=10°`): if setpoint would be more than 10°
  ahead of current, we pause the trajectory's time for that tick and
  clamp the setpoint. Prevents runaway on a jammed joint.
- Torque-OFF arms: their trajectories are continuously reset to
  `start=target=current`, so re-enabling torque is lurch-free.
- E-stop / torque-off / go-to-zero all post commands to the worker
  rather than touching state directly.

Upcoming: keyboard shortcuts + reloadable bindings (M4), motor info
display + editable kp/kd (M5 / v3).

**Known TODO carried forward**

- BiOpenArmFollower's `connect()` calls `set_zero_position()` every time,
  so the motor zeros reset to whatever pose the arm is in at startup.
  We should suppress that call to preserve the last calibration across
  app restarts. Deferred.

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
