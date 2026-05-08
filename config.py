"""Central constants for the OpenArm Controller UI.

Hardware and UI defaults live here. Runtime-editable settings (key bindings,
per-keystroke deltas) live in ``key_bindings.json`` and are loaded by
``key_bindings.py``.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

# CAN interfaces — match lerobot_controller/server.py and run_folding_direct.py
CAN_RIGHT = "can0"
CAN_LEFT = "can1"

# Camera device indices (verified in ./hardware_info)
CAMERAS = {
    "right_wrist": {"index": 0, "w": 1280, "h": 720, "fps": 30},
    "left_wrist": {"index": 11, "w": 1280, "h": 720, "fps": 30},
    "base": {"index": 1, "w": 640, "h": 480, "fps": 30},
}

# Camera strip order on the Controller tab (left → right as displayed)
CAMERA_STRIP_ORDER = ["left_wrist", "base", "right_wrist"]

# Joint ordering in right→left (matches BiOpenArmFollower.observation_features)
JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper",
]
ARM_SIDES = ["right", "left"]

# Safety / control knobs
MAX_RELATIVE_TARGET_DEG = 3.0  # per send_action, hard cap enforced by LeRobot
USE_FULL_LIMITS = True  # True ⇒ per-side ranges (75°/135°/etc), False ⇒ ±5° safety defaults

# UI polling (cameras + state display). The motor control loop runs at a
# higher rate in a dedicated worker thread — see MOTION_HZ below.
POLL_HZ = 5  # rate for reading observations on the UI thread (cameras + state)

# Motion control loop rate. Runs in a worker thread with a self-scheduled
# sleep based on time.perf_counter() so it's ~accurate independent of Qt.
MOTION_HZ = 30

# Initial cap on commanded joint velocity. The motion worker advances each
# joint's setpoint at this rate, regardless of whether the motor is keeping
# up (the lead cap below handles lag). Runtime-editable from System tab.
INITIAL_MAX_SPEED_DEG_PER_SEC = 20.0

# Fixed slow speed used by the "Slow go to zero" buttons. Intentionally
# independent of the System-tab setting — the whole point of those buttons
# is to get a predictable, gentle motion regardless of current config.
SLOW_SPEED_DEG_PER_SEC = 5.0

# Per-arm "unfold" target pose: shoulder (joint_2) fully outward, every other
# joint at 0°. Used by the "Unfold arm" buttons as a safe intermediate pose
# when going directly to zero could collide with the workspace surface.
# Sign of joint_2 is arm-specific because left and right arms' outward
# directions are mirror images of each other (per each arm's joint_limits:
# left joint_2 range = [-90, 9], right = [-9, 90]).
UNFOLD_ARM_POSE = {
    "left":  {"joint_1": 0.0, "joint_2": -90.0, "joint_3": 0.0, "joint_4": 0.0,
              "joint_5": 0.0, "joint_6":   0.0, "joint_7": 0.0, "gripper": 0.0},
    "right": {"joint_1": 0.0, "joint_2":  90.0, "joint_3": 0.0, "joint_4": 0.0,
              "joint_5": 0.0, "joint_6":   0.0, "joint_7": 0.0, "gripper": 0.0},
}

# ── Gravity compensation ──────────────────────────────────────────────
# Pinocchio-based feedforward torques. The URDF and a local copy of
# GravityCompensator were lifted from lerobot_controller/ — kept self-
# contained so this package has no runtime dependency on that server.
GRAVITY_URDF_PATH = PACKAGE_DIR / "urdf" / "openarm_bimanual.urdf"

# Initial scale factor for gravity compensation torques. 0.0 = off,
# 1.0 = model-predicted torque, >1.0 = overcompensate (useful when the
# URDF mass params underestimate your actual arm). Editable at runtime
# via the System-tab spinbox.
INITIAL_GRAVITY_COMP_SCALE = 0.5

# Spinbox range / step for the System-tab control.
GRAVITY_COMP_SCALE_MIN = 0.0
GRAVITY_COMP_SCALE_MAX = 2.0
GRAVITY_COMP_SCALE_STEP = 0.05

# Lead cap (degrees). If setpoint would be more than LEAD_CAP degrees ahead
# of the motor's observed current, we pause the trajectory until the motor
# catches up. Prevents runaway when a joint stalls (gripper jammed, arm
# hitting something) and bounds the error seen by MIT control.
LEAD_CAP_DEG = 10.0

# Per-keypress target nudge in degrees. Shift is used as a layer selector
# (shoulder vs elbow/rotation), not as a coarse-speed modifier; every
# nudge is the same size. Hold a key → OS key-repeat advances the target
# at ~30×this per second; the motion worker's max_speed_deg_per_sec cap
# is what ultimately gates motor velocity.
KEY_DELTA_DEFAULT = 1.0

# Key bindings config file — loaded at startup, reloadable via UI
DEFAULT_KEY_BINDINGS_PATH = PACKAGE_DIR / "key_bindings.json"
