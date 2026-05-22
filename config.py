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

# Trajectory staleness. If a joint's trajectory was last touched longer
# ago than this, a new set_target rebuilds from current motor position
# rather than extending the old trajectory. Prevents an ancient target
# from resuming when the user picks the arm back up after a pause.
TRAJECTORY_STALENESS_SEC = 0.5

# ── User-writable session config ──────────────────────────────────────
# Persisted runtime-tunable settings (max speed, gravity comp scale, …).
# Lives outside the package so rebuilds don't clobber it and so we don't
# pollute the repo with per-user values.
SESSION_CONFIG_DIR = Path.home() / ".openarm_ui_config"
SESSION_CONFIG_PATH = SESSION_CONFIG_DIR / "motion_settings.json"

# ── VR input (Pico 4 Ultra APK over UDP) ──────────────────────────────
# The openarmx_teleop_vr_apk APK streams controller + head poses to
# this port as ASCII-space-delimited datagrams. The protocol is one
# message per UDP packet, first token is message type (LEFT / RIGHT /
# HEAD / MODE / CALIBRATE_DONE), remaining tokens are positional floats
# or flags. See vr_input.py for the parser.
VR_UDP_BIND_ADDR = "0.0.0.0"
VR_UDP_PORT = 5100

# Sensor noise dead-bands — applied when the motion worker eventually
# consumes controller deltas. Small thresholds to stop the arm from
# chasing sensor jitter at rest.
VR_DEAD_BAND_POS_M = 0.002   # 2 mm
VR_DEAD_BAND_ROT_RAD = 0.02  # ~1.1°

# Stream freshness: if no packet arrived in this long, we treat the
# stream as stale in the UI and (later) freeze any VR-driven arms.
VR_STALE_SEC = 1.0

# Grip threshold: above this, the controller is "enabled" (dead-man
# engaged). Below this, its data is ignored by any motor-side logic.
# Only relevant in Phase 2b; Phase 2a just displays the raw value.
VR_GRIP_ENABLE_THRESHOLD = 0.5

# Translation axis remap between the Pico/Unity OpenXR world frame
# (+X right, +Y up, +Z forward, left-handed) and our robot world frame
# (+X forward, +Y left, +Z up). Applied per-arm as:
#
#     delta_robot = VR_TRANSLATION_REMAP_<ARM> @ delta_vr
#
# The openarmx_teleop_vr project's teleop_params.yaml uses the same
# matrix for both arms. We keep them separate because live hardware
# testing has already shown one axis needs a different sign on the
# right arm — the left/right direction was reversed on the right arm
# only (moving the right controller right caused the right arm to
# move left in robot frame).
#
# LEFT arm — openarmx default, not yet hardware-verified axis-by-axis:
#   robot_x = -vr_z
#   robot_y = -vr_x
#   robot_z = +vr_y
VR_TRANSLATION_REMAP_LEFT = (
    ( 0.0,  0.0, -1.0),
    (-1.0,  0.0,  0.0),
    ( 0.0,  1.0,  0.0),
)

# RIGHT arm — same as left except Y row sign flipped to fix the
# "push right controller right, right arm moves toward body" bug
# found on hardware. User confirmed: with this flip, pushing the
# right controller to the user's right should now move the right
# arm to the robot's right.
#   robot_x = -vr_z
#   robot_y = +vr_x   (flipped vs LEFT)
#   robot_z = +vr_y
VR_TRANSLATION_REMAP_RIGHT = (
    ( 0.0,  0.0, -1.0),
    ( 1.0,  0.0,  0.0),
    ( 0.0,  1.0,  0.0),
)

# Rotation remap: identity for both arms for now. The openarmx config
# uses identity orientation_matrix too, but their closed IK core
# likely does a handedness flip internally (Unity is left-handed,
# robotics is right-handed). If the wrist rotates in the wrong
# direction on hardware, replace these with 3x3 matrices that flip
# the relevant axis.
VR_ROTATION_REMAP_LEFT = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
VR_ROTATION_REMAP_RIGHT = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

# Per-keypress target nudge in degrees. Shift is used as a layer selector
# (shoulder vs elbow/rotation), not as a coarse-speed modifier; every
# nudge is the same size. Hold a key → OS key-repeat advances the target
# at ~30×this per second; the motion worker's max_speed_deg_per_sec cap
# is what ultimately gates motor velocity.
KEY_DELTA_DEFAULT = 1.0

# Key bindings config file — loaded at startup, reloadable via UI
DEFAULT_KEY_BINDINGS_PATH = PACKAGE_DIR / "key_bindings.json"
