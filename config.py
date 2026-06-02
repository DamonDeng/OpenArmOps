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

# Standalone speed cap for the gripper motor only. Independent from the
# arm-joint cap above so the operator can have slow, deliberate arm
# motion while the gripper still snaps closed/open the moment the
# controller trigger is squeezed/released. Tuned to feel "gripper-like"
# rather than "another joint" — the gripper's mechanical range is small
# (~65°) so even 360°/s is bounded by the trigger going 0→1 in one tick.
INITIAL_MAX_SPEED_DEG_PER_SEC_GRIPPER = 360.0

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
# Lives in the user's home dir, NOT the repo, because these are
# per-user preferences — different operators / different machines
# should each have their own values, and a `git clean -dfx` should
# not nuke them.
SESSION_CONFIG_DIR = Path.home() / ".openarm_ui_config"
SESSION_CONFIG_PATH = SESSION_CONFIG_DIR / "motion_settings.json"

# ── Diagnostic / replay artefacts ─────────────────────────────────────
# VR recordings and perf logs are session artefacts that are only
# useful in the context of the code that produced them — config.py
# already references specific recording filenames as the source of
# truth for the VR axis remap. Living next to the code makes the
# back-references obvious; ``.gitignore`` keeps the actual data out
# of the repo.
LOCAL_DATA_DIR = PACKAGE_DIR / "local_data"

# VR recording artefacts. JSON Lines (one record per line) so the file
# can be tailed live, grepped, and replayed by a future tool with
# minimal parsing.
VR_RECORDINGS_DIR = LOCAL_DATA_DIR / "vr_recordings"

# Motion-worker performance logs. CSV per session, one row per tick
# (~80 bytes; ~9 kB/s at 30 Hz). Used to diagnose tick-rate issues
# like "do we really sustain 30 Hz with both arms enabled?". Also
# the source for the periodic summary line on the standard logger.
PERF_LOG_DIR = LOCAL_DATA_DIR / "perf_logs"

# Camera snapshot diagnostic dumps. The System tab's "Camera Snapshots"
# button writes one PNG per camera (plus a swapped-R/B variant) here
# so the user can confirm their camera is delivering RGB vs BGR.
SNAPSHOTS_DIR = LOCAL_DATA_DIR / "snapshots"
# How often to roll up the per-tick numbers into a single INFO-level
# summary line (median / p95 / max per stage). Set to 0 to disable.
PERF_LOG_SUMMARY_INTERVAL_SEC = 5.0

# ── VR input (Pico 4 Ultra APK over UDP) ──────────────────────────────
# The openarmx_teleop_vr_apk APK streams controller + head poses to
# this port as ASCII-space-delimited datagrams. The protocol is one
# message per UDP packet, first token is message type (LEFT / RIGHT /
# HEAD / MODE / CALIBRATE_DONE), remaining tokens are positional floats
# or flags. See vr_input.py for the parser.
VR_UDP_BIND_ADDR = "0.0.0.0"
VR_UDP_PORT = 5100

# VR receiver backend.
# - "pxreasdk" (default): attach to Pico's XRoboToolkit broker
#   (RoboticsServiceProcess) via libPXREARobotSDK.so. Sustains 90 Hz
#   in dual-arm mode. Requires the service to be running on this host
#   (127.0.0.1:60061) and the headset to run XRoboToolkit-Unity-Client.
# - "udp": legacy listener for the closed-source openarmx-vr-pico.apk.
#   Caps at ~5 Hz aggregate dual-arm; kept as a fallback. See
#   docs/vr_packet_rate_investigation.md for the diagnosis trail.
VR_RECEIVER_BACKEND = "pxreasdk"

# Path to libPXREARobotSDK.so. The standard install location on
# Ubuntu is /opt/apps/roboticsservice/SDK/arm64/. ctypes also tries
# the bare basename via dlopen first, so the loader will pick this up
# automatically when LD_LIBRARY_PATH or runService.sh has been set.
VR_PXREASDK_LIB = "/opt/apps/roboticsservice/SDK/arm64/libPXREARobotSDK.so"

# Sensor noise dead-bands — applied when the motion worker eventually
# consumes controller deltas. Small thresholds to stop the arm from
# chasing sensor jitter at rest.
#
# Hysteretic dead-band: the arm enters "moving" when delta exceeds the
# OUT threshold, and only stops moving once it falls below the IN
# threshold. Without hysteresis, rate-of-change of jitter past a single
# threshold causes the cart_target to flip between snapshot and live
# delta tick by tick, making the motor command oscillate. _IN < _OUT.
VR_DEAD_BAND_POS_M_OUT = 0.004   # 4 mm — must move this far to start tracking
VR_DEAD_BAND_POS_M_IN  = 0.002   # 2 mm — drop below to settle back to snapshot
VR_DEAD_BAND_ROT_RAD_OUT = 0.035 # ~2.0°
VR_DEAD_BAND_ROT_RAD_IN  = 0.020 # ~1.1°

# One-pole low-pass on the controller pose. Smooths sensor jitter
# *before* the snapshot delta is computed so both the resting baseline
# and the moving target inherit the same noise budget. Alpha is the
# weight of the new sample at each motion tick; alpha=1.0 disables
# filtering. At 30 Hz tick rate, alpha=0.4 gives ~50 ms time constant
# — long enough to suppress single-frame APK jitter, short enough that
# fast hand motion doesn't feel laggy.
VR_POSE_FILTER_ALPHA = 0.4

# Stream freshness: if no packet arrived in this long, we treat the
# stream as stale in the UI and (later) freeze any VR-driven arms.
VR_STALE_SEC = 1.0

# Grip threshold: above this, the controller is "enabled" (dead-man
# engaged). Below this, its data is ignored by any motor-side logic.
# Set to 0.8 — empirically the APK appears to internally re-zero its
# pose reference when grip drops past somewhere ~0.6, so by the time
# our gate trips at 0.8 the APK has already reset and the very first
# packet we treat as "engaged" carries pos≈0, quat≈identity. That
# makes the snapshot we take coincide with the APK's reset, which
# eliminates the few-mm-per-engagement offset we'd see at 0.5.
# Same threshold gates the gripper trigger update — when grip is
# below 0.8 the gripper holds its last commanded value rather than
# snapping to whatever the trigger reads on the synthetic packets
# the APK streams while grip is released.
VR_GRIP_ENABLE_THRESHOLD = 0.8

# Initial control-display gain. Multiplies the controller-to-arm
# delta after frame remap so 1 cm of hand motion can drive less (or
# more) cm of arm motion. Live-editable from the System tab; saved
# alongside max_speed via session_config.json.
# Default 1.0 (1:1). The earlier 0.7 default was an attempt to keep
# targets inside the OpenArm's smaller-than-human reach, but offline
# replay of vr_log_20260602_120137 showed the asymmetric scaling
# (0.7 translation + 1.0 rotation) made every right-arm target an
# unnatural pose that no arm-shaped chain would naturally produce —
# IK ran to its iter cap on 44% of solves and fell into position-
# priority on 44%, sacrificing wrist orientation to chase position.
# At 1.0/1.0 the median solver work drops from 40 iters to 7 and the
# wrist tracks cleanly. Operators with cramped workspaces can lower
# this from the System tab.
INITIAL_VR_POS_SCALE = 1.0  # translation gain; 1.0 = arm follows hand 1:1
INITIAL_VR_ROT_SCALE = 1.0  # rotation gain; 1.0 = arm wrist follows hand 1:1
# Spinbox bounds in the UI.
VR_SCALE_MIN = 0.05
VR_SCALE_MAX = 5.0
VR_SCALE_STEP = 0.05

# IK boundary-clamp fallback (ik_solver pass 3). When enabled, IK
# targets past the workspace are walked back to the boundary along
# the current→target line, holding orientation strict — the arm
# extends toward the operator's hand instead of freezing. Costs up
# to 5 extra DLS solves per IK call. Disabled is the pre-2026-06-02
# behavior (freeze on unreachable). Toggleable from the System tab
# so operators can A/B the feel on hardware.
INITIAL_IK_BOUNDARY_FALLBACK_ENABLED = True

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
# LEFT arm — derived from a 4-posture LEFT controller recording on
# 2026-05-28 (vr_log_20260528_140228). The phases were P1 (arm hanging,
# wrist roll), P2 (arm forward → vr_y≈+0.67, vr_z≈+0.66), P3 (arm
# pointing left → vr_x≈-0.62, vr_z≈+0.62), P4 (return to zero). The
# left controller uses the SAME APK convention as the right
# controller — there's no left/right asymmetry in the wire-format
# frame. Both arms share the world frame
# (+robot_x=forward, +robot_y=operator's left, +robot_z=up), so the
# remap matrix is identical to the right-arm one.
#   robot_x = +vr_y   (forward)
#   robot_y = -vr_x   (operator's right → robot's -Y;
#                       equivalently, operator's left → robot +Y)
#   robot_z = +vr_z   (up)

# RIGHT arm — derived from a 3-posture VR recording on 2026-05-27 and
# verified in offline replay (tools/replay_vr_log_offline.py).
#
# What the APK actually sends (empirical, NOT the OpenXR convention
# the vr_input.py docstring inherited from openarmx-teleop):
#   +vr_x = operator's right
#   +vr_y = forward (out from the body)
#   +vr_z = up
# The "+Y up, +Z forward" reading is what tripped up earlier remap
# attempts; the 2026-05-27 recording's P3 (arm right, no forward)
# pinned the convention down — vr_x went to +0.61, vr_y stayed near
# zero, and vr_z went to +0.69, which only fits "y is forward, z is
# up", not the documented OpenXR convention.
#
#   robot_x = +vr_y   (forward)
#   robot_y = -vr_x   (operator's right → robot's -Y)
#   robot_z = +vr_z   (up)
VR_TRANSLATION_REMAP_RIGHT = (
    ( 0.0,  1.0,  0.0),
    (-1.0,  0.0,  0.0),
    ( 0.0,  0.0,  1.0),
)
VR_TRANSLATION_REMAP_LEFT = VR_TRANSLATION_REMAP_RIGHT

# Rotation remap: must encode the same axis permutation as the
# translation remap above, otherwise translations and rotations end up
# in different frames. Hardware test 2026-05-28 (right arm only):
# with M_r=identity but M_t=swap(X,Y), pitching the controller forward
# made the EE tip swing left/right instead of forward/back. Setting
# M_r = M_t makes a controller rotation axis ω_vr map to a robot
# rotation axis M_t @ ω_vr, matching how we map translations.
#
# Both arms share the same VR→robot translation remap (see above), so
# the rotation remap is also the same for both arms.
VR_ROTATION_REMAP_RIGHT = VR_TRANSLATION_REMAP_RIGHT
VR_ROTATION_REMAP_LEFT = VR_TRANSLATION_REMAP_LEFT

# Per-keypress target nudge in degrees. Shift is used as a layer selector
# (shoulder vs elbow/rotation), not as a coarse-speed modifier; every
# nudge is the same size. Hold a key → OS key-repeat advances the target
# at ~30×this per second; the motion worker's max_speed_deg_per_sec cap
# is what ultimately gates motor velocity.
KEY_DELTA_DEFAULT = 1.0

# Key bindings config file — loaded at startup, reloadable via UI
DEFAULT_KEY_BINDINGS_PATH = PACKAGE_DIR / "key_bindings.json"
