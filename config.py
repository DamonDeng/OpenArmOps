"""Central constants for the OpenArm Controller UI.

Hardware and UI defaults live here. Runtime-editable settings (key bindings,
per-keystroke deltas) live in ``key_bindings.json`` and are loaded by
``key_bindings.py``.
"""

from pathlib import Path

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

# UI polling
POLL_HZ = 5  # rate for reading observations (state + cameras)
SEND_MAX_HZ = 5  # rate limit for send_action while dragging sliders

# Per-keystroke deltas (degrees). Shift = coarse, Ctrl = fine.
KEY_DELTA_DEFAULT = 1.0
KEY_DELTA_SHIFT = 3.0  # matches MAX_RELATIVE_TARGET_DEG to avoid silent clipping
KEY_DELTA_CTRL = 0.2

# Key bindings config file — loaded at startup, reloadable via UI
PACKAGE_DIR = Path(__file__).parent
DEFAULT_KEY_BINDINGS_PATH = PACKAGE_DIR / "key_bindings.json"
