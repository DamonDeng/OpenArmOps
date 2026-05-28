"""Shared mutable state between tabs.

Just a plain object with attributes — no Qt signals yet because the Controller
tab polls this on every tick, so it naturally picks up new values. If we add
more writers, promote to QObject + pyqtSignal.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class RuntimeState:
    max_speed_deg_per_sec: float = config.INITIAL_MAX_SPEED_DEG_PER_SEC
    # Standalone gripper speed cap. The gripper's job is to snap
    # closed/open as the operator squeezes/releases the trigger, not
    # to ramp like an arm joint, so it has its own setting that
    # bypasses the arm-joint speed.
    max_speed_deg_per_sec_gripper: float = config.INITIAL_MAX_SPEED_DEG_PER_SEC_GRIPPER
    gravity_comp_scale: float = config.INITIAL_GRAVITY_COMP_SCALE
    # VR control-display gain. Multiplies the cumulative-since-snapshot
    # SE3 delta after frame remap so the user can tune how much arm
    # motion comes out of a given hand motion. Both default to 1.0
    # (1:1 tracking). Shared across both arms.
    vr_pos_scale: float = config.INITIAL_VR_POS_SCALE
    vr_rot_scale: float = config.INITIAL_VR_ROT_SCALE

    def max_step_per_tick(self, poll_hz: int = config.POLL_HZ) -> float:
        """Max degrees the commanded value can move per poll tick."""
        return self.max_speed_deg_per_sec / float(poll_hz)
