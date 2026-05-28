"""Absolute-pose VR tracker — alternative to the snapshot/delta clutch model.

Operator model:
  - Per VR-enable cycle, the FIRST grip press locks in a single reference
    pair: ``(ctrl_origin, arm_origin)``. Every subsequent tick (while
    grip is engaged) computes ``arm_origin * remap(ctrl_origin^-1 * ctrl_now)``.
  - Releasing grip does NOT clear the reference. The arm freezes at its
    last commanded pose; re-pressing grip resumes tracking from the
    SAME reference (so if the hand moved while grip was off the arm
    will jump on resume — operator's responsibility to slow-go-to-zero
    and re-enable VR before starting a fresh test).
  - Switching VR off (or e-stop) clears the reference.

Vs. the relative ``_vr_tick`` path:
  - No EMA pose filter (raw controller pose feeds the math).
  - No hysteretic dead-band (every tick within grip updates the target).
  - Reference is session-fixed, not per-engagement.

Why a separate class: the user wanted to A/B test this without losing
the relative path. Keeping the implementations separate makes it easy
to compare smoothness side-by-side and to delete one if it doesn't pan
out, without touching the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pinocchio as pin


@dataclass
class _AbsoluteRef:
    controller: pin.SE3
    arm: pin.SE3


class VRAbsoluteTracker:
    """Per-arm session-fixed reference tracker.

    Stateless aside from the per-arm reference. Math (axis remap, body-
    frame re-expression, scale) mirrors the relative path so that the
    same Cartesian target downstream (IK + ramped joint sends) handles
    both modes uniformly.
    """

    def __init__(self) -> None:
        self._ref: dict[str, Optional[_AbsoluteRef]] = {
            "left": None, "right": None,
        }

    def reset(self, arm: str) -> None:
        self._ref[arm] = None

    def reset_all(self) -> None:
        self._ref["left"] = None
        self._ref["right"] = None

    def has_reference(self, arm: str) -> bool:
        return self._ref[arm] is not None

    def tick(
        self,
        arm: str,
        *,
        ctrl_pose: pin.SE3,
        arm_fk: pin.SE3,
        grip_engaged: bool,
        s_pos: float,
        s_rot: float,
        translation_remap: np.ndarray,
        rotation_remap: np.ndarray,
    ) -> Optional[pin.SE3]:
        """Compute the new arm target pose.

        Returns ``None`` when no target update should be issued this tick
        (e.g. waiting for the first grip press, or grip released after
        the reference was already taken — the caller leaves cart_target
        at its last value, freezing the arm).
        """
        ref = self._ref[arm]

        if ref is None:
            if not grip_engaged:
                return None  # waiting for the operator's first engagement
            # First grip press of this VR-enable cycle: lock the reference
            # and hold the arm at its current FK (no motion this tick;
            # we have no delta yet).
            self._ref[arm] = _AbsoluteRef(controller=ctrl_pose, arm=arm_fk)
            return arm_fk

        if not grip_engaged:
            # Reference exists but grip is currently released → freeze.
            return None

        delta = ref.controller.actInv(ctrl_pose)

        # Per-axis scaling on SE3 components (same approach as the
        # relative path: NOT log/exp on the SE3 because that mixes
        # translation and rotation through the screw-axis term).
        if s_pos != 1.0:
            t_scaled = delta.translation * s_pos
        else:
            t_scaled = delta.translation
        if s_rot != 1.0:
            w = pin.log3(delta.rotation)
            R_scaled = pin.exp3(w * s_rot)
        else:
            R_scaled = delta.rotation
        delta = pin.SE3(R_scaled, t_scaled)

        # Axis remap (VR frame → robot frame). Same form as
        # MotionWorker._apply_vr_remap so both modes land in the same
        # robot-frame delta.
        t_new = translation_remap @ delta.translation
        R_new = rotation_remap @ delta.rotation @ rotation_remap.T
        delta = pin.SE3(R_new, t_new)

        # Re-express the world-frame delta in the arm origin's body frame
        # so that the SE3 left-action lands at the world-frame target:
        #   target.t = arm_origin.t + delta.t
        # See MotionWorker._vr_tick for the derivation.
        R_arm = ref.arm.rotation
        delta = pin.SE3(
            R_arm.T @ delta.rotation @ R_arm,
            R_arm.T @ delta.translation,
        )
        return ref.arm * delta
