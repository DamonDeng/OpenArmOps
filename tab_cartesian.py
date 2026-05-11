"""Cartesian (movel) tab.

Per-arm panel with six spinboxes (x/y/z in mm, roll/pitch/yaw in degrees)
plus a "Mode: Joint / Cartesian" toggle. Switching an arm to Cartesian:

  1. Reads the arm's current joint positions, runs FK to get the TCP pose.
  2. Seeds the spinboxes with that pose (so immediate target = current → no
     motion on activation).
  3. Posts ``set_mode('cartesian')`` + ``set_cart_target(pose)`` to the worker.
  4. From now on, changes to the spinboxes (and Cartesian key nudges routed
     from the Controller tab) rebuild a CartesianTarget and post it.

Design notes:

- Gripper is not part of the 6-DOF cartesian pose; it stays on its joint
  slider in the Controller tab. Cartesian Alt-key bindings just call
  worker.post_set_target directly (same path as movej).
- World-frame translation, tool-frame rotation — matches the sign map
  validated in chat.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import config
from .ik_solver import pose_from_xyzrpy, pose_to_xyzrpy
from .key_bindings import Binding
from .motion_worker import CartesianTarget, MotionWorker
from .robot_service import RobotService

logger = logging.getLogger(__name__)


# Cartesian nudge step sizes. Same spirit as KEY_DELTA_DEFAULT for joint
# mode — small per-keystroke, OS auto-repeat does the rest. Spinbox step
# sizes match so user intuition stays consistent.
STEP_TRANSLATION_M = 0.005   # 5 mm per keystroke
STEP_ROTATION_RAD = math.radians(2.0)   # 2° per keystroke
STEP_GRIPPER_DEG = 1.0       # matches joint-mode gripper step


class _ArmCartPanel(QGroupBox):
    """One per-arm panel: six spinboxes + mode toggle. The panel holds
    the arm's local CartesianTarget and pushes it to the worker on any
    change (spinbox edit, apply-current-pose, or nudge-from-keyboard).
    """

    def __init__(self, arm: str, worker: MotionWorker, parent=None) -> None:
        super().__init__(f"{arm.upper()} arm", parent)
        self.arm = arm
        self.worker = worker
        self._suspend_signals = False
        # Local authoritative target. None means "we haven't been
        # activated yet — don't post to worker."
        self._target: Optional[CartesianTarget] = None

        root = QVBoxLayout(self)

        # Mode toggle + capture-current-pose
        top = QHBoxLayout()
        self.mode_btn = QPushButton("Mode: Joint")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setStyleSheet(
            "QPushButton { padding: 6px; }"
            "QPushButton:checked { background-color: #27a; color: white; }"
        )
        self.mode_btn.clicked.connect(self._on_mode_toggle)
        top.addWidget(self.mode_btn)

        self.btn_capture = QPushButton("Capture current pose")
        self.btn_capture.setToolTip(
            "Reset the target spinboxes to the arm's current TCP pose "
            "(FK of current joint angles). Useful before switching to "
            "Cartesian mode or when the target has drifted far from reality."
        )
        self.btn_capture.clicked.connect(self._on_capture_current)
        top.addWidget(self.btn_capture)
        top.addStretch(1)
        root.addLayout(top)

        # Status line
        self.status = QLabel("Activate Cartesian mode to begin.")
        self.status.setStyleSheet("color: #888;")
        root.addWidget(self.status)

        # Six spinboxes
        form = QFormLayout()
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)

        def mk_spin(lo, hi, step, decimals, suffix):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(decimals)
            sb.setSingleStep(step)
            sb.setSuffix(suffix)
            sb.setFont(mono)
            sb.setMinimumWidth(130)
            sb.setKeyboardTracking(False)  # fire valueChanged only on commit
            sb.valueChanged.connect(self._on_spin_changed)
            return sb

        # Translation in millimetres for user readability, converted to
        # metres when we build the target. Ranges are deliberately wide —
        # IK failure (unreachable) is the real limit.
        self.sb_x = mk_spin(-1000.0, 1000.0, 5.0, 1, " mm")
        self.sb_y = mk_spin(-1000.0, 1000.0, 5.0, 1, " mm")
        self.sb_z = mk_spin(-1000.0, 1000.0, 5.0, 1, " mm")
        # Rotation in degrees.
        self.sb_roll = mk_spin(-180.0, 180.0, 2.0, 1, " °")
        self.sb_pitch = mk_spin(-180.0, 180.0, 2.0, 1, " °")
        self.sb_yaw = mk_spin(-180.0, 180.0, 2.0, 1, " °")

        form.addRow("X", self.sb_x)
        form.addRow("Y", self.sb_y)
        form.addRow("Z", self.sb_z)
        form.addRow("Roll (tool)", self.sb_roll)
        form.addRow("Pitch (tool)", self.sb_pitch)
        form.addRow("Yaw (tool)", self.sb_yaw)
        root.addLayout(form)

        hint = QLabel(
            "Translation: world frame (X forward / Y left / Z up).\n"
            "Rotation: tool frame (roll about TCP forward, pitch up, yaw left)."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self._set_spinboxes_enabled(False)

    # ------------------------------------------------------------------
    # Public API (called from ControllerTab / CartesianTab)
    # ------------------------------------------------------------------
    def capture_current(self) -> None:
        """Query the worker for current TCP pose (via FK) and seed the
        spinboxes. Does not post to the worker — call apply_target()
        afterward if you want that.
        """
        pose = self.worker.compute_fk(self.arm)
        if pose is None:
            self.status.setText("Current pose unavailable (no observation yet).")
            return
        x, y, z, roll, pitch, yaw = pose_to_xyzrpy(pose)
        self._suspend_signals = True
        self.sb_x.setValue(x * 1000.0)
        self.sb_y.setValue(y * 1000.0)
        self.sb_z.setValue(z * 1000.0)
        self.sb_roll.setValue(math.degrees(roll))
        self.sb_pitch.setValue(math.degrees(pitch))
        self.sb_yaw.setValue(math.degrees(yaw))
        self._suspend_signals = False
        self._target = CartesianTarget(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
        )
        self.status.setText(
            f"Captured: x={x:+.3f} y={y:+.3f} z={z:+.3f} m, "
            f"rpy=({math.degrees(roll):+.1f},{math.degrees(pitch):+.1f},"
            f"{math.degrees(yaw):+.1f})°"
        )

    def apply_target(self) -> None:
        """Post the current spinbox values to the worker as a cartesian
        target. No-op if _target is None (never captured)."""
        self._rebuild_target_from_spinboxes()
        if self._target is not None:
            self.worker.post_set_cart_target(self.arm, self._target)

    def nudge_from_binding(self, binding: Binding) -> None:
        """Apply a cartesian-axis keyboard nudge. Called from the
        Controller tab's keyboard filter when a cartesian binding fires
        and this arm is in cartesian mode.
        """
        if not self.mode_btn.isChecked():
            return
        if self._target is None:
            # No baseline pose; capture now so the nudge applies to a
            # meaningful starting point.
            self.capture_current()
            if self._target is None:
                return

        axis = binding.target
        sign = binding.direction
        if axis == "x":
            self._target.x += sign * STEP_TRANSLATION_M
        elif axis == "y":
            self._target.y += sign * STEP_TRANSLATION_M
        elif axis == "z":
            self._target.z += sign * STEP_TRANSLATION_M
        elif axis == "roll":
            self._target.roll += sign * STEP_ROTATION_RAD
        elif axis == "pitch":
            self._target.pitch += sign * STEP_ROTATION_RAD
        elif axis == "yaw":
            self._target.yaw += sign * STEP_ROTATION_RAD
        elif axis == "gripper":
            # Gripper is 1-DOF; forward to the joint command path. Use
            # the arm's current commanded gripper (not the pose target)
            # and nudge by 1°. We reach into the Controller tab's slider
            # via a callback set by CartesianTab.
            cb = getattr(self, "gripper_nudge_callback", None)
            if cb is not None:
                cb(self.arm, sign)
            return
        else:
            return

        # Reflect into spinboxes (without triggering valueChanged loops).
        self._suspend_signals = True
        self.sb_x.setValue(self._target.x * 1000.0)
        self.sb_y.setValue(self._target.y * 1000.0)
        self.sb_z.setValue(self._target.z * 1000.0)
        self.sb_roll.setValue(math.degrees(self._target.roll))
        self.sb_pitch.setValue(math.degrees(self._target.pitch))
        self.sb_yaw.setValue(math.degrees(self._target.yaw))
        self._suspend_signals = False

        self.worker.post_set_cart_target(self.arm, self._target)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _on_mode_toggle(self, checked: bool) -> None:
        if checked:
            # Seed target with current pose before activating, so the arm
            # doesn't try to move to whatever stale value was in the spins.
            self.capture_current()
            if self._target is None:
                # FK unavailable — abort the mode change.
                self.mode_btn.setChecked(False)
                return
            self.worker.post_set_mode(self.arm, "cartesian")
            self.worker.post_set_cart_target(self.arm, self._target)
            self.mode_btn.setText("Mode: Cartesian")
            self._set_spinboxes_enabled(True)
            self.status.setText(
                "Cartesian mode active. Edit spinboxes or use cartesian keys."
            )
        else:
            self.worker.post_set_mode(self.arm, "joint")
            self.mode_btn.setText("Mode: Joint")
            self._set_spinboxes_enabled(False)
            self.status.setText("Joint mode. Cartesian target ignored.")

    def _on_capture_current(self) -> None:
        self.capture_current()
        # If already in cartesian mode, push the captured target so the
        # arm stops moving toward any stale target.
        if self.mode_btn.isChecked() and self._target is not None:
            self.worker.post_set_cart_target(self.arm, self._target)

    def _on_spin_changed(self, _value: float) -> None:
        if self._suspend_signals:
            return
        self._rebuild_target_from_spinboxes()
        if self.mode_btn.isChecked() and self._target is not None:
            self.worker.post_set_cart_target(self.arm, self._target)

    def _rebuild_target_from_spinboxes(self) -> None:
        self._target = CartesianTarget(
            x=self.sb_x.value() / 1000.0,
            y=self.sb_y.value() / 1000.0,
            z=self.sb_z.value() / 1000.0,
            roll=math.radians(self.sb_roll.value()),
            pitch=math.radians(self.sb_pitch.value()),
            yaw=math.radians(self.sb_yaw.value()),
        )

    def _set_spinboxes_enabled(self, enabled: bool) -> None:
        for sb in (self.sb_x, self.sb_y, self.sb_z,
                   self.sb_roll, self.sb_pitch, self.sb_yaw):
            sb.setEnabled(enabled)


class CartesianTab(QWidget):
    """Side-by-side LEFT / RIGHT cartesian panels."""

    def __init__(
        self,
        robot: RobotService,
        worker: MotionWorker,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.robot = robot
        self.worker = worker

        root = QVBoxLayout(self)
        header = QLabel(
            "Cartesian (movel) control. Each arm can be activated independently. "
            "When in Cartesian mode, the tick-level IK runs at 30 Hz — any "
            "unreachable target freezes the arm with a warning in the statusbar."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #888;")
        root.addWidget(header)

        cols = QHBoxLayout()
        self.left_panel = _ArmCartPanel("left", worker, self)
        self.right_panel = _ArmCartPanel("right", worker, self)
        cols.addWidget(self.left_panel)
        cols.addWidget(self.right_panel)
        root.addLayout(cols, stretch=1)
        root.addStretch(0)

    # ------------------------------------------------------------------
    # Keyboard-nudge entry point (wired from ControllerTab)
    # ------------------------------------------------------------------
    def handle_cartesian_nudge(self, binding: Binding) -> None:
        panel = self.left_panel if binding.arm == "left" else self.right_panel
        panel.nudge_from_binding(binding)

    def set_gripper_nudge_callback(self, cb) -> None:
        """Hand CartesianTab a way to nudge a gripper target without
        building a CartesianTarget (gripper is 1-DOF). Used when Alt+e/d
        fires with an arm in cartesian mode.
        """
        self.left_panel.gripper_nudge_callback = cb
        self.right_panel.gripper_nudge_callback = cb
