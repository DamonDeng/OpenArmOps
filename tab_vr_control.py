"""VR Control tab — Phase 2b-α.

Per-arm hard-enable for VR-driven control. This commit only toggles the
worker's ``_vr_enabled[arm]`` flag, which in turn puts the arm into
cartesian mode. No controller pose is actually read yet; the motor-side
tracking wires up in Phase 2b-β.

Design agreed with user:
- Per-arm Enable button. Hard on/off switch.
- Enabling puts the arm in cartesian mode automatically (Cartesian tab
  still works for spinbox control when VR is off).
- Disabling drops back to joint mode.
- Grip will be the dead-man (Phase 2b-β); release = arm freezes in
  place; re-engage = new snapshot baseline.
- Trigger will drive the gripper like an "invisible spring" (Phase 2b-β):
  pulled = closed, released = open.
- E-stop implicitly disables VR — the worker already does that, and the
  refresh timer here picks up the state change within 200 ms.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .motion_worker import MotionWorker
from .vr_input import VRInputReceiver

logger = logging.getLogger(__name__)


class _ArmVRPanel(QGroupBox):
    """Per-arm enable + live status."""

    def __init__(self, arm: str, worker: MotionWorker, receiver: VRInputReceiver) -> None:
        super().__init__(f"{arm.upper()} arm")
        self.arm = arm
        self.worker = worker
        self.receiver = receiver

        v = QVBoxLayout(self)

        self.btn_enable = QPushButton("VR: OFF")
        self.btn_enable.setCheckable(True)
        self.btn_enable.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; }"
            "QPushButton:checked { background-color: #27a; color: white; }"
        )
        self.btn_enable.clicked.connect(self._on_toggle)
        v.addWidget(self.btn_enable)

        self.status_mode = QLabel("Mode: —")
        self.status_clutch = QLabel("Clutch: —")
        self.status_grip = QLabel("Grip: —")
        self.status_trigger = QLabel("Trigger: —")
        self.status_pose_age = QLabel("Controller: no data")
        for lbl in (self.status_mode, self.status_clutch, self.status_grip,
                    self.status_trigger, self.status_pose_age):
            lbl.setStyleSheet("color: #666;")
            v.addWidget(lbl)

        v.addStretch(1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def _on_toggle(self, checked: bool) -> None:
        self.worker.post_vr_enable(self.arm, checked)
        # Button text is refreshed by the periodic sync; update now
        # too so the UI reflects the click immediately.
        self._set_button_text(checked)

    def _set_button_text(self, enabled: bool) -> None:
        self.btn_enable.setText(f"VR: {'ON' if enabled else 'OFF'}")

    def refresh(self) -> None:
        # Keep the button synchronised with the worker's actual flag.
        # The worker may have dropped VR enable on e-stop — we don't
        # want the button to still look "on" when it isn't.
        enabled = self.worker.vr_enabled(self.arm)
        if self.btn_enable.isChecked() != enabled:
            self.btn_enable.blockSignals(True)
            self.btn_enable.setChecked(enabled)
            self.btn_enable.blockSignals(False)
        self._set_button_text(enabled)

        self.status_mode.setText(f"Mode: {self.worker.current_mode(self.arm)}")

        # Controller state readout — same receiver the VR Info tab reads.
        state = (self.receiver.left() if self.arm == "left"
                 else self.receiver.right())
        if not state.has_ever_been_seen:
            self.status_grip.setText("Grip: (no data)")
            self.status_trigger.setText("Trigger: (no data)")
            self.status_pose_age.setText("Controller: no data")
            return

        import time
        age_ms = (time.monotonic() - state.last_rx) * 1000.0
        # Grip = dead-man. Above threshold → arm tracks controller;
        # below → clutch released, arm holds.
        from . import config as uicfg
        engaged = state.grip >= uicfg.VR_GRIP_ENABLE_THRESHOLD
        if enabled and engaged:
            self.status_clutch.setText("Clutch: ENGAGED (tracking)")
            self.status_clutch.setStyleSheet("color: #2a7; font-weight: bold;")
        elif enabled:
            self.status_clutch.setText("Clutch: released (arm held)")
            self.status_clutch.setStyleSheet("color: #888;")
        else:
            self.status_clutch.setText("Clutch: (VR disabled)")
            self.status_clutch.setStyleSheet("color: #666;")

        self.status_grip.setText(
            f"Grip: {state.grip:.2f}  (threshold {uicfg.VR_GRIP_ENABLE_THRESHOLD})"
        )
        self.status_trigger.setText(
            f"Trigger: {state.trigger:.2f}  (→ gripper, 0=open -65° / 1=closed 0°)"
        )
        self.status_pose_age.setText(
            f"Controller: last packet {age_ms:.0f} ms ago"
        )


class VRControlTab(QWidget):
    """Top-level VR Control tab. Two arm panels side by side."""

    def __init__(
        self,
        worker: MotionWorker,
        receiver: VRInputReceiver,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.worker = worker
        self.receiver = receiver

        root = QVBoxLayout(self)

        hdr = QLabel(
            "Phase 2b-α — Hard enable for VR-driven control. When ON, the "
            "arm switches to cartesian mode and (in Phase 2b-β) tracks the "
            "corresponding controller pose while grip is held. Right now "
            "this toggle only moves the arm into cartesian mode with no "
            "motor tracking of the controller."
        )
        hdr.setWordWrap(True)
        hdr.setStyleSheet("color: #888;")
        root.addWidget(hdr)

        cols = QHBoxLayout()
        self.left_panel = _ArmVRPanel("left", worker, receiver)
        self.right_panel = _ArmVRPanel("right", worker, receiver)
        cols.addWidget(self.left_panel)
        cols.addWidget(self.right_panel)
        root.addLayout(cols, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_disengage_all = QPushButton("Disengage both arms")
        self.btn_disengage_all.setStyleSheet(
            "QPushButton { padding: 8px; background-color: #733; color: white; }"
            "QPushButton:hover { background-color: #844; }"
        )
        self.btn_disengage_all.clicked.connect(self._on_disengage_all)
        btn_row.addWidget(self.btn_disengage_all)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        root.addStretch(0)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(200)  # 5 Hz

    def _on_disengage_all(self) -> None:
        self.worker.post_vr_enable("left", False)
        self.worker.post_vr_enable("right", False)
        logger.info("VR Control: disengage-all clicked")

    def _refresh(self) -> None:
        self.left_panel.refresh()
        self.right_panel.refresh()
