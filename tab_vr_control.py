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

from lerobot.robots.openarm_follower.config_openarm_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
)

from . import config
from .motion_worker import MotionWorker
from .runtime_state import RuntimeState
from .vr_input import VRInputReceiver

logger = logging.getLogger(__name__)


# Tall enough to be a comfortable target through a Pico headset where the
# operator can't precisely position the cursor, but short enough that two
# stacked panels still fit on the screen.
_VR_BUTTON_MIN_HEIGHT_PX = 90
_VR_GO_HOME_MIN_HEIGHT_PX = 64


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
        self.btn_enable.setMinimumHeight(_VR_BUTTON_MIN_HEIGHT_PX)
        # Tall, big-text button so the operator can hit it through a VR
        # headset without precise cursor placement.
        self.btn_enable.setStyleSheet(
            "QPushButton { padding: 12px; font-weight: bold; font-size: 22pt; }"
            "QPushButton:checked { background-color: #27a; color: white; }"
        )
        self.btn_enable.clicked.connect(self._on_toggle)
        v.addWidget(self.btn_enable)

        # Absolute-mode toggle — alternative tracker that locks a single
        # reference pose pair on the first grip press of a VR-enable
        # cycle and keeps it across subsequent grip releases. Useful
        # when comparing smoothness against the relative (clutch +
        # filter + dead-band) path.
        self.btn_absolute = QPushButton("Mode: relative (clutch)")
        self.btn_absolute.setCheckable(True)
        self.btn_absolute.setMinimumHeight(_VR_GO_HOME_MIN_HEIGHT_PX)
        self.btn_absolute.setStyleSheet(
            "QPushButton { padding: 8px; font-size: 14pt; }"
            "QPushButton:checked { background-color: #b73; color: white; }"
        )
        self.btn_absolute.clicked.connect(self._on_absolute_toggle)
        v.addWidget(self.btn_absolute)

        # "Slow go to zero" — works whether VR is engaged or not. We
        # disengage VR for this arm first, then post slow joint targets
        # at SLOW_SPEED_DEG_PER_SEC. Sized for headset clicking.
        self.btn_go_zero = QPushButton("Slow go to zero")
        self.btn_go_zero.setMinimumHeight(_VR_GO_HOME_MIN_HEIGHT_PX)
        self.btn_go_zero.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 16pt; }"
        )
        self.btn_go_zero.clicked.connect(self._on_slow_go_to_zero)
        v.addWidget(self.btn_go_zero)

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

    def _on_absolute_toggle(self, checked: bool) -> None:
        self.worker.post_vr_absolute(self.arm, checked)
        self._set_absolute_text(checked)

    def _set_absolute_text(self, absolute: bool) -> None:
        self.btn_absolute.setText(
            "Mode: ABSOLUTE (locked ref)" if absolute else "Mode: relative (clutch)"
        )

    def _on_slow_go_to_zero(self) -> None:
        """Disengage VR for this arm, then walk every joint to 0° at
        the slow speed. Mirrors tab_controller.ControllerTab._on_slow_go_to_zero
        but operates without slider state — the joint list comes from the
        LeRobot per-arm default joint-limits dict.
        """
        # Step 1: turn off VR for this arm so the cartesian-tick stops
        # overwriting joint targets. The worker processes commands
        # in order, so the disable lands before the joint targets.
        self.worker.post_vr_enable(self.arm, False)
        self._set_button_text(False)

        # Step 2: post slow targets to 0° for every joint + gripper.
        # Use the LeRobot package's default-limits dict as the source
        # of truth for the joint name list.
        limits = (LEFT_DEFAULT_JOINTS_LIMITS if self.arm == "left"
                  else RIGHT_DEFAULT_JOINTS_LIMITS)
        for joint, (lo, hi) in limits.items():
            target = max(lo, min(hi, 0.0))
            self.worker.post_set_target(
                self.arm, joint, target,
                deg_per_sec=config.SLOW_SPEED_DEG_PER_SEC,
            )
        logger.info(
            f"VR Control: slow go-to-zero on {self.arm} arm "
            f"at {config.SLOW_SPEED_DEG_PER_SEC} °/s "
            f"(VR auto-disengaged)"
        )

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

        absolute = self.worker.vr_absolute(self.arm)
        if self.btn_absolute.isChecked() != absolute:
            self.btn_absolute.blockSignals(True)
            self.btn_absolute.setChecked(absolute)
            self.btn_absolute.blockSignals(False)
        self._set_absolute_text(absolute)

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
        state: RuntimeState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.worker = worker
        self.receiver = receiver
        self.state = state

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

        # ── Live VR position-scale adjuster ────────────────────────
        # Big buttons so the operator can hit them with the headset on.
        # Mirrors the System-tab spinbox: both write the same
        # runtime.vr_pos_scale, so the spinbox updates automatically
        # the next time the System tab refreshes.
        scale_row = QHBoxLayout()
        self.btn_scale_down = QPushButton("− Scale")
        self.btn_scale_up = QPushButton("Scale +")
        self.lbl_scale = QLabel(self._fmt_scale())
        self.lbl_scale.setAlignment(Qt.AlignCenter)
        self.lbl_scale.setStyleSheet("font-size: 18pt; font-weight: bold;")
        for btn in (self.btn_scale_down, self.btn_scale_up):
            btn.setMinimumHeight(_VR_GO_HOME_MIN_HEIGHT_PX)
            btn.setStyleSheet(
                "QPushButton { padding: 10px; font-size: 18pt; }"
            )
        self.btn_scale_down.clicked.connect(lambda: self._nudge_scale(-1))
        self.btn_scale_up.clicked.connect(lambda: self._nudge_scale(+1))
        scale_row.addWidget(self.btn_scale_down)
        scale_row.addWidget(self.lbl_scale, stretch=1)
        scale_row.addWidget(self.btn_scale_up)
        root.addLayout(scale_row)

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

    def _fmt_scale(self) -> str:
        return f"VR pos scale: {self.state.vr_pos_scale:.2f}×"

    def _nudge_scale(self, direction: int) -> None:
        step = config.VR_SCALE_STEP * direction
        new_scale = self.state.vr_pos_scale + step
        new_scale = max(config.VR_SCALE_MIN, min(config.VR_SCALE_MAX, new_scale))
        # Round to the spinbox's step granularity so repeated nudges
        # don't drift away from on-grid values.
        new_scale = round(new_scale / config.VR_SCALE_STEP) * config.VR_SCALE_STEP
        self.state.vr_pos_scale = float(new_scale)
        self.lbl_scale.setText(self._fmt_scale())
        logger.info(f"VR Control: vr_pos_scale → {new_scale:.3f}")

    def _refresh(self) -> None:
        self.left_panel.refresh()
        self.right_panel.refresh()
        # Pick up scale changes from the System-tab spinbox so the
        # readout stays in sync regardless of which tab last touched it.
        self.lbl_scale.setText(self._fmt_scale())
