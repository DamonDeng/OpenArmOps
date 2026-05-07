"""System tab — calibration and motor-info controls.

M1-ish scope brought forward: per-arm calibration and re-zero buttons
live here because startup no longer runs the interactive calibration flow.
Motor-info display and per-motor kp/kd editing remain for M5 / v3.

Calibration runs on a QThread so the CAN writes don't freeze the UI.
"""

from __future__ import annotations

import logging

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .robot_service import RobotService
from .runtime_state import RuntimeState

logger = logging.getLogger(__name__)


class _CalibWorker(QObject):
    """Runs a single robot_service operation on a worker thread.

    We go through a worker (rather than blocking the UI thread) because
    OpenArmFollower.bus.write_calibration iterates every motor on the CAN
    bus with blocking reads, which can take a second or two.
    """

    done = pyqtSignal(str)           # message on success
    failed = pyqtSignal(str)         # message on failure

    def __init__(self, robot: RobotService, op: str, arm: str):
        super().__init__()
        self.robot = robot
        self.op = op  # "calibrate" or "set_zero"
        self.arm = arm

    def run(self) -> None:
        try:
            if self.op == "calibrate":
                self.robot.calibrate_arm(self.arm)
                self.done.emit(f"Calibration written for {self.arm} arm.")
            elif self.op == "set_zero":
                self.robot.set_zero(self.arm)
                self.done.emit(f"Zero position set for {self.arm} arm.")
            else:
                self.failed.emit(f"unknown op: {self.op}")
        except Exception as e:
            logger.exception(f"{self.op} on {self.arm} failed")
            self.failed.emit(str(e))


class SystemTab(QWidget):
    def __init__(
        self,
        robot: RobotService,
        state: RuntimeState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.robot = robot
        self.state = state
        self._thread: QThread | None = None
        self._worker: _CalibWorker | None = None

        root = QVBoxLayout(self)

        # ── Motion settings ────────────────────────────────────────────
        speed_box = QGroupBox("Motion settings (apply to both arms)")
        speed_form = QFormLayout(speed_box)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setDecimals(1)
        self.speed_spin.setRange(0.1, 120.0)
        self.speed_spin.setSingleStep(1.0)
        self.speed_spin.setSuffix(" °/s")
        self.speed_spin.setValue(self.state.max_speed_deg_per_sec)
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        speed_form.addRow("Max commanded speed:", self.speed_spin)
        speed_hint = QLabel(
            "Per-tick step = max_speed / poll_hz. At 5 Hz, 5°/s = 1°/tick.\n"
            "Raise as confidence grows; safe bring-up value is 5°/s."
        )
        speed_hint.setStyleSheet("color: gray;")
        speed_form.addRow(speed_hint)
        root.addWidget(speed_box)

        # ── Calibration group ──────────────────────────────────────────
        cal_box = QGroupBox("Calibration")
        cal_layout = QVBoxLayout(cal_box)

        cal_help = QLabel(
            "Calibration sets the arm's current physical pose as 0° for every\n"
            "joint, then writes a calibration file with generic (-90°, +90°)\n"
            "motor ranges. Actual motion limits are still enforced by the\n"
            "per-side joint limits in config.\n\n"
            "Before calibrating: position the arm hanging straight down with\n"
            "the gripper closed. Both arms should have torque OFF."
        )
        cal_help.setWordWrap(True)
        cal_layout.addWidget(cal_help)

        cal_btn_row = QHBoxLayout()
        self.btn_cal_left = QPushButton("Calibrate LEFT arm")
        self.btn_cal_right = QPushButton("Calibrate RIGHT arm")
        self.btn_cal_left.clicked.connect(lambda: self._confirm_and_run("calibrate", "left"))
        self.btn_cal_right.clicked.connect(lambda: self._confirm_and_run("calibrate", "right"))
        cal_btn_row.addWidget(self.btn_cal_left)
        cal_btn_row.addWidget(self.btn_cal_right)
        cal_layout.addLayout(cal_btn_row)

        root.addWidget(cal_box)

        # ── Re-zero group ─────────────────────────────────────────────
        zero_box = QGroupBox("Re-zero (does not write calibration file)")
        zero_layout = QVBoxLayout(zero_box)

        zero_help = QLabel(
            "Set current pose as 0° without writing a calibration file. Useful\n"
            "if the reported state has drifted from the physical zero."
        )
        zero_help.setWordWrap(True)
        zero_layout.addWidget(zero_help)

        zero_btn_row = QHBoxLayout()
        self.btn_zero_left = QPushButton("Re-zero LEFT arm")
        self.btn_zero_right = QPushButton("Re-zero RIGHT arm")
        self.btn_zero_left.clicked.connect(lambda: self._confirm_and_run("set_zero", "left"))
        self.btn_zero_right.clicked.connect(lambda: self._confirm_and_run("set_zero", "right"))
        zero_btn_row.addWidget(self.btn_zero_left)
        zero_btn_row.addWidget(self.btn_zero_right)
        zero_layout.addLayout(zero_btn_row)

        root.addWidget(zero_box)

        # ── Placeholder for M5 ────────────────────────────────────────
        placeholder = QLabel(
            "Motor info (IDs, types, per-motor state) and editable kp / kd\n"
            "coming in M5 / v3."
        )
        placeholder.setStyleSheet("color: gray;")
        root.addWidget(placeholder)

        root.addStretch(1)

    # ------------------------------------------------------------------
    # Worker plumbing
    # ------------------------------------------------------------------
    def _confirm_and_run(self, op: str, arm: str) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "Another operation is running.")
            return

        if op == "calibrate":
            msg = (
                f"This will set the {arm} arm's current pose as 0° for every joint\n"
                f"and write a new calibration file.\n\n"
                f"Confirm:\n"
                f"  • Arm is in 'hanging straight down, gripper closed' pose\n"
                f"  • Torque is OFF on this arm\n\n"
                f"Proceed?"
            )
        else:
            msg = (
                f"Re-zero the {arm} arm at its current pose? This does not write\n"
                f"a calibration file; only motor zero positions change."
            )

        btn = QMessageBox.question(
            self, f"Confirm {op}", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if btn != QMessageBox.Yes:
            return

        self._set_buttons_enabled(False)
        self._thread = QThread(self)
        self._worker = _CalibWorker(self.robot, op, arm)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_done(self, message: str) -> None:
        self._cleanup_worker()
        QMessageBox.information(self, "Success", message)

    def _on_failed(self, message: str) -> None:
        self._cleanup_worker()
        QMessageBox.critical(self, "Failed", message)

    def _cleanup_worker(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for b in (
            self.btn_cal_left, self.btn_cal_right,
            self.btn_zero_left, self.btn_zero_right,
        ):
            b.setEnabled(enabled)

    def _on_speed_changed(self, value: float) -> None:
        self.state.max_speed_deg_per_sec = float(value)
        logger.info(f"max commanded speed set to {value:.1f} °/s")
