"""System tab — calibration and motor-info controls.

M1-ish scope brought forward: per-arm calibration and re-zero buttons
live here because startup no longer runs the interactive calibration flow.
Motor-info display and per-motor kp/kd editing remain for M5 / v3.

Calibration runs on a QThread so the CAN writes don't freeze the UI.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfigBase

from . import config as uiconfig
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

        # ── Camera snapshots (diagnostic) ───────────────────────────
        snap_box = QGroupBox("Camera snapshots (diagnostic)")
        snap_layout = QVBoxLayout(snap_box)
        self.btn_snapshot = QPushButton("Save camera snapshots")
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        snap_layout.addWidget(self.btn_snapshot)
        self.snap_status = QLabel("")
        self.snap_status.setStyleSheet("color: #484;")
        snap_layout.addWidget(self.snap_status)
        root.addWidget(snap_box)

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
        root.addWidget(speed_box)

        # ── Motor info ─────────────────────────────────────────────
        motor_box = QGroupBox("Motor info (updated at 2 Hz)")
        motor_layout = QHBoxLayout(motor_box)
        self.motor_table_left = self._build_motor_table("left")
        self.motor_table_right = self._build_motor_table("right")
        motor_layout.addWidget(self.motor_table_left)
        motor_layout.addWidget(self.motor_table_right)
        root.addWidget(motor_box)

        self._motor_timer = QTimer(self)
        self._motor_timer.timeout.connect(self._refresh_motor_info)
        self._motor_timer.start(500)  # 2 Hz

        # ── Calibration group ──────────────────────────────────────────
        cal_box = QGroupBox("Calibration")
        cal_layout = QVBoxLayout(cal_box)
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
        zero_btn_row = QHBoxLayout()
        self.btn_zero_left = QPushButton("Re-zero LEFT arm")
        self.btn_zero_right = QPushButton("Re-zero RIGHT arm")
        self.btn_zero_left.clicked.connect(lambda: self._confirm_and_run("set_zero", "left"))
        self.btn_zero_right.clicked.connect(lambda: self._confirm_and_run("set_zero", "right"))
        zero_btn_row.addWidget(self.btn_zero_left)
        zero_btn_row.addWidget(self.btn_zero_right)
        zero_layout.addLayout(zero_btn_row)
        root.addWidget(zero_box)

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

    # ------------------------------------------------------------------
    # Motor info
    # ------------------------------------------------------------------
    _MOTOR_COLS = ("joint", "id", "recv", "type",
                   "pos (°)", "vel (°/s)", "torque", "tMOS (°C)", "tRotor (°C)")

    def _build_motor_table(self, arm: str) -> QGroupBox:
        """Return a QGroupBox containing a 9-column × 8-row table for one arm.

        Static columns (joint name, IDs, motor type) are filled once here.
        Dynamic columns (pos/vel/torque/temps) are populated by
        _refresh_motor_info on each tick.
        """
        box = QGroupBox(f"{arm.upper()} arm")
        layout = QVBoxLayout(box)

        # Use a throwaway config just to pull the motor_config defaults
        # (send_id, recv_id, motor_type). Not connected — just reading
        # the static metadata LeRobot compiled in.
        motor_config = OpenArmFollowerConfigBase(port="dummy").motor_config

        table = QTableWidget(len(uiconfig.JOINT_NAMES), len(self._MOTOR_COLS))
        table.setHorizontalHeaderLabels(self._MOTOR_COLS)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setFocusPolicy(0)  # don't steal focus from the Controller tab
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Rows are compact — default row height is too spacey.
        row_height = 22
        table.verticalHeader().setDefaultSectionSize(row_height)
        # Let the table expand with the parent instead of stopping at its
        # default sizeHint (which triggers the scrollbar at small heights).
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Minimum height that guarantees all 8 rows + header are visible
        # even when the window is small: header (~26 px) + 8 × row_height
        # + a few px for border.
        table.setMinimumHeight(26 + len(uiconfig.JOINT_NAMES) * row_height + 6)
        # Ask the table to honor sizeHint from contents (belt + suspenders).
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        for row, joint in enumerate(uiconfig.JOINT_NAMES):
            send_id, recv_id, motor_type = motor_config[joint]
            static = [
                joint,
                f"0x{send_id:02X}",
                f"0x{recv_id:02X}",
                motor_type,
            ]
            for col, text in enumerate(static):
                item = QTableWidgetItem(text)
                item.setForeground(self.palette().text())
                table.setItem(row, col, item)
            # Pre-populate dynamic cells with placeholders so their widths
            # don't jump when data first arrives.
            for col in range(len(static), len(self._MOTOR_COLS)):
                table.setItem(row, col, QTableWidgetItem(" — "))

        table.setObjectName(f"motor_table_{arm}")
        # Store the arm name on the table so we know what key prefix to
        # use when looking up stats.
        table.setProperty("arm", arm)
        layout.addWidget(table)

        # Make the group capture its table so later code can find it
        # via the attribute set on the parent.
        box._table = table  # type: ignore[attr-defined]
        return box

    def _refresh_motor_info(self) -> None:
        stats = self.robot.get_motor_stats()
        if stats is None:
            return

        for table_box in (self.motor_table_left, self.motor_table_right):
            table = table_box._table  # type: ignore[attr-defined]
            arm = table.property("arm")
            for row, joint in enumerate(uiconfig.JOINT_NAMES):
                key = f"{arm}_{joint}"
                s = stats.get(key)
                if s is None:
                    continue
                dynamic_values = (
                    f"{s.get('position', 0.0):+7.2f}",
                    f"{s.get('velocity', 0.0):+7.2f}",
                    f"{s.get('torque', 0.0):+6.3f}",
                    f"{s.get('temp_mos', 0.0):5.1f}",
                    f"{s.get('temp_rotor', 0.0):5.1f}",
                )
                # Static cols are 0..3; dynamic cols start at 4.
                for col_offset, text in enumerate(dynamic_values):
                    col = 4 + col_offset
                    item = table.item(row, col)
                    if item is None:
                        item = QTableWidgetItem(text)
                        table.setItem(row, col, item)
                    else:
                        item.setText(text)

    def _on_snapshot(self) -> None:
        """Grab one frame from each camera and write two PNGs per camera:
        one with the bytes as we received them from OpenCV, and one with
        the R↔B channel swap we currently apply in the UI. Pillow always
        interprets arrays as RGB on save, so the "as_received" file will
        look correct iff the camera was actually giving us RGB; the
        "swapped" file will look correct iff the camera was giving us BGR.
        """
        obs = self.robot.get_observation()
        if obs is None:
            self.snap_status.setText("No observation available — is the robot connected?")
            return

        out_dir = Path("/home/damon/workspace/openarm_space/openarm_controller_ui_lerobot/snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        written: list[str] = []
        for cam_name in ("base", "left_wrist", "right_wrist"):
            frame = obs.get(cam_name)
            if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.dtype != np.uint8:
                logger.warning(f"snapshot: skipping {cam_name} (missing or unexpected type)")
                continue
            # Variant 1: write bytes verbatim. Pillow interprets as RGB.
            p1 = out_dir / f"{ts}_{cam_name}_as_received.png"
            Image.fromarray(np.ascontiguousarray(frame)).save(p1)
            # Variant 2: apply the R↔B swap we do at display time.
            swapped = np.ascontiguousarray(frame[..., ::-1])
            p2 = out_dir / f"{ts}_{cam_name}_swapped.png"
            Image.fromarray(swapped).save(p2)
            written.append(p1.name)
            written.append(p2.name)

        if written:
            self.snap_status.setText(
                f"Wrote {len(written)} file(s) to {out_dir}\n"
                f"Compare *_as_received vs *_swapped for each camera."
            )
            logger.info(f"snapshots written to {out_dir}: {written}")
        else:
            self.snap_status.setText("No frames available to save.")
