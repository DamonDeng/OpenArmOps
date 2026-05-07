"""Controller tab — M2.

Populates the two-column slider layout, the three-camera strip at the top, and
the per-arm torque toggles + emergency stop at the bottom. Read-only in M2:
moving a slider only updates the target readout; no action is sent to the
robot. Command dispatch lands in M3.

State flow:
  QTimer @ 5 Hz -> robot.get_observation() -> split per-arm/per-joint ->
      update sliders' "current" readout
      on first poll, sync slider target to current so commanding later won't jump
      update three QLabel camera panels from obs images

Cameras are HWC uint8 (BGR from OpenCV). QImage wants RGB888, so we swap on
conversion. We keep this dead simple (no background decode thread) because at
5 Hz it's well under budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from lerobot.robots.openarm_follower.config_openarm_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
)

from . import config
from .robot_service import RobotService

logger = logging.getLogger(__name__)

# Slider resolution — 10 ticks per degree gives smooth motion and easy mental math
SLIDER_SCALE = 10

# Camera panel size in pixels (width). Height auto from aspect ratio.
CAM_PANEL_WIDTH = 360


@dataclass
class _JointUI:
    """Per-joint widgets, held together so the poll loop can update them."""
    arm: str
    joint: str
    min_deg: float
    max_deg: float
    slider: QSlider
    target_label: QLabel   # shows the slider's target value
    current_label: QLabel  # shows the last observed position
    initialized: bool = False  # True once we've synced slider to current on first poll


class _CameraPanel(QWidget):
    """A labeled camera view. Holds a QLabel that we replace its pixmap on each poll."""

    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignCenter)
        self.image = QLabel()
        self.image.setFixedWidth(CAM_PANEL_WIDTH)
        self.image.setMinimumHeight(200)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet("background-color: #222; color: #888;")
        self.image.setText("(no frame yet)")
        layout.addWidget(self.title)
        layout.addWidget(self.image)

    def update_frame(self, hwc_uint8: np.ndarray) -> None:
        h, w = hwc_uint8.shape[:2]
        # OpenCV cameras hand us BGR; QImage wants RGB. Flip the last axis.
        rgb = np.ascontiguousarray(hwc_uint8[..., ::-1])
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaledToWidth(CAM_PANEL_WIDTH, Qt.SmoothTransformation)
        self.image.setPixmap(pix)


def _limits_for(arm: str, joint: str) -> tuple[float, float]:
    src = LEFT_DEFAULT_JOINTS_LIMITS if arm == "left" else RIGHT_DEFAULT_JOINTS_LIMITS
    return src[joint]


class ControllerTab(QWidget):
    def __init__(self, robot: RobotService, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.robot = robot

        root = QVBoxLayout(self)

        # ── Camera strip ─────────────────────────────────────────────
        cam_row = QHBoxLayout()
        self.cam_panels: dict[str, _CameraPanel] = {}
        for name in config.CAMERA_STRIP_ORDER:
            label = {"left_wrist": "LEFT wrist", "base": "BASE", "right_wrist": "RIGHT wrist"}[name]
            panel = _CameraPanel(label)
            self.cam_panels[name] = panel
            cam_row.addWidget(panel)
        root.addLayout(cam_row)

        # ── Slider columns ───────────────────────────────────────────
        cols_row = QHBoxLayout()
        self.joint_uis: dict[tuple[str, str], _JointUI] = {}
        self.torque_buttons: dict[str, QPushButton] = {}

        for arm in ("right", "left"):  # visual left→right: we put right first
            col = self._build_arm_column(arm)
            cols_row.addWidget(col)
        root.addLayout(cols_row, stretch=1)

        # ── Emergency stop ──────────────────────────────────────────
        estop = QPushButton("EMERGENCY STOP — disable torque on both arms")
        estop.setStyleSheet(
            "QPushButton { background-color: #a33; color: white; font-weight: bold; "
            "padding: 10px; }"
            "QPushButton:hover { background-color: #c44; }"
        )
        estop.clicked.connect(self._on_estop)
        root.addWidget(estop)

        # ── Poll timer ──────────────────────────────────────────────
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start(int(1000 / config.POLL_HZ))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_arm_column(self, arm: str) -> QWidget:
        box = QGroupBox(f"{arm.upper()} arm")
        v = QVBoxLayout(box)

        # Torque toggle at the top of the column
        toggle = QPushButton("Torque: OFF")
        toggle.setCheckable(True)
        toggle.setStyleSheet(
            "QPushButton { padding: 6px; }"
            "QPushButton:checked { background-color: #285; color: white; }"
        )
        toggle.clicked.connect(lambda checked, a=arm: self._on_torque(a, checked))
        self.torque_buttons[arm] = toggle
        v.addWidget(toggle)

        # Joint rows
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)  # slider expands, labels don't
        for row, joint in enumerate(config.JOINT_NAMES):
            lo, hi = _limits_for(arm, joint)
            label = QLabel(f"{joint} [{lo:+.0f}, {hi:+.0f}]°")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(lo * SLIDER_SCALE))
            slider.setMaximum(int(hi * SLIDER_SCALE))
            slider.setSingleStep(1)  # 0.1°
            slider.setPageStep(10)   # 1°
            slider.setValue(0)
            target_label = QLabel(" —.— °")
            target_label.setFixedWidth(80)
            target_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            current_label = QLabel(" —.— °")
            current_label.setFixedWidth(80)
            current_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            current_label.setStyleSheet("color: #666;")

            grid.addWidget(label, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(QLabel("tgt:"), row, 2)
            grid.addWidget(target_label, row, 3)
            grid.addWidget(QLabel("cur:"), row, 4)
            grid.addWidget(current_label, row, 5)

            slider.valueChanged.connect(
                lambda v, tl=target_label: tl.setText(f"{v / SLIDER_SCALE:+6.1f} °")
            )

            self.joint_uis[(arm, joint)] = _JointUI(
                arm=arm, joint=joint, min_deg=lo, max_deg=hi,
                slider=slider, target_label=target_label, current_label=current_label,
            )

        v.addLayout(grid)
        v.addStretch(1)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return box

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _on_torque(self, arm: str, enabled: bool) -> None:
        self.robot.set_torque(arm, enabled)
        btn = self.torque_buttons[arm]
        btn.setText(f"Torque: {'ON' if enabled else 'OFF'}")

    def _on_estop(self) -> None:
        self.robot.emergency_stop()
        for arm, btn in self.torque_buttons.items():
            btn.setChecked(False)
            btn.setText("Torque: OFF")
        logger.warning("UI: emergency stop pressed")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------
    def _poll(self) -> None:
        obs = self.robot.get_observation()
        if obs is None:
            return

        # Update joint currents (and, on first poll, sync sliders to current)
        for (arm, joint), ui in self.joint_uis.items():
            key = f"{arm}_{joint}.pos"
            val = obs.get(key)
            if val is None:
                continue
            ui.current_label.setText(f"{float(val):+6.1f} °")

            if not ui.initialized:
                # Snap slider to current so turning torque ON later doesn't jump
                ticks = int(float(val) * SLIDER_SCALE)
                ticks = max(ui.slider.minimum(), min(ui.slider.maximum(), ticks))
                ui.slider.blockSignals(True)
                ui.slider.setValue(ticks)
                ui.slider.blockSignals(False)
                ui.target_label.setText(f"{ticks / SLIDER_SCALE:+6.1f} °")
                ui.initialized = True

        # Update cameras
        for name, panel in self.cam_panels.items():
            frame = obs.get(name)
            if frame is None:
                continue
            if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.dtype == np.uint8:
                panel.update_frame(frame)
