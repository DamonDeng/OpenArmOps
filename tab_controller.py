"""Controller tab — M3 polish.

Adds the ramped control loop:

  poll tick (5 Hz):
    1. obs = robot.get_observation()
    2. for each joint:
         current = obs[joint_key]
         if arm torque OFF:
             target = current; commanded = current    # keep in sync
         else:
             delta = clamp(target - commanded, -step, +step)
             commanded += delta
    3. send_action({joint: commanded for all torque-ON arms})
    4. update UI (current labels, tgt labels, per-slider current marker)

The slider drives `target` (absolute degrees). `commanded` is what we
actually send; it chases `target` at ≤ max_step_per_tick.

Visual polish: QSlider subclass that paints a small tick mark at the
motor's current position so the user can watch it chase the thumb.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from lerobot.robots.openarm_follower.config_openarm_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
)

from . import config
from .robot_service import RobotService
from .runtime_state import RuntimeState

logger = logging.getLogger(__name__)

SLIDER_SCALE = 10           # ticks per degree (0.1° resolution)
CAM_PANEL_WIDTH = 360


class _MarkerSlider(QSlider):
    """A horizontal slider with an extra tick mark painted at ``marker_deg``.

    The built-in QSlider only has one handle. We paint a thin colored tick
    at the handle position that *would* correspond to ``marker_deg`` so
    the user can see where the motor actually is vs where they're aiming.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self._marker_ticks: Optional[int] = None  # None = don't draw

    def set_marker_ticks(self, ticks: Optional[int]) -> None:
        if ticks != self._marker_ticks:
            self._marker_ticks = ticks
            self.update()

    def paintEvent(self, event):  # noqa: N802 (Qt)
        super().paintEvent(event)
        if self._marker_ticks is None:
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )

        span = self.maximum() - self.minimum()
        if span <= 0:
            return
        frac = (self._marker_ticks - self.minimum()) / span
        x = int(groove.x() + frac * groove.width())

        painter = QPainter(self)
        pen = QPen(QColor(230, 130, 30))  # amber
        pen.setWidth(3)
        painter.setPen(pen)
        # Draw a vertical tick that extends slightly above & below the groove
        top = groove.y() - 3
        bot = groove.y() + groove.height() + 3
        painter.drawLine(x, top, x, bot)
        painter.end()


@dataclass
class _JointUI:
    arm: str
    joint: str
    min_deg: float
    max_deg: float
    slider: _MarkerSlider
    target_label: QLabel    # slider's target (what the user wants)
    current_label: QLabel   # last observed motor position
    target_deg: float = 0.0       # source of truth for slider-driven target
    commanded_deg: float = 0.0    # what we last sent to the motor
    current_deg: float = 0.0      # last observed
    initialized: bool = False     # True once we've synced to a real observation


class _CameraPanel(QWidget):
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
        rgb = np.ascontiguousarray(hwc_uint8[..., ::-1])
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaledToWidth(CAM_PANEL_WIDTH, Qt.SmoothTransformation)
        self.image.setPixmap(pix)

    def mark_disconnected(self) -> None:
        """Switch to a visible disconnected state. Idempotent."""
        self.image.setPixmap(QPixmap())
        self.image.setStyleSheet(
            "background-color: #2a1010; color: #e55; font-weight: bold;"
        )
        self.image.setText("CAMERA DISCONNECTED\n(restart app to recover)")


def _limits_for(arm: str, joint: str) -> tuple[float, float]:
    src = LEFT_DEFAULT_JOINTS_LIMITS if arm == "left" else RIGHT_DEFAULT_JOINTS_LIMITS
    return src[joint]


class ControllerTab(QWidget):
    # Emitted on important state transitions so the main window can update
    # its statusbar. Payload is the message to show.
    warning_changed = pyqtSignal(str)

    def __init__(
        self,
        robot: RobotService,
        state: RuntimeState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.robot = robot
        self.state = state
        # Per-arm torque state mirrored here so the poll loop can decide
        # whether to command each arm. Single source of truth is still the
        # QPushButton.isChecked(), but caching it avoids a lookup per tick.
        self._torque_on: dict[str, bool] = {"left": False, "right": False}
        self._cameras_marked_dead = False
        self._last_warning: str = ""

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

        for arm in ("left", "right"):
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

        toggle = QPushButton("Torque: OFF")
        toggle.setCheckable(True)
        toggle.setStyleSheet(
            "QPushButton { padding: 6px; }"
            "QPushButton:checked { background-color: #285; color: white; }"
        )
        toggle.clicked.connect(lambda checked, a=arm: self._on_torque(a, checked))
        self.torque_buttons[arm] = toggle
        v.addWidget(toggle)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        for row, joint in enumerate(config.JOINT_NAMES):
            lo, hi = _limits_for(arm, joint)
            label = QLabel(f"{joint} [{lo:+.0f}, {hi:+.0f}]°")
            slider = _MarkerSlider()
            slider.setMinimum(int(lo * SLIDER_SCALE))
            slider.setMaximum(int(hi * SLIDER_SCALE))
            slider.setSingleStep(1)
            slider.setPageStep(10)
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

            ui = _JointUI(
                arm=arm, joint=joint, min_deg=lo, max_deg=hi,
                slider=slider, target_label=target_label, current_label=current_label,
            )
            # When the slider moves, update both the label and target_deg.
            # No send_action here — the poll loop owns commanding.
            slider.valueChanged.connect(lambda v, u=ui: self._on_slider_moved(u, v))

            self.joint_uis[(arm, joint)] = ui

        v.addLayout(grid)
        v.addStretch(1)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return box

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_slider_moved(self, ui: _JointUI, ticks: int) -> None:
        deg = ticks / SLIDER_SCALE
        ui.target_deg = deg
        ui.target_label.setText(f"{deg:+6.1f} °")

    def _on_torque(self, arm: str, enabled: bool) -> None:
        # Before enabling torque, align target & commanded with the last
        # observed current so the arm doesn't lurch on the first tick.
        if enabled:
            for (a, _), ui in self.joint_uis.items():
                if a != arm:
                    continue
                if not ui.initialized:
                    continue
                ui.target_deg = ui.current_deg
                ui.commanded_deg = ui.current_deg
                self._set_slider_silent(ui, ui.current_deg)
                ui.target_label.setText(f"{ui.current_deg:+6.1f} °")

        self.robot.set_torque(arm, enabled)
        self._torque_on[arm] = enabled
        btn = self.torque_buttons[arm]
        btn.setText(f"Torque: {'ON' if enabled else 'OFF'}")

    def _on_estop(self) -> None:
        self.robot.emergency_stop()
        self._torque_on = {"left": False, "right": False}
        for arm, btn in self.torque_buttons.items():
            btn.setChecked(False)
            btn.setText("Torque: OFF")
        # Reset every target to current so re-enabling torque doesn't
        # resume the interrupted motion.
        for ui in self.joint_uis.values():
            if ui.initialized:
                ui.target_deg = ui.current_deg
                ui.commanded_deg = ui.current_deg
                self._set_slider_silent(ui, ui.current_deg)
                ui.target_label.setText(f"{ui.current_deg:+6.1f} °")
        logger.warning("UI: emergency stop; targets reset to current.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_slider_silent(self, ui: _JointUI, deg: float) -> None:
        """Move the slider without firing valueChanged (which would stomp target_deg)."""
        ticks = int(round(deg * SLIDER_SCALE))
        ticks = max(ui.slider.minimum(), min(ui.slider.maximum(), ticks))
        ui.slider.blockSignals(True)
        ui.slider.setValue(ticks)
        ui.slider.blockSignals(False)

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    # ------------------------------------------------------------------
    # The control loop
    # ------------------------------------------------------------------
    def _poll(self) -> None:
        # One outer guard. An uncaught exception in a QTimer.timeout slot
        # causes Qt to print a traceback but keep firing — which on a camera
        # failure creates the flood we saw. Better to swallow, log once, and
        # surface via the statusbar.
        try:
            self._poll_unchecked()
        except Exception as e:
            self._update_warning(f"Poll error: {e!s}")
            logger.exception("poll iteration failed")

    def _poll_unchecked(self) -> None:
        obs = self.robot.get_observation()
        if obs is None:
            self._update_warning("Observation unavailable")
            return

        # Check for camera-dead transition — update panels and statusbar once.
        if self.robot.cameras_dead and not self._cameras_marked_dead:
            for panel in self.cam_panels.values():
                panel.mark_disconnected()
            self._cameras_marked_dead = True
            self._update_warning(
                "Cameras disconnected — control loop still running on motor state only"
            )
        elif not self.robot.cameras_dead and self._last_warning:
            # Clear stale warning if cameras came back (they shouldn't, per design)
            self._update_warning("")

        step_cap = self.state.max_step_per_tick(config.POLL_HZ)
        action: dict[str, float] = {}

        for (arm, joint), ui in self.joint_uis.items():
            key = f"{arm}_{joint}.pos"
            cur = obs.get(key)
            if cur is None:
                continue
            cur = float(cur)
            ui.current_deg = cur
            ui.current_label.setText(f"{cur:+6.1f} °")
            ui.slider.set_marker_ticks(int(round(cur * SLIDER_SCALE)))

            if not ui.initialized:
                ui.target_deg = cur
                ui.commanded_deg = cur
                self._set_slider_silent(ui, cur)
                ui.target_label.setText(f"{cur:+6.1f} °")
                ui.initialized = True
                continue

            if not self._torque_on[arm]:
                ui.target_deg = cur
                ui.commanded_deg = cur
                self._set_slider_silent(ui, cur)
                ui.target_label.setText(f"{cur:+6.1f} °")
                continue

            delta = ui.target_deg - ui.commanded_deg
            if abs(delta) <= step_cap:
                ui.commanded_deg = ui.target_deg
            else:
                ui.commanded_deg += step_cap if delta > 0 else -step_cap

            ui.commanded_deg = self._clamp(ui.commanded_deg, ui.min_deg, ui.max_deg)
            action[f"{arm}_{joint}.pos"] = ui.commanded_deg

        # Update cameras — only when they're still alive. Once dead, the
        # panels display the red placeholder and we leave them alone.
        if not self._cameras_marked_dead:
            for name, panel in self.cam_panels.items():
                frame = obs.get(name)
                if frame is None:
                    continue
                if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.dtype == np.uint8:
                    panel.update_frame(frame)

        if action:
            try:
                self.robot.send_action(action)
            except Exception as e:
                logger.error(f"send_action failed: {e}")
                self._update_warning(f"send_action failed: {e!s}")

    def _update_warning(self, msg: str) -> None:
        if msg != self._last_warning:
            self._last_warning = msg
            self.warning_changed.emit(msg)
