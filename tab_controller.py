"""Controller tab — worker-driven version.

UI thread responsibilities:
  - Render sliders, camera strip, buttons.
  - On slider / keyboard / button events, post commands to the MotionWorker.
  - At 5 Hz, read camera frames from the robot and update panels.
  - On ``state_updated`` signal from the worker (~30 Hz), update the "cur"
    labels and amber current-position markers on each slider.

Control-loop concerns (trajectories, MIT setpoints, lead cap, send_action)
live entirely in ``motion_worker.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt5.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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
from .key_bindings import Binding, BindingTable, load_bindings
from .motion_worker import MotionWorker
from .robot_service import RobotService
from .runtime_state import RuntimeState

logger = logging.getLogger(__name__)

SLIDER_SCALE = 10         # ticks per degree (0.1° resolution)
CAM_PANEL_WIDTH = 360


class _MarkerSlider(QSlider):
    """Horizontal slider with a second tick mark painted at ``marker_deg``.

    QSlider has one handle; we paint an extra amber tick at the motor's
    observed current position so the user can watch it chase the thumb.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self._marker_ticks: Optional[int] = None

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
    target_label: QLabel
    current_label: QLabel


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
        # No channel swap: our OpenCV pipeline delivers RGB directly (verified
        # via the System-tab "Camera snapshots" diagnostic — as_received
        # versions of all three cameras showed correct colors, swapped ones
        # were inverted). LeRobot's OpenCVCamera appears to convert to RGB
        # before handing the frame to us, so the classic cv2-returns-BGR
        # assumption does not apply here. Keep an eye on this: if you ever
        # replace the camera backend, re-run the diagnostic.
        h, w = hwc_uint8.shape[:2]
        frame = np.ascontiguousarray(hwc_uint8)
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaledToWidth(CAM_PANEL_WIDTH, Qt.SmoothTransformation)
        self.image.setPixmap(pix)

    def mark_disconnected(self) -> None:
        self.image.setPixmap(QPixmap())
        self.image.setStyleSheet(
            "background-color: #2a1010; color: #e55; font-weight: bold;"
        )
        self.image.setText("CAMERA DISCONNECTED\n(restart app to recover)")


def _limits_for(arm: str, joint: str) -> tuple[float, float]:
    src = LEFT_DEFAULT_JOINTS_LIMITS if arm == "left" else RIGHT_DEFAULT_JOINTS_LIMITS
    return src[joint]


class _KeyboardFilter(QObject):
    """Application-level event filter that tracks the set of currently-held
    keys. The actual firing of bindings happens in ControllerTab's own
    QTimer (see ``_fire_held_keys``); this filter only maintains the set.

    Design rationale: OS key-repeat normally fires only the *most recently
    pressed* key in a rapid stream, which means holding `e` + `f`
    simultaneously would usually only auto-repeat `f`. For simultaneous
    multi-joint control we need to fire each held key's binding once per
    timer tick ourselves.
    """

    def __init__(self, tab: "ControllerTab") -> None:
        super().__init__()
        self.tab = tab

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        et = event.type()

        # Reset held state when our window loses focus so a key stuck in
        # the "held" set because a modal dialog grabbed focus (or an
        # alt-tab consumed the release event) can't keep firing when we
        # return.
        if et == QEvent.WindowDeactivate:
            self.tab._clear_held_keys()
            return False

        if et not in (QEvent.KeyPress, QEvent.KeyRelease):
            return False
        if not self.tab.keyboard_enabled:
            return False
        ke: QKeyEvent = event  # type: ignore[assignment]

        # Auto-repeat events are synthetic — we ignore them because we
        # run our own periodic timer. Real press/release events still
        # update the held set.
        if ke.isAutoRepeat():
            # For KeyPress auto-repeat we still consume so the default
            # widget doesn't also receive the repeat (e.g. a spinbox
            # would treat it as typing).
            return et == QEvent.KeyPress

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{et} key=0x{ke.key():X} text={ke.text()!r} "
                f"mods=0x{int(ke.modifiers()):X} obj={type(obj).__name__}"
            )

        # Resolve key character (falling back to key-code for Ctrl+letter
        # which strips text() on X11).
        text = ke.text()
        ch = text[0].lower() if text else ""
        if not ch or not ch.isalpha():
            k = ke.key()
            if Qt.Key_A <= k <= Qt.Key_Z:
                ch = chr(ord("a") + (k - Qt.Key_A))
            else:
                return False

        # If focus is on a text-entry widget, let the widget handle it.
        fw = QApplication.focusWidget()
        if fw is not None:
            from PyQt5.QtWidgets import QAbstractSpinBox, QLineEdit
            if isinstance(fw, (QAbstractSpinBox, QLineEdit)):
                return False

        if et == QEvent.KeyPress:
            # Remember that this letter is currently held. Modifier state
            # is re-read fresh every timer tick, not at press time — that
            # way "press e, then press Shift" picks up the new modifier
            # without waiting for release/re-press.
            self.tab._held_keys.add(ch)
            # Consume so the default widget doesn't also process it.
            return True

        # KeyRelease
        self.tab._held_keys.discard(ch)
        return True


class ControllerTab(QWidget):
    warning_changed = pyqtSignal(str)

    def __init__(
        self,
        robot: RobotService,
        state: RuntimeState,
        worker: MotionWorker,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.robot = robot
        self.state = state
        self.worker = worker
        self._torque_on: dict[str, bool] = {"left": False, "right": False}
        self._cameras_marked_dead = False
        self._last_warning: str = ""
        # Sliders need to be initialized once with the observed current so
        # they don't show 0° before the worker publishes state.
        self._sliders_initialized: set[tuple[str, str]] = set()

        # Keyboard bindings — reloadable at runtime via the System-tab
        # button. Default to the bindings file loaded at startup; if that
        # failed we fall through with empty tables so the filter just
        # ignores every key.
        try:
            self.bindings: BindingTable = load_bindings()
        except Exception as e:
            logger.error(f"key bindings load failed: {e}")
            from .key_bindings import BindingTable as _BT
            self.bindings = _BT(joint={}, cartesian={})
        self.keyboard_enabled: bool = True

        # Currently-held letter keys. Populated by _KeyboardFilter on
        # genuine press/release events (auto-repeat is ignored). A timer
        # in this class iterates this set at KEY_TIMER_HZ to apply
        # nudges, allowing simultaneous multi-key input.
        self._held_keys: set[str] = set()

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

        # ── Keyboard control toggle ─────────────────────────────────
        kb_row = QHBoxLayout()
        self.kb_enable = QCheckBox("Keyboard control enabled (app-wide)")
        self.kb_enable.setChecked(True)
        self.kb_enable.stateChanged.connect(self._on_kb_toggle)
        kb_row.addWidget(self.kb_enable)
        self.kb_status = QLabel("")
        self.kb_status.setStyleSheet("color: #666;")
        kb_row.addWidget(self.kb_status)
        kb_row.addStretch(1)
        root.addLayout(kb_row)
        self._refresh_kb_status_label()

        # ── Assisted poses ──────────────────────────────────────────
        # Both rows use SLOW_SPEED_DEG_PER_SEC so motion is predictable
        # regardless of the System-tab max-speed setting. Unfold is the
        # safer first step when the arm is deep in the workspace — reaching
        # out to the side clears the table before going to zero.
        unfold_row = QHBoxLayout()
        self.btn_unfold_left = QPushButton("Unfold arm — LEFT")
        self.btn_unfold_right = QPushButton("Unfold arm — RIGHT")
        for btn in (self.btn_unfold_left, self.btn_unfold_right):
            btn.setStyleSheet("QPushButton { padding: 8px; }")
        self.btn_unfold_left.clicked.connect(lambda: self._on_unfold_arm("left"))
        self.btn_unfold_right.clicked.connect(lambda: self._on_unfold_arm("right"))
        unfold_row.addWidget(self.btn_unfold_left)
        unfold_row.addWidget(self.btn_unfold_right)
        root.addLayout(unfold_row)

        zero_row = QHBoxLayout()
        self.btn_zero_left = QPushButton("Slow go to zero — LEFT arm")
        self.btn_zero_right = QPushButton("Slow go to zero — RIGHT arm")
        for btn in (self.btn_zero_left, self.btn_zero_right):
            btn.setStyleSheet("QPushButton { padding: 8px; }")
        self.btn_zero_left.clicked.connect(lambda: self._on_slow_go_to_zero("left"))
        self.btn_zero_right.clicked.connect(lambda: self._on_slow_go_to_zero("right"))
        zero_row.addWidget(self.btn_zero_left)
        zero_row.addWidget(self.btn_zero_right)
        root.addLayout(zero_row)

        # ── Emergency stop ──────────────────────────────────────────
        estop = QPushButton("EMERGENCY STOP — disable torque on both arms")
        estop.setStyleSheet(
            "QPushButton { background-color: #a33; color: white; font-weight: bold; "
            "padding: 10px; }"
            "QPushButton:hover { background-color: #c44; }"
        )
        estop.clicked.connect(self._on_estop)
        root.addWidget(estop)

        # ── Camera poll timer (UI thread, 5 Hz) ─────────────────────
        # Camera frames come from robot.get_observation's cached camera
        # buffers; it's cheap. The motion worker already reads motor state
        # at 30 Hz and publishes via state_updated — we don't need to
        # duplicate that read here.
        self.cam_timer = QTimer(self)
        self.cam_timer.timeout.connect(self._poll_cameras)
        self.cam_timer.start(int(1000 / config.POLL_HZ))

        # ── Worker signals ──────────────────────────────────────────
        self.worker.state_updated.connect(self._on_state_updated)
        self.worker.send_error.connect(self._on_send_error)

        # ── Keyboard event filter ───────────────────────────────────
        # Installed on the QApplication so key presses work from any tab
        # (the user wanted app-level capture when the toggle is on). Kept
        # on this tab as an instance attribute so it's GC-safe for the
        # whole lifetime of the UI. Filter only tracks press/release;
        # bindings are fired by _kb_timer below.
        self._kb_filter = _KeyboardFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._kb_filter)

        # ── Keyboard firing timer ───────────────────────────────────
        # 30 Hz scan of _held_keys: each held letter's binding fires once
        # per tick. Conflict-detect keys that would command the same
        # (arm, joint|axis) in opposite directions and silently drop
        # just those — non-conflicting held keys in the same tick still
        # fire, so e.g. `e + f` drives j1 and j2 simultaneously.
        self._kb_timer = QTimer(self)
        self._kb_timer.timeout.connect(self._fire_held_keys)
        self._kb_timer.start(int(1000 / 30))

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
            # Slider movement: update label + post target to worker.
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
        ui.target_label.setText(f"{deg:+6.1f} °")
        # Post to the worker. For torque-OFF joints this is harmless — the
        # worker continuously resets those trajectories to current anyway.
        self.worker.post_set_target(ui.arm, ui.joint, deg)

    def _on_torque(self, arm: str, enabled: bool) -> None:
        self.worker.post_torque(arm, enabled)
        self._torque_on[arm] = enabled
        btn = self.torque_buttons[arm]
        btn.setText(f"Torque: {'ON' if enabled else 'OFF'}")

    def _on_estop(self) -> None:
        self.worker.post_estop()
        self._torque_on = {"left": False, "right": False}
        for arm, btn in self.torque_buttons.items():
            btn.setChecked(False)
            btn.setText("Torque: OFF")
        # Sliders snap to current on the next state_updated tick because the
        # worker resets trajectories on estop; here we just update labels so
        # the UI doesn't show a stale target for a second.
        for (_arm, _joint), ui in self.joint_uis.items():
            # Mirror the slider's value → target label so they're consistent
            # with the arm's new "hold current" trajectory.
            pass  # the state_updated callback will sync within 33 ms
        logger.warning("UI: emergency stop posted to worker")

    def _on_unfold_arm(self, arm: str) -> None:
        """Move the named arm to the configured "unfold" pose at the slow speed.

        The target pose (``config.UNFOLD_ARM_POSE[arm]``) is shoulder fully
        outward, everything else at 0° — a safe intermediate when the arm
        is deep in the workspace and a direct move to zero would collide.
        Speed is fixed at ``SLOW_SPEED_DEG_PER_SEC``.
        """
        pose = config.UNFOLD_ARM_POSE[arm]
        changed = 0
        for (a, joint), ui in self.joint_uis.items():
            if a != arm:
                continue
            raw_target = float(pose.get(joint, 0.0))
            target = self._clamp(raw_target, ui.min_deg, ui.max_deg)
            self._set_slider_silent(ui, target)
            ui.target_label.setText(f"{target:+6.1f} °")
            self.worker.post_set_target(
                arm, joint, target,
                deg_per_sec=config.SLOW_SPEED_DEG_PER_SEC,
            )
            changed += 1
        logger.info(
            f"unfold arm: {arm}, {changed} slider(s) targeting pose {pose} "
            f"at {config.SLOW_SPEED_DEG_PER_SEC} °/s"
        )

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def _clear_held_keys(self) -> None:
        """Drop any keys we thought were held. Called when the app window
        loses focus so a key whose release event we never saw (because
        focus moved to another window) stops firing when we return.
        """
        if self._held_keys:
            logger.debug(f"window deactivated; clearing held keys {self._held_keys}")
            self._held_keys.clear()

    def _fire_held_keys(self) -> None:
        """Called at 30 Hz. For every held letter, resolve its binding
        (joint-mode or cartesian-mode based on the target arm's current
        mode), detect conflicts (same target, opposite directions) and
        drop only the conflicting keys, then apply the rest.
        """
        if not self.keyboard_enabled or not self._held_keys:
            return

        # Re-read modifier state each tick so "press e then press Shift"
        # picks up the new modifier without needing to re-press e.
        mods = QApplication.keyboardModifiers()
        shift = bool(mods & Qt.ShiftModifier)
        ctrl = bool(mods & Qt.ControlModifier)
        alt = bool(mods & Qt.AltModifier)
        held_count = int(shift) + int(ctrl) + int(alt)
        if held_count > 1:
            return  # multi-modifier = ambiguous, ignore as agreed
        if shift:
            modifier = "shift"
        elif ctrl:
            modifier = "ctrl"
        elif alt:
            modifier = "alt"
        else:
            modifier = "none"

        # Resolve every held letter to its binding (or drop it if no
        # binding matches the current modifier or the target-arm's mode).
        resolved: list[Binding] = []
        tables = self.bindings
        for ch in list(self._held_keys):  # list() so we can tolerate mutation
            lookup_key = (ch, modifier)
            joint_hit = tables.joint.get(lookup_key)
            cart_hit = tables.cartesian.get(lookup_key)
            for hit in (joint_hit, cart_hit):
                if hit is None:
                    continue
                if self.worker.current_mode(hit.arm) != hit.mode:
                    continue
                resolved.append(hit)
                break

        if not resolved:
            return

        # Conflict detection: group by (mode, arm, target). If a group has
        # bindings of mixed signs, drop the whole group.
        groups: dict[tuple[str, str, str], list[Binding]] = {}
        for b in resolved:
            groups.setdefault((b.mode, b.arm, b.target), []).append(b)

        for group_key, group in groups.items():
            signs = {b.direction for b in group}
            if len(signs) > 1:
                # Conflicting directions — drop this group only.
                logger.debug(f"conflict on {group_key}: dropping {len(group)} binding(s)")
                continue
            # All bindings in group have the same sign. Firing one of
            # them is sufficient (same target, same direction) — firing
            # multiple would over-nudge the same target.
            self._apply_key_nudge(group[0])

    def _on_kb_toggle(self, state: int) -> None:
        self.keyboard_enabled = bool(state == Qt.Checked)
        # Drop any keys that were mid-press when the user toggled off,
        # so toggling back on doesn't suddenly fire stale held keys.
        if not self.keyboard_enabled:
            self._clear_held_keys()
        self._refresh_kb_status_label()
        logger.info(f"keyboard control: {'on' if self.keyboard_enabled else 'off'}")

    def _refresh_kb_status_label(self) -> None:
        if not self.bindings:
            self.kb_status.setText("(no bindings loaded)")
        else:
            self.kb_status.setText(f"({len(self.bindings)} keys bound)")

    def reload_bindings(self) -> tuple[bool, str]:
        """Reload key_bindings.json from disk. Returns (success, message).
        Called by the System-tab "Reload key bindings" button.
        """
        try:
            new_bindings = load_bindings()
        except Exception as e:
            logger.error(f"reload bindings failed: {e}")
            return False, f"Reload failed: {e}"
        self.bindings = new_bindings
        self._refresh_kb_status_label()
        logger.info(f"reloaded {len(new_bindings)} key binding(s)")
        return True, f"Reloaded {len(new_bindings)} binding(s)."

    def nudge_gripper_target(self, arm: str, sign: int) -> None:
        """Nudge the named arm's gripper slider by ``sign * KEY_DELTA_DEFAULT``.

        Entry point for CartesianTab: when a user presses Alt+e/d while an
        arm is in Cartesian mode, the gripper binding routes here because
        gripper is 1-DOF and has no cartesian analog.
        """
        ui = self.joint_uis.get((arm, "gripper"))
        if ui is None:
            return
        if not self._torque_on.get(arm, False):
            return
        delta = sign * config.KEY_DELTA_DEFAULT
        new_target = self._clamp(ui.slider.value() / SLIDER_SCALE + delta,
                                 ui.min_deg, ui.max_deg)
        self._set_slider_silent(ui, new_target)
        ui.target_label.setText(f"{new_target:+6.1f} °")
        self.worker.post_set_target(arm, "gripper", new_target)

    def _apply_key_nudge(self, binding: Binding) -> None:
        """Apply one per-keypress nudge from a bound key.

        For joint-mode bindings: nudge the slider's target by 1°.
        For cartesian-mode bindings: delegate to the Cartesian tab so it
        can update its spinboxes and post a new cart target to the worker.

        Common preconditions:
        - Target arm's torque must be ON; otherwise silently drop.
        """
        arm = binding.arm
        if not self._torque_on.get(arm, False):
            return

        if binding.mode == "cartesian":
            # The Cartesian tab owns the cart-target state and UI. Route
            # this nudge there and let it handle spinbox + worker post.
            cb = getattr(self, "cartesian_nudge_callback", None)
            if cb is not None:
                cb(binding)
            return

        delta = binding.direction * config.KEY_DELTA_DEFAULT

        ui = self.joint_uis.get((arm, binding.joint))
        if ui is None:
            return

        # Nudge from the slider's current visible target rather than from
        # motor current. That way holding a key accumulates smoothly at
        # the intended rate regardless of whether the motor is tracking.
        current_target = ui.slider.value() / SLIDER_SCALE
        new_target = self._clamp(current_target + delta, ui.min_deg, ui.max_deg)
        self._set_slider_silent(ui, new_target)
        ui.target_label.setText(f"{new_target:+6.1f} °")
        self.worker.post_set_target(arm, binding.joint, new_target)

    def _on_slow_go_to_zero(self, arm: str) -> None:
        """Move the named arm to 0° at a fixed gentle speed.

        Posts a set_target to the worker for each of the arm's joints with
        ``deg_per_sec=SLOW_SPEED_DEG_PER_SEC``. This overrides the current
        System-tab max-speed setting only for these 8 trajectories; later
        user actions still use the runtime setting.
        """
        changed = 0
        for (a, joint), ui in self.joint_uis.items():
            if a != arm:
                continue
            target = self._clamp(0.0, ui.min_deg, ui.max_deg)
            self._set_slider_silent(ui, target)
            ui.target_label.setText(f"{target:+6.1f} °")
            self.worker.post_set_target(
                arm, joint, target,
                deg_per_sec=config.SLOW_SPEED_DEG_PER_SEC,
            )
            changed += 1
        logger.info(
            f"slow go-to-zero: {arm} arm, {changed} slider(s) targeting 0° "
            f"at {config.SLOW_SPEED_DEG_PER_SEC} °/s"
        )

    # ------------------------------------------------------------------
    # Worker signals
    # ------------------------------------------------------------------
    def _on_state_updated(self, state: dict) -> None:
        """Called via Qt queued signal from the motion worker thread."""
        for (arm, joint), ui in self.joint_uis.items():
            cur = state.get(f"{arm}_{joint}.pos")
            if cur is None:
                continue
            cur = float(cur)
            ui.current_label.setText(f"{cur:+6.1f} °")
            ui.slider.set_marker_ticks(int(round(cur * SLIDER_SCALE)))

            # First-time: snap the slider thumb to current so dragging later
            # doesn't start from a stale 0°.
            key = (arm, joint)
            if key not in self._sliders_initialized:
                self._set_slider_silent(ui, cur)
                ui.target_label.setText(f"{cur:+6.1f} °")
                self._sliders_initialized.add(key)

    def _on_send_error(self, msg: str) -> None:
        self._update_warning(f"send_action failed: {msg}")

    # ------------------------------------------------------------------
    # Camera poll (UI thread, 5 Hz)
    # ------------------------------------------------------------------
    def _poll_cameras(self) -> None:
        try:
            obs = self.robot.get_observation()
        except Exception as e:
            self._update_warning(f"Camera poll error: {e!s}")
            logger.exception("camera poll failed")
            return
        if obs is None:
            return

        # If cameras transitioned to dead, update panels + statusbar.
        if self.robot.cameras_dead and not self._cameras_marked_dead:
            for panel in self.cam_panels.values():
                panel.mark_disconnected()
            self._cameras_marked_dead = True
            self._update_warning(
                "Cameras disconnected — motion continues on motor state only"
            )
            return

        if self._cameras_marked_dead:
            return  # nothing to update

        for name, panel in self.cam_panels.items():
            frame = obs.get(name)
            if frame is None:
                continue
            if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.dtype == np.uint8:
                panel.update_frame(frame)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_slider_silent(self, ui: _JointUI, deg: float) -> None:
        ticks = int(round(deg * SLIDER_SCALE))
        ticks = max(ui.slider.minimum(), min(ui.slider.maximum(), ticks))
        ui.slider.blockSignals(True)
        ui.slider.setValue(ticks)
        ui.slider.blockSignals(False)

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def _update_warning(self, msg: str) -> None:
        if msg != self._last_warning:
            self._last_warning = msg
            self.warning_changed.emit(msg)
