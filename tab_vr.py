"""VR tab — live debug dashboard for the Pico 4 Ultra UDP stream.

Phase 2a: read-only. No motor commands issued. Use this tab to verify
that the APK is connecting, data is arriving, coordinate axes are
labelled correctly, and buttons fire in sync with physical presses.
Motor integration lands in Phase 2b.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import config
from .vr_input import ControllerState, HeadState, StreamStats, VRInputReceiver

logger = logging.getLogger(__name__)


MONO_FONT = None  # lazy-init so we don't need a QApplication at import time


def _mono() -> QFont:
    global MONO_FONT
    if MONO_FONT is None:
        f = QFont("Monospace")
        f.setStyleHint(QFont.TypeWriter)
        MONO_FONT = f
    return MONO_FONT


class _ButtonDot(QLabel):
    """Tiny circle label that fills / outlines based on pressed state."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label
        self.setAlignment(Qt.AlignCenter)
        self.setFixedWidth(28)
        self.set_pressed(False)

    def set_pressed(self, pressed: bool) -> None:
        if pressed:
            # Filled amber dot when pressed.
            self.setText(f"● {self.label}")
            self.setStyleSheet("color: #e67; font-weight: bold;")
        else:
            self.setText(f"○ {self.label}")
            self.setStyleSheet("color: #666;")


class _ControllerPanel(QGroupBox):
    """Side-by-side panel showing every wire-format field for one
    controller. Laid out with the same left-column / right-column
    rhythm the Controller tab uses so the eye doesn't have to retrain.
    """

    def __init__(self, arm: str) -> None:
        super().__init__(f"{arm.upper()} controller")
        self.arm = arm
        layout = QVBoxLayout(self)

        # Position
        pos_box = QGroupBox("Position (m)")
        pos_grid = QGridLayout(pos_box)
        self.l_tx = QLabel(" —")
        self.l_ty = QLabel(" —")
        self.l_tz = QLabel(" —")
        for i, (name, w) in enumerate((("x", self.l_tx), ("y", self.l_ty), ("z", self.l_tz))):
            pos_grid.addWidget(QLabel(name), i, 0)
            w.setFont(_mono())
            pos_grid.addWidget(w, i, 1)
        layout.addWidget(pos_box)

        # Quaternion
        q_box = QGroupBox("Quaternion (xyzw)")
        q_grid = QGridLayout(q_box)
        self.l_qx = QLabel(" —")
        self.l_qy = QLabel(" —")
        self.l_qz = QLabel(" —")
        self.l_qw = QLabel(" —")
        for i, (name, w) in enumerate((
            ("x", self.l_qx), ("y", self.l_qy),
            ("z", self.l_qz), ("w", self.l_qw),
        )):
            q_grid.addWidget(QLabel(name), i, 0)
            w.setFont(_mono())
            q_grid.addWidget(w, i, 1)
        layout.addWidget(q_box)

        # Analog axes (trigger, grip, rate)
        ax_box = QGroupBox("Axes")
        ax_grid = QGridLayout(ax_box)
        self.l_trigger_val = QLabel("—")
        self.pb_trigger = QProgressBar()
        self.pb_trigger.setRange(0, 1000)
        self.pb_trigger.setTextVisible(False)
        self.pb_trigger.setFixedHeight(12)
        self.l_grip_val = QLabel("—")
        self.pb_grip = QProgressBar()
        self.pb_grip.setRange(0, 1000)
        self.pb_grip.setTextVisible(False)
        self.pb_grip.setFixedHeight(12)
        self.l_rate_val = QLabel("—")
        self.pb_rate = QProgressBar()
        self.pb_rate.setRange(0, 1000)
        self.pb_rate.setTextVisible(False)
        self.pb_rate.setFixedHeight(12)
        for i, (name, val_lbl, bar) in enumerate((
            ("trigger", self.l_trigger_val, self.pb_trigger),
            ("grip",    self.l_grip_val,    self.pb_grip),
            ("rate",    self.l_rate_val,    self.pb_rate),
        )):
            ax_grid.addWidget(QLabel(name), i, 0)
            bar.setMinimumWidth(160)
            ax_grid.addWidget(bar, i, 1)
            val_lbl.setFont(_mono())
            val_lbl.setMinimumWidth(60)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            ax_grid.addWidget(val_lbl, i, 2)
        layout.addWidget(ax_box)

        # Buttons
        btn_box = QGroupBox("Buttons")
        btn_row = QHBoxLayout(btn_box)
        self.btn_a = _ButtonDot("A")
        self.btn_b = _ButtonDot("B")
        self.btn_x = _ButtonDot("X")
        self.btn_y = _ButtonDot("Y")
        for w in (self.btn_a, self.btn_b, self.btn_x, self.btn_y):
            btn_row.addWidget(w)
        btn_row.addStretch(1)
        layout.addWidget(btn_box)

        # Last-seen footer
        self.l_age = QLabel("last seen: never")
        self.l_age.setStyleSheet("color: #888;")
        layout.addWidget(self.l_age)

        layout.addStretch(1)

    def update_from(self, state: ControllerState, now: float) -> None:
        if not state.has_ever_been_seen:
            self.l_age.setText("last seen: never")
            return
        self.l_tx.setText(f"{state.tx:+9.4f}")
        self.l_ty.setText(f"{state.ty:+9.4f}")
        self.l_tz.setText(f"{state.tz:+9.4f}")
        self.l_qx.setText(f"{state.qx:+9.5f}")
        self.l_qy.setText(f"{state.qy:+9.5f}")
        self.l_qz.setText(f"{state.qz:+9.5f}")
        self.l_qw.setText(f"{state.qw:+9.5f}")

        self.pb_trigger.setValue(int(max(0.0, min(1.0, state.trigger)) * 1000))
        self.l_trigger_val.setText(f"{state.trigger:5.3f}")
        self.pb_grip.setValue(int(max(0.0, min(1.0, state.grip)) * 1000))
        self.l_grip_val.setText(f"{state.grip:5.3f}")
        self.pb_rate.setValue(int(max(0.0, min(1.0, state.rate)) * 1000))
        self.l_rate_val.setText(f"{state.rate:5.3f}")

        self.btn_a.set_pressed(bool(state.a))
        self.btn_b.set_pressed(bool(state.b))
        self.btn_x.set_pressed(bool(state.x))
        self.btn_y.set_pressed(bool(state.y))

        age_ms = (now - state.last_rx) * 1000.0
        stale = age_ms > config.VR_STALE_SEC * 1000.0
        self.l_age.setText(f"last seen: {age_ms:.0f} ms ago" + (" — STALE" if stale else ""))
        self.l_age.setStyleSheet("color: #c44;" if stale else "color: #888;")


class _HeadPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("HEAD")
        g = QGridLayout(self)
        self.l_pos = QLabel("tx —  ty —  tz —")
        self.l_quat = QLabel("qx —  qy —  qz —  qw —")
        self.l_age = QLabel("last seen: never")
        self.l_pos.setFont(_mono())
        self.l_quat.setFont(_mono())
        self.l_age.setStyleSheet("color: #888;")
        g.addWidget(self.l_pos, 0, 0)
        g.addWidget(self.l_quat, 0, 1)
        g.addWidget(self.l_age, 1, 0, 1, 2)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

    def update_from(self, state: HeadState, now: float) -> None:
        if not state.has_ever_been_seen:
            self.l_pos.setText("tx —  ty —  tz —")
            self.l_quat.setText("qx —  qy —  qz —  qw —")
            self.l_age.setText("last seen: never")
            return
        self.l_pos.setText(f"tx {state.tx:+8.4f}  ty {state.ty:+8.4f}  tz {state.tz:+8.4f}")
        self.l_quat.setText(
            f"qx {state.qx:+8.5f}  qy {state.qy:+8.5f}  "
            f"qz {state.qz:+8.5f}  qw {state.qw:+8.5f}"
        )
        age_ms = (now - state.last_rx) * 1000.0
        self.l_age.setText(f"last seen: {age_ms:.0f} ms ago")


class _StreamPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Stream")
        g = QGridLayout(self)
        self.l_bind = QLabel(f"bind: {config.VR_UDP_BIND_ADDR}:{config.VR_UDP_PORT}")
        self.l_source = QLabel("from: —")
        self.l_rate = QLabel("rate: — Hz")
        self.l_total = QLabel("packets: 0")
        self.l_bytes = QLabel("bytes: 0")
        self.l_errors = QLabel("errors: 0   unknown: 0")
        self.l_last = QLabel("last: never")
        self.l_mode = QLabel("mode: —")
        self.l_calib = QLabel("last calibrate: never")

        for lbl in (self.l_bind, self.l_source, self.l_rate, self.l_total,
                    self.l_bytes, self.l_errors, self.l_last,
                    self.l_mode, self.l_calib):
            lbl.setFont(_mono())

        g.addWidget(self.l_bind,    0, 0)
        g.addWidget(self.l_source,  0, 1)
        g.addWidget(self.l_rate,    1, 0)
        g.addWidget(self.l_last,    1, 1)
        g.addWidget(self.l_total,   2, 0)
        g.addWidget(self.l_bytes,   2, 1)
        g.addWidget(self.l_errors,  3, 0, 1, 2)
        g.addWidget(self.l_mode,    4, 0)
        g.addWidget(self.l_calib,   4, 1)

    def update_from(
        self,
        stats: StreamStats,
        mode: str,
        last_calibrate: float,
        now: float,
    ) -> None:
        if stats.last_source:
            self.l_source.setText(f"from: {stats.last_source}")
        self.l_rate.setText(f"rate: {stats.rate_hz(now):5.1f} Hz")
        self.l_total.setText(f"packets: {stats.total_packets}")
        self.l_bytes.setText(f"bytes: {stats.total_bytes}")
        self.l_errors.setText(
            f"errors: {stats.parse_errors}   unknown: {stats.unknown_messages}"
        )
        if stats.last_rx > 0.0:
            age_ms = (now - stats.last_rx) * 1000.0
            stale = age_ms > config.VR_STALE_SEC * 1000.0
            self.l_last.setText(
                f"last: {age_ms:.0f} ms ago" + (" — STALE" if stale else "")
            )
            self.l_last.setStyleSheet("color: #c44;" if stale else "")
        else:
            self.l_last.setText("last: never")
            self.l_last.setStyleSheet("color: #888;")

        self.l_mode.setText(f"mode: {mode or '—'}")
        if last_calibrate > 0.0:
            ago = now - last_calibrate
            if ago < 60:
                self.l_calib.setText(f"last calibrate: {ago:.1f} s ago")
            else:
                self.l_calib.setText(f"last calibrate: {ago/60:.1f} min ago")
        else:
            self.l_calib.setText("last calibrate: never")


class VRTab(QWidget):
    """Top-level VR debug tab."""

    def __init__(self, receiver: VRInputReceiver, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.receiver = receiver

        root = QVBoxLayout(self)

        header = QLabel(
            "Phase 2a — Live readout of the Pico 4 Ultra UDP stream. No motor "
            "commands are sent yet. Wiggle a controller and watch the numbers "
            "change to verify the pipeline before we wire it into the arm."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #888;")
        root.addWidget(header)

        self.stream_panel = _StreamPanel()
        root.addWidget(self.stream_panel)

        ctrl_row = QHBoxLayout()
        self.left_panel = _ControllerPanel("left")
        self.right_panel = _ControllerPanel("right")
        ctrl_row.addWidget(self.left_panel)
        ctrl_row.addWidget(self.right_panel)
        root.addLayout(ctrl_row, stretch=1)

        self.head_panel = _HeadPanel()
        root.addWidget(self.head_panel)

        # Pull snapshots from the receiver. The receiver also emits
        # state_updated at ~10 Hz but we wire to that signal too so the
        # UI never lags more than an emit interval behind the data.
        self.receiver.state_updated.connect(self._on_state_updated)

    def _on_state_updated(self, snap: dict) -> None:
        now = time.monotonic()
        left: ControllerState = snap["left"]
        right: ControllerState = snap["right"]
        head: HeadState = snap["head"]
        stats: StreamStats = snap["stats"]
        mode: str = snap["mode"]
        last_cal: float = snap["last_calibrate_done"]

        self.left_panel.update_from(left, now)
        self.right_panel.update_from(right, now)
        self.head_panel.update_from(head, now)
        self.stream_panel.update_from(stats, mode, last_cal, now)
