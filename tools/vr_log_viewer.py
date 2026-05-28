"""3D replay tool for VR controller logs (the .jsonl files written by the
System tab's "Save & clear" button).

Run as a module from the repository root so the package's relative imports
resolve, even though this script doesn't depend on the live UI:

    python3 -m openarm_controller_ui_lerobot.tools.vr_log_viewer \
        ~/.openarm_ui_config/vr_recordings/vr_log_20260526_100341.jsonl

Or with no path argument — a file picker opens at the default recordings
directory.

Coordinate frame
----------------
Renders poses *as received on the wire* — the Pico/Unity OpenXR convention
(+X right, +Y up, +Z forward, left-handed). No remap is applied. This is
deliberate: the viewer's job is to show what the device actually sent us,
which is the raw material we then have to interpret in the motion worker.
A future toggle can apply ``VR_TRANSLATION_REMAP_*`` to inspect what
the worker would feed to IK.

Visualization
-------------
* World-frame tripod at the origin, dim grey, ~30 cm long, with X/Y/Z
  labelled red/green/blue. Static reference.
* LEFT controller and RIGHT controller each rendered as:
    - a small wireframe cube (~5 cm) at the controller's position
    - a full RGB tripod (X red, Y green, Z blue, ~15 cm) rotated by the
      controller's quaternion
* Optional breadcrumb trails — the last N positions as a thin polyline.

Playback controls
-----------------
* Timeline slider for scrubbing
* Play / Pause / Step ± / Jump ±10
* Speed selector: 0.1× / 0.25× / 0.5× / 1× / 2× / 4×
* Filter checkboxes: show LEFT, show RIGHT, skip-synthetic
* Side panel: per-controller text readout (pos, quat, grip, buttons)

Synthetic packets
-----------------
The APK sends an exact-zero pose (pos=0, qx=qy=qz=0, qw=1) on grip release
and continues to send identity packets while grip stays disengaged. The
"Skip synthetic" filter elides those frames so the playback shows only
real motion.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Pin pyqtgraph to PyQt5. ``apt install python3-pyqtgraph`` pulled in
# python3-pyqt6 as a recommended dep, so without this hint pyqtgraph may
# pick PyQt6 — and its GLViewWidget would then live in a Qt6 application
# context while our QApplication / QFileDialog use PyQt5, triggering
# "Must construct a QApplication before a QWidget" the moment we build
# the 3D view. Setting the env var before the import resolves the binding.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Log parsing
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Packet:
    t: float           # monotonic seconds from the recorder
    kind: str          # "LEFT" | "RIGHT" | "HEAD" | "MODE" | "CALIBRATE_DONE"
    source: str
    raw: str           # original ASCII line, kept for the readout panel

    # Parsed fields (only valid when kind in {LEFT, RIGHT, HEAD}):
    pos: np.ndarray = None        # shape (3,)
    quat: np.ndarray = None       # (qx, qy, qz, qw)
    trigger: float = 0.0
    grip: float = 0.0
    a: int = 0
    b: int = 0
    x: int = 0
    y: int = 0
    rate: float = 0.0
    ts_ns: int = 0
    synthetic: bool = False


_SYNTH_POS_TOL = 1e-9
_SYNTH_QW_TOL = 1e-9


def _is_synthetic(pos: np.ndarray, quat: np.ndarray) -> bool:
    """Exact-zero pos with identity quaternion (qx=qy=qz=0, qw=1) is the
    APK's "grip released" sentinel. Real IMU readings never hit zero on
    the dot.
    """
    return (
        np.all(np.abs(pos) < _SYNTH_POS_TOL)
        and abs(quat[0]) < _SYNTH_POS_TOL
        and abs(quat[1]) < _SYNTH_POS_TOL
        and abs(quat[2]) < _SYNTH_POS_TOL
        and abs(quat[3] - 1.0) < _SYNTH_QW_TOL
    )


def parse_log(path: Path) -> list[Packet]:
    """Read a .jsonl recorder file and return a chronologically-ordered
    list of Packet objects. Malformed lines are skipped with a warning.
    """
    packets: list[Packet] = []
    with path.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"line {ln_no}: bad JSON ({e}); skipping")
                continue
            t = float(rec.get("t", 0.0))
            source = str(rec.get("from", ""))
            raw = str(rec.get("raw", ""))
            tokens = raw.split()
            if not tokens:
                continue
            kind = tokens[0].upper()
            pkt = Packet(t=t, kind=kind, source=source, raw=raw)
            try:
                if kind in ("LEFT", "RIGHT") and len(tokens) >= 16:
                    pkt.pos = np.array([float(tokens[1]), float(tokens[2]),
                                        float(tokens[3])], dtype=float)
                    pkt.quat = np.array([float(tokens[4]), float(tokens[5]),
                                         float(tokens[6]), float(tokens[7])],
                                        dtype=float)
                    pkt.trigger = float(tokens[8])
                    pkt.grip = float(tokens[9])
                    pkt.a = int(float(tokens[10]))
                    pkt.b = int(float(tokens[11]))
                    pkt.x = int(float(tokens[12]))
                    pkt.y = int(float(tokens[13]))
                    pkt.rate = float(tokens[14])
                    pkt.ts_ns = int(float(tokens[15]))
                    pkt.synthetic = _is_synthetic(pkt.pos, pkt.quat)
                elif kind == "HEAD" and len(tokens) >= 9:
                    pkt.pos = np.array([float(tokens[1]), float(tokens[2]),
                                        float(tokens[3])], dtype=float)
                    pkt.quat = np.array([float(tokens[4]), float(tokens[5]),
                                         float(tokens[6]), float(tokens[7])],
                                        dtype=float)
                    pkt.ts_ns = int(float(tokens[8]))
            except (ValueError, IndexError) as e:
                logger.warning(f"line {ln_no}: parse error ({e}); raw={raw!r}")
                continue
            packets.append(pkt)
    packets.sort(key=lambda p: p.t)
    return packets


# ──────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────
def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert (qx, qy, qz, qw) to a 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1 - (yy + zz), xy - wz,       xz + wy],
        [xy + wz,       1 - (xx + zz), yz - wx],
        [xz - wy,       yz + wx,       1 - (xx + yy)],
    ])


# ──────────────────────────────────────────────────────────────────────────
# Scene primitives
# ──────────────────────────────────────────────────────────────────────────
class Tripod:
    """Three colored line segments showing a coordinate frame.

    Position is set by ``set_pose(pos, R)`` where ``R`` is a 3x3 rotation
    matrix; we draw axes along the rotated basis vectors. Each axis is a
    separate ``GLLinePlotItem`` so colors render distinctly.
    """

    def __init__(self, view: gl.GLViewWidget, length: float, width: float = 3.0,
                 alpha: float = 1.0):
        self.length = length
        self.x = gl.GLLinePlotItem(width=width, antialias=True,
                                   color=(1.0, 0.2, 0.2, alpha))
        self.y = gl.GLLinePlotItem(width=width, antialias=True,
                                   color=(0.2, 1.0, 0.2, alpha))
        self.z = gl.GLLinePlotItem(width=width, antialias=True,
                                   color=(0.3, 0.5, 1.0, alpha))
        for item in (self.x, self.y, self.z):
            view.addItem(item)
        self.set_pose(np.zeros(3), np.eye(3))

    def set_pose(self, pos: np.ndarray, R: np.ndarray) -> None:
        L = self.length
        ex = pos + R @ np.array([L, 0, 0])
        ey = pos + R @ np.array([0, L, 0])
        ez = pos + R @ np.array([0, 0, L])
        self.x.setData(pos=np.array([pos, ex]))
        self.y.setData(pos=np.array([pos, ey]))
        self.z.setData(pos=np.array([pos, ez]))

    def set_visible(self, visible: bool) -> None:
        for item in (self.x, self.y, self.z):
            item.setVisible(visible)


class WireCube:
    """12-edge wireframe cube. We use it as a body marker around the
    controller's reported position so the user can see the controller
    has volume. Cube faces follow the controller's rotation.
    """

    # 8 cube vertices in the local frame (centered at origin), edge length 1.
    _VERTS = 0.5 * np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=float)
    _EDGES = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),     # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),     # top face
        (0, 4), (1, 5), (2, 6), (3, 7),     # verticals
    ])

    def __init__(self, view: gl.GLViewWidget, edge_length: float,
                 color: tuple[float, float, float, float], width: float = 1.5):
        self.edge_length = edge_length
        self.color = color
        # Build one GLLinePlotItem per edge so we can move them all by
        # rebuilding their endpoints in set_pose.
        self.items = []
        for _ in range(len(self._EDGES)):
            it = gl.GLLinePlotItem(width=width, antialias=True, color=color)
            view.addItem(it)
            self.items.append(it)
        self.set_pose(np.zeros(3), np.eye(3))

    def set_pose(self, pos: np.ndarray, R: np.ndarray) -> None:
        local = self._VERTS * self.edge_length
        world = (R @ local.T).T + pos
        for it, (a, b) in zip(self.items, self._EDGES):
            it.setData(pos=np.array([world[a], world[b]]))

    def set_visible(self, visible: bool) -> None:
        for it in self.items:
            it.setVisible(visible)


class Trail:
    """Faint polyline of the last N positions. Cheap to maintain — we
    keep a deque-like ring buffer and rebuild the polyline on each
    update. N is small enough (default 200) that np.array() per frame
    is negligible compared to GL draw cost.
    """

    def __init__(self, view: gl.GLViewWidget, max_len: int,
                 color: tuple[float, float, float, float], width: float = 1.0):
        self.max_len = max_len
        self.points: list[np.ndarray] = []
        self.item = gl.GLLinePlotItem(width=width, antialias=True, color=color)
        view.addItem(self.item)

    def reset(self) -> None:
        self.points = []
        self.item.setData(pos=np.zeros((0, 3)))

    def append(self, pos: np.ndarray) -> None:
        self.points.append(pos.copy())
        if len(self.points) > self.max_len:
            self.points = self.points[-self.max_len:]
        if len(self.points) >= 2:
            self.item.setData(pos=np.array(self.points))
        else:
            self.item.setData(pos=np.zeros((0, 3)))

    def set_visible(self, visible: bool) -> None:
        self.item.setVisible(visible)


# ──────────────────────────────────────────────────────────────────────────
# Button indicator widget (per-controller)
# ──────────────────────────────────────────────────────────────────────────
class ButtonPad(QWidget):
    """A row of four colored squares (A B X Y) plus two filled bars
    (trigger, grip). Updated from a Packet via ``set_state``.

    Visual contract:
      - Square is dim outline when button=0, filled bright when button=1.
      - Bars fill 0–100% based on the analog 0..1 value, with a marker
        line at 0.5 (the engage threshold).
    """

    _BTN_COLORS = {
        "A": QColor(255, 80, 80),
        "B": QColor(80, 200, 80),
        "X": QColor(80, 140, 255),
        "Y": QColor(255, 200, 60),
    }

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setMinimumWidth(220)
        self._a = 0
        self._b = 0
        self._x = 0
        self._y = 0
        self._trigger = 0.0
        self._grip = 0.0

    def set_state(self, a: int, b: int, x: int, y: int,
                  trigger: float, grip: float) -> None:
        self._a, self._b, self._x, self._y = a, b, x, y
        self._trigger = trigger
        self._grip = grip
        self.update()

    def paintEvent(self, ev) -> None:  # noqa: N802 (Qt signature)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        # Top half: button squares. Bottom half: trigger + grip bars.
        sq_size = 28
        spacing = 8
        total_btn_w = 4 * sq_size + 3 * spacing
        x0 = (w - total_btn_w) // 2
        y0 = 4
        for i, (name, val) in enumerate(
                (("A", self._a), ("B", self._b),
                 ("X", self._x), ("Y", self._y))):
            color = self._BTN_COLORS[name]
            x = x0 + i * (sq_size + spacing)
            rect = QRectF(x, y0, sq_size, sq_size)
            if val:
                p.setBrush(color)
                p.setPen(QPen(color.lighter(140), 2))
            else:
                p.setBrush(QColor(40, 42, 48))
                p.setPen(QPen(color.darker(160), 1))
            p.drawRoundedRect(rect, 4, 4)
            # Letter label
            p.setPen(QColor(240, 240, 240) if val else color.darker(120))
            font = QFont("monospace", 11)
            font.setBold(True)
            p.setFont(font)
            p.drawText(rect, Qt.AlignCenter, name)

        # Bars below the buttons.
        bar_y = y0 + sq_size + 6
        bar_h = 12
        bar_w = w - 80
        for label, value, hue in (
                ("trig", self._trigger, QColor(180, 180, 220)),
                ("grip", self._grip,    QColor(220, 180, 120))):
            p.setPen(QColor(200, 200, 200))
            p.setFont(QFont("monospace", 9))
            p.drawText(QRectF(4, bar_y, 36, bar_h),
                       Qt.AlignVCenter | Qt.AlignLeft, label)
            # Track
            p.setBrush(QColor(40, 42, 48))
            p.setPen(QPen(QColor(80, 80, 90), 1))
            p.drawRect(QRectF(40, bar_y, bar_w, bar_h))
            # Fill
            fill_w = max(0.0, min(1.0, value)) * bar_w
            p.setBrush(hue)
            p.setPen(Qt.NoPen)
            p.drawRect(QRectF(40, bar_y, fill_w, bar_h))
            # 0.5 threshold marker
            p.setPen(QPen(QColor(255, 100, 100), 1, Qt.DashLine))
            mid_x = 40 + 0.5 * bar_w
            p.drawLine(int(mid_x), bar_y, int(mid_x), bar_y + bar_h)
            # Numeric readout
            p.setPen(QColor(240, 240, 240))
            p.drawText(QRectF(40, bar_y, bar_w, bar_h),
                       Qt.AlignCenter, f"{value:.2f}")
            bar_y += bar_h + 4
        p.end()


# ──────────────────────────────────────────────────────────────────────────
# Timeline tick strip — colored ticks marking button-press packets.
# ──────────────────────────────────────────────────────────────────────────
class ButtonTimelineStrip(QWidget):
    """A thin horizontal strip aligned under the slider showing one tick
    per packet that has any button pressed. Tick colors mirror the button:
    A=red, B=green, X=blue, Y=yellow. Two rows: top=LEFT controller,
    bottom=RIGHT, so you can tell at a glance which side the press came
    from. A vertical line marks the current play head.
    """

    _COLORS = {
        "A": QColor(255, 80, 80),
        "B": QColor(80, 200, 80),
        "X": QColor(80, 140, 255),
        "Y": QColor(255, 200, 60),
    }

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._packets: list[Packet] = []
        self._index = 0

    def set_packets(self, packets: list[Packet]) -> None:
        self._packets = packets
        self.update()

    def set_index(self, idx: int) -> None:
        self._index = idx
        self.update()

    def _tick_x(self, idx: int) -> float:
        if not self._packets:
            return 0.0
        return (idx / max(1, len(self._packets) - 1)) * (self.width() - 1)

    def paintEvent(self, ev) -> None:  # noqa: N802
        p = QPainter(self)
        # Background
        p.fillRect(self.rect(), QColor(28, 30, 36))
        if not self._packets:
            return
        h = self.height()
        # Two horizontal bands: LEFT (top), RIGHT (bottom).
        left_y0, left_y1 = 2, h // 2 - 1
        right_y0, right_y1 = h // 2 + 1, h - 2
        # Faint band backgrounds
        p.fillRect(QRectF(0, left_y0, self.width(), left_y1 - left_y0),
                   QColor(36, 38, 46))
        p.fillRect(QRectF(0, right_y0, self.width(), right_y1 - right_y0),
                   QColor(36, 38, 46))
        # Side labels
        p.setPen(QColor(140, 140, 160))
        p.setFont(QFont("monospace", 7))
        p.drawText(QRectF(2, left_y0, 24, left_y1 - left_y0),
                   Qt.AlignVCenter | Qt.AlignLeft, "L")
        p.drawText(QRectF(2, right_y0, 24, right_y1 - right_y0),
                   Qt.AlignVCenter | Qt.AlignLeft, "R")
        # Draw a tick per packet that has any button pressed. We draw all
        # active button colors stacked horizontally within the same x to
        # show multi-button presses (rare but possible).
        for idx, pkt in enumerate(self._packets):
            if pkt.kind not in ("LEFT", "RIGHT"):
                continue
            actives = []
            for name, val in (("A", pkt.a), ("B", pkt.b),
                              ("X", pkt.x), ("Y", pkt.y)):
                if val:
                    actives.append(name)
            if not actives:
                continue
            x = self._tick_x(idx)
            y0 = left_y0 if pkt.kind == "LEFT" else right_y0
            y1 = left_y1 if pkt.kind == "LEFT" else right_y1
            # Stack actives top-to-bottom within the band so multi-press
            # is visible. Single press just fills the whole band.
            band_h = y1 - y0
            sub_h = band_h / len(actives)
            for k, name in enumerate(actives):
                p.setPen(Qt.NoPen)
                p.setBrush(self._COLORS[name])
                p.drawRect(QRectF(x, y0 + k * sub_h, 2.0, sub_h))
        # Play head
        if self._packets:
            xh = self._tick_x(self._index)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1))
            p.drawLine(int(xh), 0, int(xh), h)
        p.end()


# ──────────────────────────────────────────────────────────────────────────
# Main viewer window
# ──────────────────────────────────────────────────────────────────────────
class VRLogViewer(QMainWindow):
    SPEED_OPTIONS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    DEFAULT_SPEED_INDEX = 3   # 1.0×

    def __init__(self, packets: list[Packet], log_path: Path):
        super().__init__()
        self.packets = packets
        self.log_path = log_path

        # Build a per-kind index list so we can step through LEFT-only or
        # RIGHT-only frames if the user wants. The slider scrubs over the
        # full packet list (all kinds) so the timeline is faithful to the
        # log's actual cadence.
        self.t0 = packets[0].t if packets else 0.0
        self.duration = (packets[-1].t - self.t0) if packets else 0.0

        self.setWindowTitle(f"VR log viewer — {log_path.name}  "
                            f"({len(packets)} packets, {self.duration:.1f} s)")
        self.resize(1400, 900)

        # ── 3D scene ─────────────────────────────────────────────────
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((20, 22, 28))
        # ~1 m camera distance — controller positions are typically <0.5 m
        # from the device origin, so this frames the action well.
        self.view.opts['distance'] = 1.2
        self.view.opts['elevation'] = 25
        self.view.opts['azimuth'] = 45

        # World-frame tripod (faint, longer than the controller tripods so
        # it stays distinguishable when controllers are near origin).
        self._world_tripod = Tripod(self.view, length=0.30, width=2.0,
                                    alpha=0.35)
        # Controllers
        self._left_tripod = Tripod(self.view, length=0.15, width=3.0)
        self._right_tripod = Tripod(self.view, length=0.15, width=3.0)
        self._left_cube = WireCube(self.view, edge_length=0.05,
                                   color=(0.4, 0.6, 1.0, 0.9))
        self._right_cube = WireCube(self.view, edge_length=0.05,
                                    color=(1.0, 0.6, 0.4, 0.9))
        self._left_trail = Trail(self.view, max_len=200,
                                 color=(0.4, 0.6, 1.0, 0.5))
        self._right_trail = Trail(self.view, max_len=200,
                                  color=(1.0, 0.6, 0.4, 0.5))

        # Last seen pose per kind (since LEFT and RIGHT arrive interleaved,
        # we want to keep both visible at their most recent values whenever
        # the slider is on a packet of either kind).
        self._last_pose: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "LEFT":  (np.zeros(3), np.eye(3)),
            "RIGHT": (np.zeros(3), np.eye(3)),
        }
        self._last_packet: dict[str, Packet] = {}

        # ── Side panel: per-controller text readout ──────────────────
        self.label_left = QLabel("LEFT  : (no data)")
        self.label_left.setFont(QFont("monospace"))
        self.label_left.setStyleSheet("color: #6cf;")
        self.label_right = QLabel("RIGHT : (no data)")
        self.label_right.setFont(QFont("monospace"))
        self.label_right.setStyleSheet("color: #fc6;")
        self.label_meta = QLabel("")
        self.label_meta.setFont(QFont("monospace"))

        # Per-controller button-state pads. These show A/B/X/Y as filled
        # squares + trigger/grip as analog bars. They're driven from the
        # _last_packet for each kind so both pads keep showing meaningful
        # state even on alternating LEFT/RIGHT packets.
        self.pad_left = ButtonPad()
        self.pad_right = ButtonPad()

        # ── Filter checkboxes ────────────────────────────────────────
        self.cb_show_left = QCheckBox("Show LEFT")
        self.cb_show_right = QCheckBox("Show RIGHT")
        self.cb_show_left.setChecked(True)
        self.cb_show_right.setChecked(True)
        self.cb_show_left.toggled.connect(self._on_show_left)
        self.cb_show_right.toggled.connect(self._on_show_right)

        self.cb_skip_synth = QCheckBox("Skip synthetic packets")
        self.cb_skip_synth.setChecked(False)
        self.cb_skip_synth.setToolTip(
            "Skip packets where pos=(0,0,0) and quat=(0,0,0,1) — the APK "
            "emits these on grip release. Off by default so you can see "
            "the reset behaviour."
        )

        self.cb_show_trail = QCheckBox("Show trails")
        self.cb_show_trail.setChecked(True)
        self.cb_show_trail.toggled.connect(self._on_show_trail)

        # ── Playback controls ────────────────────────────────────────
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, len(packets) - 1))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.valueChanged.connect(self._on_slider_changed)

        self.btn_first = QPushButton("⏮")
        self.btn_back10 = QPushButton("⏪")
        self.btn_step_back = QPushButton("◀")
        self.btn_play = QPushButton("▶")
        self.btn_step_fwd = QPushButton("▶|")
        self.btn_fwd10 = QPushButton("⏩")
        self.btn_last = QPushButton("⏭")

        self.btn_first.clicked.connect(lambda: self._set_index(0))
        self.btn_back10.clicked.connect(lambda: self._step(-10))
        self.btn_step_back.clicked.connect(lambda: self._step(-1))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_step_fwd.clicked.connect(lambda: self._step(1))
        self.btn_fwd10.clicked.connect(lambda: self._step(10))
        self.btn_last.clicked.connect(
            lambda: self._set_index(len(self.packets) - 1))

        self.combo_speed = QComboBox()
        for s in self.SPEED_OPTIONS:
            self.combo_speed.addItem(f"{s:g}×", s)
        self.combo_speed.setCurrentIndex(self.DEFAULT_SPEED_INDEX)

        self.btn_load = QPushButton("Load file…")
        self.btn_load.clicked.connect(self._on_load_file)

        # Jump to previous / next packet that has any A/B/X/Y pressed.
        # Useful for recordings where button presses are sparse (e.g. an
        # XYXY sentinel buried in a few minutes of motion).
        self.btn_prev_btn = QPushButton("⟨ btn")
        self.btn_next_btn = QPushButton("btn ⟩")
        self.btn_prev_btn.setToolTip(
            "Jump to the previous packet with any A/B/X/Y pressed."
        )
        self.btn_next_btn.setToolTip(
            "Jump to the next packet with any A/B/X/Y pressed."
        )
        self.btn_prev_btn.clicked.connect(self._jump_prev_button)
        self.btn_next_btn.clicked.connect(self._jump_next_button)

        # Timeline tick strip showing every button-press packet across
        # the whole recording. Uses the same horizontal scale as the
        # slider above it.
        self.button_strip = ButtonTimelineStrip()
        self.button_strip.set_packets(packets)

        # ── Status bar ───────────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._lbl_idx = QLabel("0/0")
        self._lbl_time = QLabel("t=0.000 s")
        self.status.addWidget(self._lbl_idx)
        self.status.addWidget(self._lbl_time)

        # ── Layout ───────────────────────────────────────────────────
        side = QVBoxLayout()
        side.setContentsMargins(8, 8, 8, 8)

        info_box = QGroupBox("Current packet")
        info_layout = QVBoxLayout(info_box)
        info_layout.addWidget(self.label_left)
        info_layout.addWidget(self._labelled_pad("LEFT input", self.pad_left,
                                                 "#6cf"))
        info_layout.addWidget(self.label_right)
        info_layout.addWidget(self._labelled_pad("RIGHT input", self.pad_right,
                                                 "#fc6"))
        info_layout.addWidget(self.label_meta)
        side.addWidget(info_box)

        filt_box = QGroupBox("Display")
        filt_layout = QVBoxLayout(filt_box)
        filt_layout.addWidget(self.cb_show_left)
        filt_layout.addWidget(self.cb_show_right)
        filt_layout.addWidget(self.cb_show_trail)
        filt_layout.addWidget(self.cb_skip_synth)
        side.addWidget(filt_box)

        legend = QLabel(
            "<b>Legend</b><br>"
            "<span style='color:#f33'>X</span> &nbsp;"
            "<span style='color:#3f3'>Y</span> &nbsp;"
            "<span style='color:#6af'>Z</span> &nbsp;(world & controller)<br>"
            "<span style='color:#6cf'>LEFT cube/trail</span> · "
            "<span style='color:#fc6'>RIGHT cube/trail</span><br>"
            "Frame: VR (Pico/OpenXR): +X right, +Y up, +Z forward (LH)"
        )
        legend.setWordWrap(True)
        side.addWidget(legend)
        side.addStretch(1)

        playback_row = QHBoxLayout()
        for b in (self.btn_first, self.btn_back10, self.btn_step_back,
                  self.btn_play, self.btn_step_fwd, self.btn_fwd10,
                  self.btn_last):
            playback_row.addWidget(b)
        playback_row.addWidget(QLabel("Speed:"))
        playback_row.addWidget(self.combo_speed)
        # Button-press navigation
        playback_row.addSpacing(12)
        playback_row.addWidget(self.btn_prev_btn)
        playback_row.addWidget(self.btn_next_btn)
        playback_row.addStretch(1)
        playback_row.addWidget(self.btn_load)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self.view, stretch=1)
        center_layout.addWidget(self.slider)
        center_layout.addWidget(self.button_strip)
        center_layout.addLayout(playback_row)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(center, stretch=1)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMinimumWidth(360)
        side_widget.setMaximumWidth(420)
        root_layout.addWidget(side_widget)
        self.setCentralWidget(root)

        # ── Playback state ───────────────────────────────────────────
        self._index = 0
        self._playing = False
        # The play timer fires at 60 Hz and advances ``self._index`` based
        # on log-time-elapsed × speed. This decouples replay smoothness
        # from packet rate (some bursts are 10 packets per ms; we'd
        # otherwise blow through them in a single frame).
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 Hz
        self._timer.timeout.connect(self._advance_play)
        self._play_anchor_wall: float = 0.0
        self._play_anchor_log: float = 0.0

        if packets:
            self._set_index(0)

    # ─────────────────────────────────────────────────────────────────
    # Slider / button slots
    # ─────────────────────────────────────────────────────────────────
    def _on_slider_changed(self, value: int) -> None:
        # Ignore programmatic re-emits while we're updating the slider
        # to match the play head. The signal is connected unconditionally
        # but _set_index uses blockSignals when it's the source.
        self._set_index(value, slider_initiated=True)

    def _set_index(self, idx: int, slider_initiated: bool = False) -> None:
        if not self.packets:
            return
        idx = max(0, min(idx, len(self.packets) - 1))
        self._index = idx
        if not slider_initiated:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
        # Reset play anchor any time we jump non-monotonically — otherwise
        # the next _advance_play tick will try to "catch up" through every
        # skipped packet.
        if self._playing:
            self._play_anchor_wall = time.monotonic()
            self._play_anchor_log = self.packets[idx].t
        self._render_at_index(idx)

    def _step(self, delta: int) -> None:
        if self.cb_skip_synth.isChecked() and self.packets:
            # Walk past synthetic packets so a "step" advances by a
            # meaningful packet, not one skipped frame.
            i = self._index
            step = 1 if delta > 0 else -1
            remaining = abs(delta)
            while remaining > 0:
                i += step
                if i < 0 or i >= len(self.packets):
                    break
                if self.packets[i].synthetic:
                    continue
                remaining -= 1
            self._set_index(i)
        else:
            self._set_index(self._index + delta)

    def _toggle_play(self) -> None:
        if not self.packets:
            return
        self._playing = not self._playing
        self.btn_play.setText("⏸" if self._playing else "▶")
        if self._playing:
            self._play_anchor_wall = time.monotonic()
            self._play_anchor_log = self.packets[self._index].t
            self._timer.start()
        else:
            self._timer.stop()

    @staticmethod
    def _has_button(pkt: "Packet") -> bool:
        return pkt.kind in ("LEFT", "RIGHT") and bool(
            pkt.a or pkt.b or pkt.x or pkt.y
        )

    def _jump_prev_button(self) -> None:
        for i in range(self._index - 1, -1, -1):
            if self._has_button(self.packets[i]):
                self._set_index(i)
                return
        # Wrap-around announce via status, but don't jump to avoid surprises.
        self.status.showMessage("No earlier button-press packet.", 2000)

    def _jump_next_button(self) -> None:
        for i in range(self._index + 1, len(self.packets)):
            if self._has_button(self.packets[i]):
                self._set_index(i)
                return
        self.status.showMessage("No later button-press packet.", 2000)

    @staticmethod
    def _labelled_pad(title: str, pad: "ButtonPad", color_hex: str) -> QWidget:
        """Wrap a ButtonPad in a thin frame with a small title label so
        the side panel groups (label_*, pad_*) read as a unit.
        """
        wrapper = QFrame()
        wrapper.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(4, 0, 4, 4)
        layout.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {color_hex}; font-weight: bold;")
        title_lbl.setFont(QFont("monospace", 8))
        layout.addWidget(title_lbl)
        layout.addWidget(pad)
        return wrapper

    def _advance_play(self) -> None:
        if not self._playing or not self.packets:
            return
        speed = self.combo_speed.currentData()
        elapsed_wall = time.monotonic() - self._play_anchor_wall
        target_log_t = self._play_anchor_log + elapsed_wall * speed
        # Walk forward through packets up to target_log_t.
        i = self._index
        end = len(self.packets) - 1
        skip_synth = self.cb_skip_synth.isChecked()
        while i < end and self.packets[i + 1].t <= target_log_t:
            i += 1
            if skip_synth and self.packets[i].synthetic:
                # Don't render it — but we still advance past it so the
                # play head keeps up with wall time.
                continue
        if i >= end:
            self._playing = False
            self._timer.stop()
            self.btn_play.setText("▶")
        if i != self._index:
            self._set_index(i)

    # ─────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────
    def _render_at_index(self, idx: int) -> None:
        """Update the scene to show the controller pose at ``packets[idx]``.

        We don't reset previously-drawn controllers — LEFT and RIGHT updates
        arrive interleaved, so the *other* controller stays at its last seen
        pose. Trail-append happens for the current packet only.
        """
        pkt = self.packets[idx]
        t_rel = pkt.t - self.t0
        self._lbl_idx.setText(f"{idx + 1}/{len(self.packets)}")
        self._lbl_time.setText(f"t={t_rel:.3f} s   raw={pkt.kind}")
        # Move the play head on the timeline strip.
        self.button_strip.set_index(idx)

        if pkt.kind in ("LEFT", "RIGHT") and pkt.pos is not None:
            R = quat_to_rotmat(pkt.quat)
            self._last_pose[pkt.kind] = (pkt.pos, R)
            self._last_packet[pkt.kind] = pkt
            if pkt.kind == "LEFT":
                self._left_tripod.set_pose(pkt.pos, R)
                self._left_cube.set_pose(pkt.pos, R)
                if not pkt.synthetic:
                    self._left_trail.append(pkt.pos)
            else:
                self._right_tripod.set_pose(pkt.pos, R)
                self._right_cube.set_pose(pkt.pos, R)
                if not pkt.synthetic:
                    self._right_trail.append(pkt.pos)

        self._update_readouts()

    def _update_readouts(self) -> None:
        # Drive the per-controller button pads from the most recent packet
        # of each kind. If no packet of that kind has been seen yet, show
        # an idle pad (all zeros).
        for kind, pad in (("LEFT", self.pad_left),
                          ("RIGHT", self.pad_right)):
            pkt = self._last_packet.get(kind)
            if pkt is None:
                pad.set_state(0, 0, 0, 0, 0.0, 0.0)
            else:
                pad.set_state(pkt.a, pkt.b, pkt.x, pkt.y,
                              pkt.trigger, pkt.grip)
        for kind, lbl in (("LEFT", self.label_left),
                          ("RIGHT", self.label_right)):
            pkt = self._last_packet.get(kind)
            if pkt is None or pkt.pos is None:
                lbl.setText(f"{kind:5s}: (no data yet)")
                continue
            tag = " [SYNTH]" if pkt.synthetic else ""
            buttons = []
            for name, val in (("A", pkt.a), ("B", pkt.b),
                              ("X", pkt.x), ("Y", pkt.y)):
                if val:
                    buttons.append(name)
            btn_str = "".join(buttons) if buttons else "—"
            lbl.setText(
                f"{kind:5s}: pos=({pkt.pos[0]:+7.3f},{pkt.pos[1]:+7.3f},"
                f"{pkt.pos[2]:+7.3f}) m{tag}\n"
                f"       quat=({pkt.quat[0]:+6.3f},{pkt.quat[1]:+6.3f},"
                f"{pkt.quat[2]:+6.3f},{pkt.quat[3]:+6.3f})\n"
                f"       trig={pkt.trigger:.2f}  grip={pkt.grip:.2f}  "
                f"btn={btn_str}  rate={pkt.rate:.2f}"
            )
        # Meta line: source, log time, packet kind summary
        idx = self._index
        pkt = self.packets[idx]
        synth_count = sum(1 for p in self.packets if p.synthetic)
        self.label_meta.setText(
            f"\nfile : {self.log_path.name}\n"
            f"src  : {pkt.source}\n"
            f"total: {len(self.packets)} pkts, {synth_count} synthetic"
        )

    # ─────────────────────────────────────────────────────────────────
    # Filter checkbox slots
    # ─────────────────────────────────────────────────────────────────
    def _on_show_left(self, checked: bool) -> None:
        self._left_tripod.set_visible(checked)
        self._left_cube.set_visible(checked)
        self._left_trail.set_visible(checked and self.cb_show_trail.isChecked())

    def _on_show_right(self, checked: bool) -> None:
        self._right_tripod.set_visible(checked)
        self._right_cube.set_visible(checked)
        self._right_trail.set_visible(checked and self.cb_show_trail.isChecked())

    def _on_show_trail(self, checked: bool) -> None:
        self._left_trail.set_visible(checked and self.cb_show_left.isChecked())
        self._right_trail.set_visible(checked and self.cb_show_right.isChecked())

    # ─────────────────────────────────────────────────────────────────
    # File loading
    # ─────────────────────────────────────────────────────────────────
    def _on_load_file(self) -> None:
        start_dir = str(_default_log_dir())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open VR log", start_dir, "VR log (*.jsonl);;All files (*)"
        )
        if not path:
            return
        try:
            packets = parse_log(Path(path))
        except Exception as e:
            logger.exception("Failed to load log")
            self.status.showMessage(f"Load failed: {e}", 5000)
            return
        if not packets:
            self.status.showMessage("File contained no packets.", 5000)
            return
        # Replace state in place rather than spawning a new window — the
        # user wanted "load file" to feel like switching tracks.
        self.log_path = Path(path)
        self.packets = packets
        self.t0 = packets[0].t
        self.duration = packets[-1].t - self.t0
        self.setWindowTitle(
            f"VR log viewer — {self.log_path.name}  "
            f"({len(packets)} packets, {self.duration:.1f} s)"
        )
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(packets) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._left_trail.reset()
        self._right_trail.reset()
        self._last_packet.clear()
        self.button_strip.set_packets(packets)
        self._set_index(0)


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────
def _default_log_dir() -> Path:
    return Path.home() / ".openarm_ui_config" / "vr_recordings"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if argv is None:
        argv = sys.argv

    log_path: Path | None = None
    if len(argv) > 1:
        log_path = Path(argv[1]).expanduser()
        if not log_path.exists():
            logger.error(f"file not found: {log_path}")
            return 2

    app = QApplication(argv)

    if log_path is None:
        start_dir = str(_default_log_dir())
        chosen, _ = QFileDialog.getOpenFileName(
            None, "Open VR log", start_dir,
            "VR log (*.jsonl);;All files (*)"
        )
        if not chosen:
            logger.info("no file selected, exiting")
            return 0
        log_path = Path(chosen)

    packets = parse_log(log_path)
    if not packets:
        logger.error(f"no packets parsed from {log_path}")
        return 2

    win = VRLogViewer(packets, log_path)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
