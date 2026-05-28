"""UDP receiver for Pico 4 Ultra controller + head poses.

Protocol (wire format verified via tcpdump on 2026-05-11 against the
openarmx_teleop_vr_apk v6 APK; coordinate frame characterized via the
2026-05-27 three-posture recording):

    Each datagram is one ASCII line, space-delimited, ending in newline
    optional. First token identifies the payload:

      LEFT  tx ty tz  qx qy qz qw  trigger grip  A B X Y  rate  ts_ns
      RIGHT tx ty tz  qx qy qz qw  trigger grip  A B X Y  rate  ts_ns
      HEAD  tx ty tz  qx qy qz qw  ts_ns
      MODE  relative|absolute
      CALIBRATE_DONE

    Translation metres, quaternion xyzw order, trigger/grip 0..1,
    A/B/X/Y 0 or 1. Rate is the on-controller slider 0..1.

    Coordinate frame (right-handed, as the APK actually sends — this
    is NOT the documented OpenXR "+Y up, +Z forward" convention):
      +X = operator's right
      +Y = forward (out from the operator's body)
      +Z = up
    Origin resets at each grip rising-edge. The vr→robot remap in
    config.VR_TRANSLATION_REMAP_RIGHT depends on this.

Design:
- One QThread owns a non-blocking UDP socket bound to VR_UDP_PORT. Polls
  with select() so shutdown is responsive.
- Parsed state is kept in thread-safe attributes (Python dict get/set is
  atomic enough for our single-writer-many-reader pattern; we wrap in a
  lock for safety anyway).
- UI is notified via a Qt signal at a throttled rate (10 Hz) — no
  point re-rendering faster than the eye can follow, and the incoming
  stream is ~100 Hz.
- Stats (packet count, rate, last-packet-age) tracked internally.

Phase 2a: no motor integration. The motion worker will sample the state
directly in Phase 2b.
"""

from __future__ import annotations

import json
import logging
import select
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from . import config

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    """Latest known state for one controller. All fields use wire-format
    units: metres, xyzw quaternion, 0..1 analog values, 0/1 button flags.
    """
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    trigger: float = 0.0
    grip: float = 0.0
    a: int = 0
    b: int = 0
    x: int = 0
    y: int = 0
    rate: float = 0.0
    ts_ns: int = 0
    last_rx: float = 0.0  # our monotonic receive time (seconds)

    @property
    def has_ever_been_seen(self) -> bool:
        return self.last_rx > 0.0


@dataclass
class HeadState:
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    ts_ns: int = 0
    last_rx: float = 0.0

    @property
    def has_ever_been_seen(self) -> bool:
        return self.last_rx > 0.0


@dataclass
class StreamStats:
    total_packets: int = 0
    total_bytes: int = 0
    unknown_messages: int = 0
    parse_errors: int = 0
    last_rx: float = 0.0
    last_source: str = ""      # "192.168.3.94:35841"
    # Rolling 1-second packet-count window for rate calc.
    _window: deque = field(default_factory=lambda: deque(maxlen=256))

    def record(self, now: float, size_bytes: int, source: str) -> None:
        self.total_packets += 1
        self.total_bytes += size_bytes
        self.last_rx = now
        self.last_source = source
        self._window.append(now)

    def rate_hz(self, now: float) -> float:
        # Count packets in the last 1 s window.
        cutoff = now - 1.0
        # Trim from the left while items are older than 1s.
        w = self._window
        while w and w[0] < cutoff:
            w.popleft()
        return float(len(w))


class VRInputReceiver(QThread):
    """Listens for APK datagrams and publishes state via a Qt signal.

    The signal fires at ~10 Hz with a snapshot dict so the UI can render
    current values. The signal is emitted even when no packets have
    arrived, so the tab can show "no data" / "stream stale" states
    consistently.
    """

    state_updated = pyqtSignal(dict)   # {"left": ControllerState, ...}

    def __init__(self) -> None:
        super().__init__()
        self._stop = False
        self._lock = threading.Lock()

        # Source-of-truth state (mutated by the receiver thread,
        # read by the emit_loop and anyone sampling for motor control).
        self._left = ControllerState()
        self._right = ControllerState()
        self._head = HeadState()
        self._mode: str = ""            # "relative" / "absolute" / ""
        self._last_calibrate_done: float = 0.0
        self._stats = StreamStats()

        # UI emit throttle: 10 Hz — fast enough to look live, slow
        # enough not to starve the Qt event loop.
        self._emit_interval = 0.1
        self._last_emit = 0.0

        # Manual packet recorder. Off by default at startup — the user
        # has to flip a toggle in the System tab to enable it. When
        # enabled, every received datagram is appended to an in-memory
        # buffer along with its arrival timestamp and source address.
        # The buffer is bounded only by available memory; user is
        # expected to "Save & clear" when finished. We keep the buffer
        # behind its own lock so the UI's stats poll doesn't fight
        # the receiver thread for self._lock.
        self._record_enabled = False
        self._record_lock = threading.Lock()
        self._record_buffer: deque[tuple[float, str, bytes]] = deque()
        self._record_bytes = 0  # cached size sum, kept in sync on append/clear

    # ------------------------------------------------------------------
    # Public read API — any thread can call these (they snapshot under
    # the lock to avoid torn reads of multi-field dataclasses).
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "left": self._left,
                "right": self._right,
                "head": self._head,
                "mode": self._mode,
                "last_calibrate_done": self._last_calibrate_done,
                "stats": self._stats,
            }

    def left(self) -> ControllerState:
        with self._lock:
            return self._left

    def right(self) -> ControllerState:
        with self._lock:
            return self._right

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------
    # Packet recorder — manual on/off, in-memory buffer, save-and-clear.
    # ------------------------------------------------------------------
    def set_recording(self, enabled: bool) -> None:
        """Turn raw-packet recording on or off. Toggling does not touch
        the existing buffer — turning off then on later resumes
        appending to the same deque. Use ``save_and_clear`` to flush.
        """
        with self._record_lock:
            self._record_enabled = bool(enabled)
        logger.info(f"VR packet recording {'ENABLED' if enabled else 'DISABLED'}")

    def recording_stats(self) -> tuple[bool, int, int]:
        """Return ``(enabled, packet_count, total_bytes)`` for live UI
        display. Cheap — just snapshots three counters under the lock.
        """
        with self._record_lock:
            return self._record_enabled, len(self._record_buffer), self._record_bytes

    def save_and_clear(self, path: Path) -> tuple[int, int]:
        """Write the current buffer to ``path`` as JSON Lines, then drop
        the in-memory buffer. Returns ``(packets_written, bytes_written)``.

        Atomic swap pattern: under the lock we replace the buffer with
        a fresh deque so the receiver thread can keep appending while
        we serialize the snapshot to disk on the calling thread.

        Format: one JSON object per line, fields:
          - ``t``: monotonic seconds since the receiver started (float)
          - ``from``: ``"ip:port"`` source address string
          - ``raw``: the raw datagram body, decoded as ASCII with
                     errors=replace (the wire format is space-delimited
                     ASCII, so this is human-readable and round-trips
                     for replay).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._record_lock:
            snapshot = self._record_buffer
            self._record_buffer = deque()
            self._record_bytes = 0

        packets = len(snapshot)
        if packets == 0:
            path.write_text("", encoding="utf-8")
            return 0, 0

        bytes_written = 0
        with path.open("w", encoding="utf-8") as f:
            for t, source, raw in snapshot:
                line = json.dumps({
                    "t": round(t, 6),
                    "from": source,
                    "raw": raw.decode("ascii", errors="replace"),
                }) + "\n"
                f.write(line)
                bytes_written += len(line.encode("utf-8"))
        return packets, bytes_written

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((config.VR_UDP_BIND_ADDR, config.VR_UDP_PORT))
        except OSError as e:
            logger.error(
                f"VR UDP bind failed on {config.VR_UDP_BIND_ADDR}:"
                f"{config.VR_UDP_PORT}: {e}. Receiver disabled."
            )
            return
        sock.setblocking(False)
        logger.info(
            f"VR receiver listening on udp://{config.VR_UDP_BIND_ADDR}:"
            f"{config.VR_UDP_PORT}"
        )

        try:
            while not self._stop:
                # select() with a short timeout keeps shutdown responsive
                # while letting us service bursts without CPU-spinning.
                ready, _, _ = select.select([sock], [], [], 0.05)
                if ready:
                    try:
                        # Drain everything the kernel has for us, to
                        # avoid falling behind under bursts. We keep the
                        # latest per-type anyway, so dropping older
                        # duplicates is harmless.
                        while True:
                            try:
                                data, addr = sock.recvfrom(2048)
                            except BlockingIOError:
                                break
                            self._handle_packet(data, addr)
                    except Exception as e:
                        logger.exception(f"VR recv error: {e}")

                self._maybe_emit()
        finally:
            sock.close()
            logger.info("VR receiver stopped")

    # ------------------------------------------------------------------
    # Packet handling
    # ------------------------------------------------------------------
    def _handle_packet(self, data: bytes, addr: tuple[str, int]) -> None:
        now = time.monotonic()
        size = len(data)
        source = f"{addr[0]}:{addr[1]}"

        # Recorder hook: append the raw datagram (pre-parse) so the log
        # captures every packet we received, even malformed ones. Cheap
        # branch when disabled.
        if self._record_enabled:
            with self._record_lock:
                # Re-check under the lock — set_recording could have
                # flipped between the outer test and here.
                if self._record_enabled:
                    self._record_buffer.append((now, source, data))
                    self._record_bytes += size

        try:
            line = data.decode("ascii", errors="replace").strip()
        except Exception:
            with self._lock:
                self._stats.parse_errors += 1
                self._stats.record(now, size, source)
            return

        with self._lock:
            self._stats.record(now, size, source)

        tokens = line.split()
        if not tokens:
            return

        kind = tokens[0].upper()
        rest = tokens[1:]

        try:
            if kind == "LEFT":
                self._apply_controller(self._left, rest, now)
            elif kind == "RIGHT":
                self._apply_controller(self._right, rest, now)
            elif kind == "HEAD":
                self._apply_head(rest, now)
            elif kind == "MODE":
                with self._lock:
                    self._mode = rest[0] if rest else ""
            elif kind == "CALIBRATE_DONE":
                with self._lock:
                    self._last_calibrate_done = now
            else:
                with self._lock:
                    self._stats.unknown_messages += 1
        except Exception as e:
            with self._lock:
                self._stats.parse_errors += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"parse error on {kind!r}: {e}; line={line!r}")

    def _apply_controller(
        self,
        target: ControllerState,
        tokens: list[str],
        now: float,
    ) -> None:
        # Verified wire format (15 fields after kind):
        #   tx ty tz qx qy qz qw trigger grip A B X Y rate ts_ns
        if len(tokens) < 15:
            raise ValueError(f"controller datagram too short: {len(tokens)} fields")
        with self._lock:
            target.tx = float(tokens[0])
            target.ty = float(tokens[1])
            target.tz = float(tokens[2])
            target.qx = float(tokens[3])
            target.qy = float(tokens[4])
            target.qz = float(tokens[5])
            target.qw = float(tokens[6])
            target.trigger = float(tokens[7])
            target.grip = float(tokens[8])
            target.a = int(float(tokens[9]))
            target.b = int(float(tokens[10]))
            target.x = int(float(tokens[11]))
            target.y = int(float(tokens[12]))
            target.rate = float(tokens[13])
            target.ts_ns = int(float(tokens[14]))
            target.last_rx = now

    def _apply_head(self, tokens: list[str], now: float) -> None:
        # Head datagram: tx ty tz qx qy qz qw ts_ns (8 fields)
        if len(tokens) < 8:
            raise ValueError(f"head datagram too short: {len(tokens)} fields")
        with self._lock:
            self._head.tx = float(tokens[0])
            self._head.ty = float(tokens[1])
            self._head.tz = float(tokens[2])
            self._head.qx = float(tokens[3])
            self._head.qy = float(tokens[4])
            self._head.qz = float(tokens[5])
            self._head.qw = float(tokens[6])
            self._head.ts_ns = int(float(tokens[7]))
            self._head.last_rx = now

    # ------------------------------------------------------------------
    # UI emit
    # ------------------------------------------------------------------
    def _maybe_emit(self) -> None:
        now = time.monotonic()
        if now - self._last_emit < self._emit_interval:
            return
        self._last_emit = now
        # Emit a shallow snapshot. dataclass instances are sent by
        # reference across the Qt signal — the UI copies out fields it
        # needs; no cross-thread mutation concerns since the UI only reads.
        self.state_updated.emit(self.snapshot())
