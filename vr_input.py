"""VR controller receiver — backend-agnostic interface + UDP backend.

Two backend implementations live in the codebase:

1. ``UDPVRReceiver`` (this file): the legacy receiver for the closed-
   source ``openarmx-vr-pico.apk``. Listens on UDP port
   ``VR_UDP_PORT`` for ASCII space-delimited datagrams. Caps out at
   ~5 Hz aggregate in dual-arm mode due to APK-side or Wi-Fi-side
   loss; see ``docs/vr_packet_rate_investigation.md``.

2. ``PXREASDKVRReceiver`` (``vr_input_pxreasdk.py``): the new
   default. Attaches to Pico's XRoboToolkit broker
   (``RoboticsServiceProcess``) via ``libPXREARobotSDK.so`` and
   consumes ``PXREADeviceStateJson`` callbacks. Sustains 90 Hz in
   dual-arm mode. Use the factory ``make_vr_receiver()`` to
   construct the configured backend.

Both backends share the same public surface — ``ControllerState``,
``HeadState``, ``StreamStats``, the ``state_updated`` Qt signal at
10 Hz, and the ``snapshot``/``left``/``right``/``mark_consumed``/
``set_recording``/``recording_stats``/``save_and_clear`` methods —
so the rest of the app (motion worker, VR tab, controller tab, system
tab) is backend-agnostic.

UDP wire format (legacy, as the closed-source APK sent it):

    Each datagram is one ASCII line, space-delimited:

      LEFT  tx ty tz  qx qy qz qw  trigger grip  A B X Y  rate  ts_ns
      RIGHT tx ty tz  qx qy qz qw  trigger grip  A B X Y  rate  ts_ns
      HEAD  tx ty tz  qx qy qz qw  ts_ns
      MODE  relative|absolute
      CALIBRATE_DONE

    Coordinate frame (right-handed, as the APK actually sends —
    NOT the documented OpenXR "+Y up, +Z forward"):
      +X = operator's right
      +Y = forward (out from the operator's body)
      +Z = up
    Origin resets at each grip rising-edge. The vr→robot remap in
    ``config.VR_TRANSLATION_REMAP_RIGHT`` depends on this.
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
    # Latest ts_ns the motion worker has acknowledged via mark_consumed.
    # When a new packet overwrites this state and the previous ts_ns was
    # > last_consumed_ts_ns, the previous packet is being thrown away
    # before any consumer saw it — incremented in unread_overwrites.
    last_consumed_ts_ns: int = 0
    unread_overwrites: int = 0

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
    # Kernel-side UDP drops on our bind port, sampled from /proc/net/udp.
    # These are packets the kernel discarded before recvfrom got them
    # (typically SO_RCVBUF overflow). Sampled once per second by the
    # receiver thread; baselined at startup so the value is "drops since
    # this process started", not the system-wide total since boot.
    kernel_drops: int = 0
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
    """Backend-agnostic base for VR-controller receivers.

    Subclasses implement ``run()`` (the receiver loop) and call
    ``_record_raw()`` / direct mutation of ``_left`` / ``_right`` /
    ``_head`` / ``_stats`` (under ``self._lock``) plus ``_maybe_emit()``
    once per service cycle. Public read API and the recorder live here
    so all backends present an identical surface to the UI / motion
    worker.

    ``state_updated`` fires at ~10 Hz with a snapshot dict so the UI
    can render current values. It is emitted even when no packets have
    arrived, so tabs can show "no data" / "stream stale" states
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

    def mark_consumed(self, arm: str, ts_ns: int) -> None:
        """Acknowledge that the consumer has processed a packet with this
        ``ts_ns``. Used by the motion worker so that ``_apply_controller``
        can tell apart "overwrites a packet that's already been used"
        from "overwrites a packet that nobody has seen yet"; only the
        latter increments ``unread_overwrites``.
        """
        with self._lock:
            target = self._left if arm == "left" else self._right
            if ts_ns > target.last_consumed_ts_ns:
                target.last_consumed_ts_ns = ts_ns

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
    # Helpers shared by all backends.
    # ------------------------------------------------------------------
    def _record_raw_packet(self, now: float, source: str, data: bytes) -> None:
        """Append a raw payload to the recorder buffer if recording is on.
        Cheap when disabled (one boolean check). Subclasses must call
        this from their receive path with the original bytes so replay
        files are wire-faithful.
        """
        if not self._record_enabled:
            return
        with self._record_lock:
            if self._record_enabled:
                self._record_buffer.append((now, source, data))
                self._record_bytes += len(data)

    def _maybe_emit(self) -> None:
        """Throttle the ``state_updated`` Qt signal to ~10 Hz. Called by
        each backend's receive loop after handling a batch of packets.
        """
        now = time.monotonic()
        if now - self._last_emit < self._emit_interval:
            return
        self._last_emit = now
        # Emit a shallow snapshot. dataclass instances are sent by
        # reference across the Qt signal — the UI copies out fields it
        # needs; no cross-thread mutation concerns since the UI only reads.
        self.state_updated.emit(self.snapshot())

    # ------------------------------------------------------------------
    # Thread body — overridden by each concrete backend.
    # ------------------------------------------------------------------
    def run(self) -> None:
        raise NotImplementedError(
            "VRInputReceiver is abstract; use a concrete backend "
            "(UDPVRReceiver, PXREASDKVRReceiver) or the make_vr_receiver() factory."
        )


class UDPVRReceiver(VRInputReceiver):
    """Legacy UDP receiver for the closed-source openarmx-vr-pico.apk.

    Listens on ``VR_UDP_PORT`` for ASCII space-delimited datagrams
    (LEFT/RIGHT/HEAD/MODE/CALIBRATE_DONE). Capped at ~5 Hz aggregate
    in dual-arm mode by APK / Wi-Fi loss; see
    ``docs/vr_packet_rate_investigation.md``. Kept as a fallback for
    operators still running the legacy APK; the default is now
    ``PXREASDKVRReceiver``.
    """

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bump SO_RCVBUF so a brief stall in the receiver thread
            # doesn't lose packets to kernel-buffer overflow. Default is
            # ~212 KB on Linux 6.x; 1 MB holds ~5 s of 100 Hz traffic.
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            except OSError as e:
                logger.warning(f"VR UDP SO_RCVBUF bump failed: {e}")
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

        # Baseline kernel-side drops at startup so we report drops since
        # *this process* started rather than the system-wide total.
        kernel_drops_baseline = self._read_proc_udp_drops(config.VR_UDP_PORT)
        next_kernel_drops_check = time.monotonic() + 1.0

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

                now = time.monotonic()
                if now >= next_kernel_drops_check:
                    abs_drops = self._read_proc_udp_drops(config.VR_UDP_PORT)
                    if abs_drops >= 0:
                        with self._lock:
                            self._stats.kernel_drops = max(
                                0, abs_drops - kernel_drops_baseline
                            )
                    next_kernel_drops_check = now + 1.0

                self._maybe_emit()
        finally:
            sock.close()
            logger.info("VR receiver stopped")

    @staticmethod
    def _read_proc_udp_drops(port: int) -> int:
        """Return the kernel's drop counter for our UDP bind port.

        ``/proc/net/udp`` columns: ``sl  local_address  rem_address  st
        tx_q rx_q tr tm->when retrnsmt uid timeout inode ref pointer
        drops``. Local address is ``HEXIP:HEXPORT``. Returns -1 if the
        port can't be found (interface down, IPv6-only bind, etc.) or
        the file isn't readable. Cheap (one fopen + linear scan, ~few
        hundred lines on a busy host).
        """
        port_hex = f"{port:04X}"
        try:
            with open("/proc/net/udp", "r") as f:
                next(f, None)  # header
                for line in f:
                    cols = line.split()
                    if len(cols) < 13:
                        continue
                    local = cols[1]
                    if not local.endswith(":" + port_hex):
                        continue
                    return int(cols[12])
        except (OSError, ValueError):
            pass
        return -1

    # ------------------------------------------------------------------
    # Packet handling
    # ------------------------------------------------------------------
    def _handle_packet(self, data: bytes, addr: tuple[str, int]) -> None:
        now = time.monotonic()
        size = len(data)
        source = f"{addr[0]}:{addr[1]}"

        # Recorder hook: append the raw datagram (pre-parse) so the log
        # captures every packet we received, even malformed ones.
        self._record_raw_packet(now, source, data)

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
            # Latest-wins overwrite tracking: if the existing state has
            # a ts_ns the motion worker hasn't acknowledged yet, the
            # incoming packet is about to throw it away unseen.
            if target.last_rx > 0.0 and target.ts_ns > target.last_consumed_ts_ns:
                target.unread_overwrites += 1
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


# ---------------------------------------------------------------------------
# Backend factory — picks a concrete receiver based on config.
# ---------------------------------------------------------------------------
def make_vr_receiver(backend: Optional[str] = None) -> VRInputReceiver:
    """Return a configured VR receiver. ``backend`` defaults to
    ``config.VR_RECEIVER_BACKEND``. Recognized values:

    - ``"pxreasdk"`` — XRoboToolkit broker via libPXREARobotSDK.so
      (default; sustained 90 Hz, dual-arm in one frame)
    - ``"udp"``      — legacy UDP listener for the closed-source
      openarmx-vr-pico.apk

    Falls back to UDP and logs a warning if pxreasdk is requested but
    the SDK module fails to import (for instance if the service
    package isn't installed on this host).
    """
    name = (backend or getattr(config, "VR_RECEIVER_BACKEND", "pxreasdk")).lower()
    if name == "udp":
        return UDPVRReceiver()
    if name == "pxreasdk":
        try:
            from .vr_input_pxreasdk import PXREASDKVRReceiver
        except Exception as e:
            logger.error(
                f"PXREASDK backend unavailable ({e}); falling back to UDP. "
                "Install Pico's RoboticsService and ensure libPXREARobotSDK.so "
                "is on LD_LIBRARY_PATH to use the high-rate path."
            )
            return UDPVRReceiver()
        return PXREASDKVRReceiver()
    logger.error(f"Unknown VR_RECEIVER_BACKEND={name!r}; falling back to UDP.")
    return UDPVRReceiver()

