"""Per-tick performance log for the motion worker.

Captures stage-level timings (state read, VR ticks, IK / cartesian ticks,
action build, MIT send) and writes them to a CSV file under
``~/.openarm_ui_config/perf_logs/`` plus periodic summary lines on the
standard logger.

Why a dedicated file rather than INFO logs per tick: at 30 Hz with two
arms a per-tick log line would dominate the standard log. Keeping the
raw rows in a CSV (cheap to grep / load with pandas) and only emitting
roll-ups on the standard logger gives both fidelity and signal.

Thread model: the motion worker is single-threaded for the tick loop,
so the logger doesn't need its own lock — the worker owns the
``MotionPerfLogger`` instance and is the only writer.
"""

from __future__ import annotations

import csv
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Optional

from . import config

logger = logging.getLogger(__name__)


# Stage names recorded per tick. Keep the order consistent so CSV
# columns are stable across sessions.
_STAGES: tuple[str, ...] = (
    "drain_cmd",
    "read_state",
    "vr_left",
    "vr_right",
    "cart_left",
    "cart_right",
    "action_build",
    "send",
)

_CSV_HEADER: tuple[str, ...] = (
    "t_rel_s",
    "tick",
    *(f"{s}_ms" for s in _STAGES),
    "total_ms",
    "vr_left_on",
    "vr_right_on",
    "cart_left_on",
    "cart_right_on",
    # Counts of joints whose setpoint hit the lead cap this tick (per
    # arm, 0..8 each). Non-zero means the trajectory was paused
    # because the motor couldn't keep up with the commanded speed.
    "lead_cap_left",
    "lead_cap_right",
)


class MotionPerfLogger:
    """Per-tick stage timing recorder.

    Usage in the worker tick:
        logger.tick_begin()
        ... work ...
        logger.stage("read_state", ms)
        ... work ...
        logger.tick_end(flags=...)
    """

    def __init__(
        self,
        log_dir: Path = config.PERF_LOG_DIR,
        summary_interval_sec: float = config.PERF_LOG_SUMMARY_INTERVAL_SEC,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._summary_interval_sec = float(summary_interval_sec)

        self._t0: float = 0.0
        self._tick_index: int = 0
        self._tick_t0: float = 0.0
        self._stage_ms: dict[str, float] = {}

        self._file: Optional[IO[str]] = None
        self._writer: Optional[csv.writer] = None

        # Rolling window for the periodic summary. One entry per tick;
        # capped at MOTION_HZ * summary_interval to bound memory.
        self._window_total_ms: list[float] = []
        self._window_stages: dict[str, list[float]] = {s: [] for s in _STAGES}
        self._next_summary_at: float = 0.0

        self._path: Optional[Path] = None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Create the log directory and open a fresh CSV. Call once at the
        top of the worker's run() before the tick loop.
        """
        self._log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self._log_dir / f"motion_perf_{ts}.csv"
        self._file = self._path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_CSV_HEADER)
        self._t0 = time.perf_counter()
        self._next_summary_at = self._t0 + self._summary_interval_sec
        logger.info(f"motion perf log opened: {self._path}")

    def close(self) -> None:
        """Flush and close the log. Idempotent."""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception as e:
                logger.warning(f"motion perf log close failed: {e}")
            self._file = None
            self._writer = None
            logger.info(f"motion perf log closed: {self._path}")

    # ------------------------------------------------------------------
    # Per-tick API
    # ------------------------------------------------------------------
    def tick_begin(self) -> None:
        self._tick_t0 = time.perf_counter()
        self._stage_ms.clear()

    def stage(self, name: str, elapsed_ms: float) -> None:
        """Record one stage's elapsed milliseconds. Names not in _STAGES
        are silently dropped — the CSV schema has fixed columns.
        """
        if name in self._stage_ms:
            # Multiple stage(name, ...) calls within a single tick add up
            # (e.g. cartesian_left + cartesian_right would each go to a
            # different bucket — this branch is just defensive).
            self._stage_ms[name] += float(elapsed_ms)
        else:
            self._stage_ms[name] = float(elapsed_ms)

    def tick_end(
        self,
        *,
        vr_left_on: bool = False,
        vr_right_on: bool = False,
        cart_left_on: bool = False,
        cart_right_on: bool = False,
        lead_cap_left: int = 0,
        lead_cap_right: int = 0,
    ) -> None:
        """Close out the current tick: write a CSV row and roll into the
        summary window. Cheap (no fsync, no flush per row).
        """
        if self._writer is None:
            return
        now = time.perf_counter()
        total_ms = (now - self._tick_t0) * 1000.0
        t_rel_s = now - self._t0

        row = [
            f"{t_rel_s:.4f}",
            self._tick_index,
            *(f"{self._stage_ms.get(s, 0.0):.3f}" for s in _STAGES),
            f"{total_ms:.3f}",
            int(vr_left_on),
            int(vr_right_on),
            int(cart_left_on),
            int(cart_right_on),
            int(lead_cap_left),
            int(lead_cap_right),
        ]
        try:
            self._writer.writerow(row)
        except Exception as e:
            logger.warning(f"motion perf log write failed: {e}")

        self._tick_index += 1
        self._window_total_ms.append(total_ms)
        for s in _STAGES:
            self._window_stages[s].append(self._stage_ms.get(s, 0.0))

        if (self._summary_interval_sec > 0.0
                and now >= self._next_summary_at):
            self._emit_summary()
            self._next_summary_at = now + self._summary_interval_sec
            # Flush so the file on disk reflects what was summarised.
            try:
                if self._file is not None:
                    self._file.flush()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _emit_summary(self) -> None:
        n = len(self._window_total_ms)
        if n == 0:
            return

        def stats(vals: list[float]) -> str:
            med = statistics.median(vals)
            p95 = _percentile(vals, 95)
            mx = max(vals)
            return f"{med:5.1f}/{p95:5.1f}/{mx:5.1f}"

        parts = [f"perf n={n}", f"total={stats(self._window_total_ms)}"]
        for s in _STAGES:
            parts.append(f"{s}={stats(self._window_stages[s])}")

        logger.info(" ".join(parts) + "  (med/p95/max ms)")

        # Reset the window for the next interval.
        self._window_total_ms.clear()
        for s in _STAGES:
            self._window_stages[s].clear()


def _percentile(values: list[float], pct: float) -> float:
    """Cheap nearest-rank percentile. Sufficient for ~150-tick windows."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[k]
