"""Per-tick-per-arm cartesian decision log for the motion worker.

Every tick that an arm is in cartesian mode, the worker writes one row
capturing the full chain: what target pose came in, what physical q
was, what IK produced, and what the half-step / dead-zone / lead-cap
emitted. The point is offline trembling diagnosis — a CSV that you can
load with pandas and ask things like "show me ticks where commanded
moved >0.1° on j3 while target_pose was identical to last tick."

Companion to ``motion_perf_log.py``. They live side by side because
they answer different questions:
  - perf_log → "are we sustaining 30 Hz, where's the time going?"
  - cart_log → "given the inputs, did the math do something sensible?"

Format: CSV. ~50 columns; ~400 bytes per row at the current shape.
At 30 Hz with both arms in cartesian mode that's ~24 kB/s, ~1.4 MB
per minute. Manageable on disk; users running long sessions can
delete old files freely (each session writes a fresh timestamped file).

Thread model: one writer (motion worker tick loop), one CSV file open
for the session lifetime. No flush per row — rely on file buffer.
``close()`` flushes on exit. The perf_log already gets a periodic
flush every summary interval; we piggyback on the same cadence by
calling ``flush()`` from the worker right after the perf summary.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Optional, Sequence

from . import config

logger = logging.getLogger(__name__)


# Column groups. Kept as separate tuples so it's obvious what each one
# captures and the header order is stable across releases.
_BASE_COLS = (
    "t_rel_s",
    "tick",
    "arm",
    "mode",        # "vr" / "vr_abs" / "manual" — provenance for diagnosis
    "fresh",       # 1 if the drive-gate was set (target was just written)
    "gated_hold", # 1 if we returned early at the freshness gate
    "dead_zone",  # 1 if the joint-delta dead-zone collapsed cmd to physical
)

_TARGET_COLS = (
    "target_x", "target_y", "target_z",
    "target_roll", "target_pitch", "target_yaw",
    "target_gripper",
)

# Per-joint columns are repeated three times — physical, IK output,
# commanded — so a pandas user can do `df[ik_q1] - df[phys_q1]` etc.
_PHYS_COLS = tuple(f"phys_q{i+1}" for i in range(7))
_IK_COLS = tuple(f"ik_q{i+1}" for i in range(7))
_CMD_COLS = tuple(f"cmd_q{i+1}" for i in range(7))

_IK_STATUS_COLS = (
    "ik_usable",
    "ik_converged",
    "ik_clamped",                # joint-limit clamp
    "ik_position_priority",      # used pass-2 (relaxed orientation)
    "ik_boundary_clamped",       # used pass-3 (workspace-edge fallback)
    "ik_iters",
    "ik_pos_err_mm",
    "ik_rot_err_deg",
)

_HEADER = (
    *_BASE_COLS,
    *_TARGET_COLS,
    *_PHYS_COLS,
    *_IK_COLS,
    *_CMD_COLS,
    *_IK_STATUS_COLS,
    # Largest |cmd[j] − phys[j]| over the 7 arm joints — quick scan
    # column. Pandas: filter df[df.cmd_max_step > 1.0] to find ticks
    # that demanded a big motor jump.
    "cmd_max_step_deg",
    # Largest |ik[j] − phys[j]| — same idea but pre-half-step. The
    # ratio (cmd_max_step / ik_max_step) should approximate alpha
    # except where the per-tick step cap kicked in.
    "ik_max_step_deg",
)


class CartesianTickLogger:
    """Per-tick-per-arm cartesian decision recorder.

    Public surface mirrors MotionPerfLogger:
      - ``open()`` / ``close()`` — bracket the session
      - ``write_row(...)`` — emit one (tick, arm) row
      - ``flush()`` — push to disk; called by the worker on the same
        cadence as the perf summary
      - ``path`` — for the UI/log to display where data is going

    Cheap when called: one csv.writerow + dict-style append. The
    per-row formatting cost is dominated by 21 floats * %.4f, which
    benchmarks at ~25 µs per row on the GB10 — well under our tick
    budget.
    """

    def __init__(self, log_dir: Path = config.CART_LOG_DIR) -> None:
        self._log_dir = Path(log_dir)
        self._file: Optional[IO[str]] = None
        self._writer: Optional[csv.writer] = None
        self._t0: float = 0.0
        self._tick_index: int = 0
        self._path: Optional[Path] = None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Create the log directory and open a fresh CSV. Filename is
        timestamped at session start so consecutive runs don't clobber
        each other; old files accumulate in CART_LOG_DIR until the
        operator deletes them.
        """
        self._log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self._log_dir / f"cart_log_{ts}.csv"
        self._file = self._path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_HEADER)
        self._t0 = time.perf_counter()
        logger.info(f"cart log opened: {self._path}")

    def close(self) -> None:
        """Flush and close. Idempotent."""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception as e:
                logger.warning(f"cart log close failed: {e}")
            self._file = None
            self._writer = None
            logger.info(f"cart log closed: {self._path}")

    def flush(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            except Exception:
                pass

    def increment_tick(self) -> None:
        """Bump the tick counter once per worker tick. Called by the
        worker after both arms have been processed (so left and right
        rows for the same physical tick share the same tick id).
        """
        self._tick_index += 1

    # ------------------------------------------------------------------
    # Per-row API
    # ------------------------------------------------------------------
    def write_row(
        self,
        *,
        arm: str,
        mode: str,
        fresh: bool,
        gated_hold: bool,
        dead_zone: bool,
        target: Optional[object],          # CartesianTarget or None
        phys_q: Sequence[float],            # 7 elements
        ik_q: Optional[Sequence[float]],    # 7 elements or None when gated
        cmd_q: Sequence[float],             # 7 elements
        ik_result: Optional[object] = None,
    ) -> None:
        """Emit one row.

        ``ik_q`` and ``ik_result`` are None when the freshness gate
        skipped IK entirely. We still write a row in that case so the
        time series has no gaps — the diagnostic CSV's whole point is
        being able to spot "we held still for 200 ticks, then suddenly
        emitted commands" at a glance.
        """
        if self._writer is None:
            return
        now = time.perf_counter()
        t_rel_s = now - self._t0

        if target is None:
            t_vals = ("", "", "", "", "", "", "")
        else:
            t_vals = (
                f"{getattr(target, 'x', 0.0):.6f}",
                f"{getattr(target, 'y', 0.0):.6f}",
                f"{getattr(target, 'z', 0.0):.6f}",
                f"{getattr(target, 'roll', 0.0):.6f}",
                f"{getattr(target, 'pitch', 0.0):.6f}",
                f"{getattr(target, 'yaw', 0.0):.6f}",
                f"{(getattr(target, 'gripper', None) or 0.0):.4f}",
            )

        phys_vals = tuple(f"{float(v):.4f}" for v in phys_q)
        if ik_q is None:
            ik_vals = ("", "", "", "", "", "", "")
            ik_max_step = ""
        else:
            ik_vals = tuple(f"{float(v):.4f}" for v in ik_q)
            ik_max_step = f"{max(abs(float(ik_q[i]) - float(phys_q[i])) for i in range(7)):.4f}"
        cmd_vals = tuple(f"{float(v):.4f}" for v in cmd_q)
        cmd_max_step = max(
            abs(float(cmd_q[i]) - float(phys_q[i])) for i in range(7)
        )

        if ik_result is None:
            ik_status = ("", "", "", "", "", "", "", "")
        else:
            ik_status = (
                int(bool(getattr(ik_result, "usable", False))),
                int(bool(getattr(ik_result, "converged", False))),
                int(bool(getattr(ik_result, "clamped", False))),
                int(bool(getattr(ik_result, "position_priority_used", False))),
                int(bool(getattr(ik_result, "boundary_clamped", False))),
                int(getattr(ik_result, "iters", 0)),
                f"{float(getattr(ik_result, 'pos_err_mm', 0.0)):.3f}",
                f"{float(getattr(ik_result, 'rot_err_deg', 0.0)):.3f}",
            )

        row = (
            f"{t_rel_s:.4f}",
            self._tick_index,
            arm,
            mode,
            int(bool(fresh)),
            int(bool(gated_hold)),
            int(bool(dead_zone)),
            *t_vals,
            *phys_vals,
            *ik_vals,
            *cmd_vals,
            *ik_status,
            f"{cmd_max_step:.4f}",
            ik_max_step,
        )
        try:
            self._writer.writerow(row)
        except Exception as e:
            logger.warning(f"cart log write failed: {e}")
