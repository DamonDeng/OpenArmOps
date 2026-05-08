"""Shared mutable state between tabs.

Just a plain object with attributes — no Qt signals yet because the Controller
tab polls this on every tick, so it naturally picks up new values. If we add
more writers, promote to QObject + pyqtSignal.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class RuntimeState:
    max_speed_deg_per_sec: float = config.INITIAL_MAX_SPEED_DEG_PER_SEC
    gravity_comp_scale: float = config.INITIAL_GRAVITY_COMP_SCALE

    def max_step_per_tick(self, poll_hz: int = config.POLL_HZ) -> float:
        """Max degrees the commanded value can move per poll tick."""
        return self.max_speed_deg_per_sec / float(poll_hz)
