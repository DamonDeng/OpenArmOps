"""30 Hz motion control worker thread.

Owns the per-joint trajectories and all motor IO. The UI thread never touches
trajectories directly — it posts commands into ``self.command_queue``. At the
top of each tick the worker drains the queue, reads motor state, advances each
trajectory, and sends one MIT command batch per arm.

Design decisions (see chat log for reasoning):

- Fixed 30 Hz rate controlled with ``time.perf_counter()`` + ``time.sleep()``,
  independent of Qt event loop timing.
- Each joint owns a ``JointTrajectory`` with start/target/elapsed/total/step.
  Setpoint grows linearly from start to target at a time-based rate. The motor
  tracks via MIT position control; if it lags, the growing setpoint increases
  the error term, which grows torque naturally.
- Lead cap: setpoint is clamped to within ±LEAD_CAP of the motor's observed
  current. Pauses the trajectory's time when the motor can't keep up. Bounds
  the MIT error term and prevents runaway.
- Torque-OFF arms: their joint trajectories are continuously reset to
  ``start=target=current`` so re-enabling torque won't cause a lurch.
- State is published to the UI via Qt signals (``state_updated``), which are
  thread-safe in Qt.
"""

from __future__ import annotations

import logging
import math
import queue
import time
from dataclasses import dataclass, field
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from . import config
from .robot_service import RobotService
from .runtime_state import RuntimeState

logger = logging.getLogger(__name__)


@dataclass
class JointTrajectory:
    """Linear trajectory in joint space, time-indexed in worker ticks."""
    start_deg: float
    target_deg: float
    total_steps: int
    elapsed_steps: int = 0
    deg_per_tick: float = 0.0

    @classmethod
    def new(cls, start: float, target: float, deg_per_sec: float, hz: int) -> "JointTrajectory":
        dist = abs(target - start)
        if dist < 1e-4 or deg_per_sec <= 0:
            # Already there (or degenerate speed): a one-tick trajectory.
            # Setpoint snaps to target on the very next tick.
            return cls(start_deg=start, target_deg=target, total_steps=1,
                       elapsed_steps=0, deg_per_tick=0.0)
        total = max(1, int(math.ceil(dist * hz / deg_per_sec)))
        sign = 1.0 if target > start else -1.0
        return cls(
            start_deg=start,
            target_deg=target,
            total_steps=total,
            elapsed_steps=0,
            deg_per_tick=sign * deg_per_sec / hz,
        )

    def setpoint(self) -> float:
        """Ideal setpoint for the tick we're about to execute."""
        if self.elapsed_steps + 1 >= self.total_steps:
            return self.target_deg
        return self.start_deg + (self.elapsed_steps + 1) * self.deg_per_tick

    def advance(self) -> None:
        if self.elapsed_steps < self.total_steps:
            self.elapsed_steps += 1

    def is_done(self) -> bool:
        return self.elapsed_steps >= self.total_steps


@dataclass
class _Command:
    """Message posted from UI thread to the worker. Minimal; keep serializable."""
    kind: str               # "set_target" | "torque" | "estop" | "stop"
    arm: Optional[str] = None
    joint: Optional[str] = None
    target_deg: Optional[float] = None
    torque_enabled: Optional[bool] = None
    # Per-call speed override for set_target. When None, the worker uses
    # runtime.max_speed_deg_per_sec (the System-tab setting). The override
    # only affects the trajectory built for this one command — later
    # commands fall back to the runtime setting unless they override too.
    deg_per_sec: Optional[float] = None


class MotionWorker(QThread):
    """Motion control loop running in a dedicated QThread.

    Signals:
        state_updated(dict): emits {joint_key: current_deg, ...} every tick
        send_error(str): emitted if a send_action fails
    """

    state_updated = pyqtSignal(dict)      # {"right_joint_1.pos": 12.3, ...}
    send_error = pyqtSignal(str)

    def __init__(self, robot: RobotService, runtime: RuntimeState) -> None:
        super().__init__()
        self.robot = robot
        self.runtime = runtime
        self.command_queue: queue.Queue[_Command] = queue.Queue()
        self._stop_flag = False

        # All 16 trajectories, keyed by "{arm}_{joint}.pos" — same shape the
        # robot's action/observation dicts use. Initialized in run() once we
        # have a real observation to seed from.
        self._trajectories: dict[str, JointTrajectory] = {}
        self._torque_on: dict[str, bool] = {"left": False, "right": False}
        self._last_current: dict[str, float] = {}
        self._initialized = False  # True once we have a first observation

    # ------------------------------------------------------------------
    # UI-facing API — these only post to the queue, they don't touch state.
    # ------------------------------------------------------------------
    def post_set_target(
        self,
        arm: str,
        joint: str,
        target_deg: float,
        deg_per_sec: Optional[float] = None,
    ) -> None:
        """Post a target-change command to the worker.

        If ``deg_per_sec`` is provided it overrides the runtime max speed
        for this single trajectory only (see _Command.deg_per_sec). Useful
        for buttons like "Slow go to zero" that should move at a fixed
        gentle speed regardless of the current System-tab setting.
        """
        self.command_queue.put(_Command(
            kind="set_target", arm=arm, joint=joint, target_deg=target_deg,
            deg_per_sec=deg_per_sec,
        ))

    def post_torque(self, arm: str, enabled: bool) -> None:
        self.command_queue.put(_Command(kind="torque", arm=arm, torque_enabled=enabled))

    def post_estop(self) -> None:
        self.command_queue.put(_Command(kind="estop"))

    def stop(self) -> None:
        """Signal the worker to exit after the current tick."""
        self._stop_flag = True
        self.command_queue.put(_Command(kind="stop"))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        period = 1.0 / config.MOTION_HZ
        logger.info(f"MotionWorker started at {config.MOTION_HZ} Hz (period {period*1000:.1f} ms)")

        next_tick = time.perf_counter()
        while not self._stop_flag:
            try:
                self._tick_once()
            except Exception as e:
                logger.exception("motion worker tick failed")
                self.send_error.emit(str(e))

            next_tick += period
            now = time.perf_counter()
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Missed the deadline; resync the schedule so we don't spiral
                # into an unbounded catch-up loop if one tick was very slow.
                next_tick = now

        logger.info("MotionWorker exiting")

    def _tick_once(self) -> None:
        # 1. Drain pending commands from the UI
        self._drain_commands()

        # 2. Read motor state (state-only; cameras are handled on UI thread)
        current = self._read_state()
        if current is None:
            return  # transient read error; keep ticking, try next time
        self._last_current = current

        # 3. On first real observation, seed one pass-through trajectory per joint
        if not self._initialized:
            for key, cur in current.items():
                self._trajectories[key] = JointTrajectory.new(
                    start=cur, target=cur,
                    deg_per_sec=self.runtime.max_speed_deg_per_sec,
                    hz=config.MOTION_HZ,
                )
            self._initialized = True

        # 4. Build the action dict from torque-ON joints' trajectories
        action: dict[str, float] = {}
        for key, traj in self._trajectories.items():
            arm = "right" if key.startswith("right_") else "left"
            cur = current.get(key)
            if cur is None:
                continue

            if not self._torque_on[arm]:
                # Keep trajectory pinned to current so re-enabling torque
                # won't lurch. Also overwrites any stale target from when
                # torque was last on.
                self._trajectories[key] = JointTrajectory.new(
                    start=cur, target=cur,
                    deg_per_sec=self.runtime.max_speed_deg_per_sec,
                    hz=config.MOTION_HZ,
                )
                continue

            setpoint = traj.setpoint()

            # Lead cap: if the motor is lagging far behind, pause the
            # trajectory by not advancing it this tick. The setpoint for
            # this tick is clamped so MIT error stays bounded.
            low = cur - config.LEAD_CAP_DEG
            high = cur + config.LEAD_CAP_DEG
            lagging = setpoint > high or setpoint < low
            setpoint = max(low, min(high, setpoint))

            action[key] = setpoint
            if not lagging:
                traj.advance()

        # 5. Publish state to UI (thread-safe Qt signal)
        self.state_updated.emit(dict(current))

        # 6. Send one action batch
        if action:
            try:
                self.robot.send_action(action)
            except Exception as e:
                logger.error(f"send_action failed: {e}")
                self.send_error.emit(str(e))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.command_queue.get_nowait()
            except queue.Empty:
                return

            if cmd.kind == "stop":
                self._stop_flag = True
                return
            if cmd.kind == "set_target":
                key = f"{cmd.arm}_{cmd.joint}.pos"
                cur = self._last_current.get(key)
                if cur is None:
                    # No observation yet — defer to first real tick by
                    # storing the target and letting the next tick rebuild.
                    # Rare: only hits if user drags before init.
                    cur = float(cmd.target_deg)  # best we can do
                # Per-call override wins; otherwise use the System-tab setting.
                speed = (
                    float(cmd.deg_per_sec)
                    if cmd.deg_per_sec is not None
                    else self.runtime.max_speed_deg_per_sec
                )
                self._trajectories[key] = JointTrajectory.new(
                    start=cur,
                    target=float(cmd.target_deg),
                    deg_per_sec=speed,
                    hz=config.MOTION_HZ,
                )
            elif cmd.kind == "torque":
                self.robot.set_torque(cmd.arm, bool(cmd.torque_enabled))
                self._torque_on[cmd.arm] = bool(cmd.torque_enabled)
                # If we just turned torque ON, pin trajectory to current so
                # the arm holds. Current won't be perfectly up-to-date but
                # next tick's read will correct us.
                if cmd.torque_enabled:
                    for key in list(self._trajectories.keys()):
                        if not key.startswith(f"{cmd.arm}_"):
                            continue
                        cur = self._last_current.get(key, 0.0)
                        self._trajectories[key] = JointTrajectory.new(
                            start=cur, target=cur,
                            deg_per_sec=self.runtime.max_speed_deg_per_sec,
                            hz=config.MOTION_HZ,
                        )
            elif cmd.kind == "estop":
                for arm in ("left", "right"):
                    self.robot.set_torque(arm, False)
                    self._torque_on[arm] = False
                # Reset all trajectories to no-motion at current.
                for key in list(self._trajectories.keys()):
                    cur = self._last_current.get(key, 0.0)
                    self._trajectories[key] = JointTrajectory.new(
                        start=cur, target=cur,
                        deg_per_sec=self.runtime.max_speed_deg_per_sec,
                        hz=config.MOTION_HZ,
                    )
                logger.warning("motion worker: e-stop consumed; torque off; trajectories pinned")

    def _read_state(self) -> Optional[dict[str, float]]:
        """Read motor positions only. Cameras are read by the UI thread."""
        obs = self.robot.get_observation()
        if obs is None:
            return None
        # Keep only .pos scalars. Cameras and .vel/.torque are noise here.
        return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
