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
from .gravity_comp import GravityCompensator
from .ik_solver import (
    CartesianIKSolver,
    IKResult,
    pose_from_xyzrpy,
    pose_to_xyzrpy,
)
from .robot_service import RobotService
from .runtime_state import RuntimeState
from .vr_input import VRInputReceiver

try:
    import numpy as np
    import pinocchio as pin
except ImportError:  # pragma: no cover — pinocchio was already required by gravity_comp
    raise

logger = logging.getLogger(__name__)


@dataclass
class JointTrajectory:
    """Linear trajectory in joint space, time-indexed in worker ticks.

    The trajectory owns a wall-clock ``last_updated`` timestamp used by
    the staleness check: if we haven't touched it for STALENESS_SEC we
    assume the user stopped caring and a future set_target should
    rebuild from the motor's actual current position instead of
    extending. This prevents a very old trajectory from suddenly
    becoming active again and lurching the arm.
    """
    start_deg: float
    target_deg: float
    total_steps: int
    elapsed_steps: int = 0
    deg_per_tick: float = 0.0
    last_updated: float = 0.0   # time.perf_counter() of last touch

    @classmethod
    def new(cls, start: float, target: float, deg_per_sec: float, hz: int) -> "JointTrajectory":
        dist = abs(target - start)
        if dist < 1e-4 or deg_per_sec <= 0:
            # Already there (or degenerate speed): a one-tick trajectory.
            # Setpoint snaps to target on the very next tick.
            return cls(start_deg=start, target_deg=target, total_steps=1,
                       elapsed_steps=0, deg_per_tick=0.0,
                       last_updated=time.perf_counter())
        total = max(1, int(math.ceil(dist * hz / deg_per_sec)))
        sign = 1.0 if target > start else -1.0
        return cls(
            start_deg=start,
            target_deg=target,
            total_steps=total,
            elapsed_steps=0,
            deg_per_tick=sign * deg_per_sec / hz,
            last_updated=time.perf_counter(),
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

    # ------------------------------------------------------------------
    # Target-extension path — avoids rebuilding from motor current on
    # every set_target call, which otherwise pins the commanded setpoint
    # close to the lagging motor under held-key auto-repeat.
    # ------------------------------------------------------------------
    def extends_in_same_direction(self, new_target: float) -> bool:
        """True iff ``new_target`` is further along the motion we're
        already doing. If the trajectory has effectively completed
        (elapsed >= total), we treat any new target as an extension
        when its direction matches our last sign; otherwise use the
        un-reached target as reference.
        """
        if self.deg_per_tick == 0.0:
            # Previous trajectory was degenerate (already at target);
            # any new target is equivalent to a fresh direction.
            return False
        sign = 1.0 if self.deg_per_tick > 0.0 else -1.0
        # Where is the commanded setpoint heading? If new_target is on
        # the same side as sign relative to target_deg, we're extending.
        # Edge case: new_target exactly equal to target_deg — harmless
        # to treat as extension (zero-distance extend).
        if sign > 0:
            return new_target >= self.target_deg
        else:
            return new_target <= self.target_deg

    def extend_target(self, new_target: float, deg_per_sec: float, hz: int) -> None:
        """Update target in-place, preserving elapsed_steps + start_deg so
        the time-based setpoint march continues from where it was.
        ``deg_per_tick`` is re-derived from the (possibly changed) speed
        setting but keeps the same sign.
        """
        self.target_deg = new_target
        self.last_updated = time.perf_counter()
        if deg_per_sec <= 0:
            # No motion; let setpoint() clamp to target on the next tick.
            self.deg_per_tick = 0.0
            return

        sign = 1.0 if self.deg_per_tick >= 0.0 else -1.0
        # Recompute deg_per_tick so speed changes from the System tab
        # take effect immediately on held-key accumulation.
        self.deg_per_tick = sign * deg_per_sec / hz

        # Recompute total_steps from the current setpoint, not start,
        # so we don't pre-mark the trajectory "done" just because start
        # is now far behind. Setpoint() returns target once
        # elapsed >= total, so we count future ticks only.
        cur_setpoint = self.setpoint()
        remaining_dist = abs(new_target - cur_setpoint)
        if remaining_dist < 1e-4:
            # Already at (new) target — make the trajectory report done.
            self.total_steps = self.elapsed_steps
            return
        remaining_ticks = max(1, int(math.ceil(remaining_dist * hz / deg_per_sec)))
        # Total is "current elapsed" + "ticks still needed". Don't reset
        # elapsed so setpoint() keeps counting from start_deg.
        self.total_steps = self.elapsed_steps + remaining_ticks


@dataclass
class CartesianTarget:
    """Six-DOF + gripper target pose expressed in the bimanual URDF's
    world frame (translation in metres, rpy in radians) plus an optional
    gripper absolute angle.
    """
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    gripper: Optional[float] = None     # degrees; None = leave gripper alone


@dataclass
class _Command:
    """Message posted from UI thread to the worker. Minimal; keep serializable."""
    # "set_target"         — joint target change
    # "torque"             — torque on/off
    # "estop"              — e-stop
    # "stop"               — thread exit
    # "set_mode"           — switch an arm between joint/cartesian mode
    # "set_cart_target"    — set the cartesian target pose for an arm
    kind: str
    arm: Optional[str] = None
    joint: Optional[str] = None
    target_deg: Optional[float] = None
    torque_enabled: Optional[bool] = None
    # Per-call speed override for set_target. When None, the worker uses
    # runtime.max_speed_deg_per_sec (the System-tab setting). The override
    # only affects the trajectory built for this one command — later
    # commands fall back to the runtime setting unless they override too.
    deg_per_sec: Optional[float] = None
    mode: Optional[str] = None          # for set_mode: "joint" | "cartesian"
    cart_target: Optional[CartesianTarget] = None


class MotionWorker(QThread):
    """Motion control loop running in a dedicated QThread.

    Signals:
        state_updated(dict): emits {joint_key: current_deg, ...} every tick
        send_error(str): emitted if a send_action fails
    """

    state_updated = pyqtSignal(dict)      # {"right_joint_1.pos": 12.3, ...}
    send_error = pyqtSignal(str)

    def __init__(
        self,
        robot: RobotService,
        runtime: RuntimeState,
        vr_receiver: VRInputReceiver | None = None,
    ) -> None:
        super().__init__()
        self.robot = robot
        self.runtime = runtime
        # Optional: set in Phase 2b-α so the worker can read controller
        # state directly when VR control is enabled for an arm. Phase 2b-α
        # only stores the reference — no per-tick reads yet.
        self.vr_receiver = vr_receiver
        self.command_queue: queue.Queue[_Command] = queue.Queue()
        self._stop_flag = False

        # All 16 trajectories, keyed by "{arm}_{joint}.pos" — same shape the
        # robot's action/observation dicts use. Initialized in run() once we
        # have a real observation to seed from.
        self._trajectories: dict[str, JointTrajectory] = {}
        self._torque_on: dict[str, bool] = {"left": False, "right": False}
        self._last_current: dict[str, float] = {}
        self._initialized = False  # True once we have a first observation

        # Gravity compensation — loaded lazily in run() so startup doesn't
        # fail if the URDF is missing. Either both are set or both are None.
        self._gc_left: GravityCompensator | None = None
        self._gc_right: GravityCompensator | None = None

        # Per-arm control mode. When "joint" the worker consumes
        # self._trajectories (the existing path). When "cartesian" the
        # worker runs IK on the arm's cart_target every tick, writes the
        # resulting joint angles into self._trajectories, and the normal
        # ramped send path executes them.
        self._mode: dict[str, str] = {"left": "joint", "right": "joint"}
        self._cart_target: dict[str, Optional[CartesianTarget]] = {
            "left": None, "right": None,
        }
        # IK solvers — lazy-loaded with gravity comp.
        self._ik_left: Optional[CartesianIKSolver] = None
        self._ik_right: Optional[CartesianIKSolver] = None
        # Last successful IK solution per arm, used to seed the next solve.
        self._last_ik_q_deg: dict[str, Optional[list[float]]] = {
            "left": None, "right": None,
        }
        # Stick flag — True when the most recent IK call for an arm failed
        # (did not converge or got clamped to joint limits). When True we
        # keep the arm pinned to its last good joint target until either
        # the target changes or the user fixes the pose.
        self._ik_failed: dict[str, bool] = {"left": False, "right": False}

        # Per-arm VR-control hard enable. Phase 2b-α just tracks the flag
        # (so the UI can toggle and we can see the state round-trip); no
        # motor behavior change yet. Phase 2b-β will, when True, poll
        # self.vr_receiver each tick and update self._cart_target from
        # controller deltas when grip is above threshold.
        self._vr_enabled: dict[str, bool] = {"left": False, "right": False}

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

    def post_set_mode(self, arm: str, mode: str) -> None:
        """Switch one arm between 'joint' and 'cartesian' control modes."""
        self.command_queue.put(_Command(kind="set_mode", arm=arm, mode=mode))

    def post_set_cart_target(self, arm: str, target: CartesianTarget) -> None:
        """Set the cartesian target pose for one arm. Only has effect in
        cartesian mode."""
        self.command_queue.put(_Command(kind="set_cart_target", arm=arm, cart_target=target))

    def current_mode(self, arm: str) -> str:
        """Read-only mode accessor for the UI. Thread-safety: Python
        dict read is atomic for a single key, and this is only read by
        the UI thread to decide slider enable/disable."""
        return self._mode.get(arm, "joint")

    def post_vr_enable(self, arm: str, enabled: bool) -> None:
        """Toggle VR hard-enable for one arm. Phase 2b-α: flips the flag
        and switches the arm to cartesian mode when enabled. The motor-
        side behavior (grip-gated pose tracking, trigger→gripper, etc.)
        lands in Phase 2b-β.
        """
        self.command_queue.put(_Command(kind="vr_enable", arm=arm, torque_enabled=enabled))

    def vr_enabled(self, arm: str) -> bool:
        """Read-only VR-enable state for the UI."""
        return self._vr_enabled.get(arm, False)

    def compute_fk(self, arm: str) -> Optional[pin.SE3]:
        """Return the arm's current TCP pose (from the last observed joint
        positions). Useful for seeding Cartesian tab spinboxes. Returns
        None if IK solver isn't loaded yet or state isn't available.
        """
        solver = self._ik_left if arm == "left" else self._ik_right
        if solver is None:
            return None
        q7 = [self._last_current.get(f"{arm}_joint_{i+1}.pos") for i in range(7)]
        if any(v is None for v in q7):
            return None
        return solver.forward_kinematics([float(v) for v in q7])

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

        # Load gravity compensation models. Both instances use the same
        # bimanual URDF; each picks its own 7 joints by name (arm_side arg).
        try:
            urdf = str(config.GRAVITY_URDF_PATH)
            self._gc_left = GravityCompensator(urdf, arm_side="left")
            self._gc_right = GravityCompensator(urdf, arm_side="right")
            logger.info(
                f"Gravity comp loaded from {urdf} "
                f"(initial scale={self.runtime.gravity_comp_scale})"
            )
        except Exception as e:
            self._gc_left = None
            self._gc_right = None
            logger.error(
                f"Gravity comp disabled (URDF load failed): {e}. "
                "Setpoints will be sent with tau_ff=0."
            )

        # Load per-arm IK solvers. They need joint_limits, which we pull
        # from the robot service. If that's unavailable we ship zeros and
        # IK stays disabled for that arm.
        try:
            urdf = str(config.GRAVITY_URDF_PATH)
            left_cfg = self.robot.arm_config_snapshot("left") or {}
            right_cfg = self.robot.arm_config_snapshot("right") or {}
            if left_cfg.get("joint_limits"):
                self._ik_left = CartesianIKSolver(urdf, "left", left_cfg["joint_limits"])
            if right_cfg.get("joint_limits"):
                self._ik_right = CartesianIKSolver(urdf, "right", right_cfg["joint_limits"])
            logger.info(
                f"IK solvers loaded: left={self._ik_left is not None}, "
                f"right={self._ik_right is not None}"
            )
        except Exception as e:
            self._ik_left = None
            self._ik_right = None
            logger.error(f"IK solver load failed: {e}. Cartesian mode disabled.")

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

        # 3a. For arms in cartesian mode, solve IK once and write the
        # resulting joint angles as new targets. The rest of the tick
        # (ramp + lead cap + send) runs exactly as in joint mode.
        for arm in ("left", "right"):
            if self._mode[arm] != "cartesian":
                continue
            self._cartesian_tick(arm, current)

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

        # 6. Send one MIT batch per arm, folding in gravity comp torques.
        # We bypass robot.send_action() because it hardcodes tau_ff=0 and
        # we want the Damiao MIT packet's torque feedforward slot.
        if action:
            self._send_mit_batches(action, current)

    def _send_mit_batches(
        self,
        action: dict[str, float],
        current: dict[str, float],
    ) -> None:
        """Split the per-joint action dict by arm, compute gravity comp
        torques per arm, and send one MIT batch per arm through the
        robot service.
        """
        scale = float(self.runtime.gravity_comp_scale)

        for arm, gc in (("left", self._gc_left), ("right", self._gc_right)):
            # Collect this arm's actions (joint_name -> pos_deg).
            arm_prefix = f"{arm}_"
            arm_action: dict[str, float] = {}
            for key, pos in action.items():
                if key.startswith(arm_prefix):
                    # Strip "{arm}_" and ".pos" to get bare motor name
                    motor = key[len(arm_prefix):].removesuffix(".pos")
                    arm_action[motor] = pos
            if not arm_action:
                continue

            cfg = self.robot.arm_config_snapshot(arm)
            if cfg is None:
                continue
            kp_list = cfg["position_kp"]
            kd_list = cfg["position_kd"]

            # Compute gravity torques in the same joint order (j1..j7).
            tau_ff = [0.0] * 8  # index 7 (gripper) stays 0
            if gc is not None and scale != 0.0:
                pos_deg_7 = [
                    current.get(f"{arm}_joint_{i+1}.pos", 0.0) for i in range(7)
                ]
                try:
                    raw = gc.compute_tau_gravity(pos_deg_7)
                    for i in range(7):
                        tau_ff[i] = raw[i] * scale
                except Exception as e:
                    logger.error(f"gravity comp failed for {arm} arm: {e}")

            # Build the MIT command dict in the shape _mit_control_batch wants.
            motor_order = config.JOINT_NAMES  # joint_1..joint_7, gripper
            commands: dict[str, tuple[float, float, float, float, float]] = {}
            for i, motor in enumerate(motor_order):
                if motor not in arm_action:
                    continue
                kp = kp_list[i] if isinstance(kp_list, list) else kp_list
                kd = kd_list[i] if isinstance(kd_list, list) else kd_list
                commands[motor] = (
                    kp, kd,
                    arm_action[motor],
                    0.0,              # velocity feedforward unused
                    tau_ff[i],
                )

            if not commands:
                continue
            try:
                self.robot.send_mit_batch(arm, commands)
            except Exception as e:
                logger.error(f"send_mit_batch failed for {arm} arm: {e}")
                self.send_error.emit(str(e))

    # ------------------------------------------------------------------
    # Trajectory update helper
    # ------------------------------------------------------------------
    def _set_joint_target(
        self,
        key: str,
        current_deg: float,
        new_target_deg: float,
        deg_per_sec: float,
    ) -> None:
        """Update a joint's trajectory. If an existing trajectory is fresh
        AND extends in the same direction as the new target, we update it
        in-place so held-key auto-repeat accumulates motion instead of
        resetting the setpoint to current motor position every keystroke.
        Otherwise rebuild from the motor's actual current position.
        """
        now = time.perf_counter()
        existing = self._trajectories.get(key)
        stale = (
            existing is None
            or (now - existing.last_updated) > config.TRAJECTORY_STALENESS_SEC
        )
        if (not stale and existing is not None
                and existing.extends_in_same_direction(new_target_deg)):
            existing.extend_target(new_target_deg, deg_per_sec=deg_per_sec,
                                   hz=config.MOTION_HZ)
        else:
            self._trajectories[key] = JointTrajectory.new(
                start=current_deg,
                target=new_target_deg,
                deg_per_sec=deg_per_sec,
                hz=config.MOTION_HZ,
            )

    # ------------------------------------------------------------------
    # Cartesian mode
    # ------------------------------------------------------------------
    def _cartesian_tick(self, arm: str, current: dict[str, float]) -> None:
        """Solve IK for this arm's cartesian target, overwrite its joint
        trajectories with the solution. Only runs when arm is in cartesian
        mode. Joint-space ramp + lead cap run unchanged after this.
        """
        solver = self._ik_left if arm == "left" else self._ik_right
        target = self._cart_target[arm]
        if solver is None or target is None:
            return

        # Seed from last IK solution if available, else from current motor q.
        seed = self._last_ik_q_deg[arm]
        if seed is None:
            seed = [current.get(f"{arm}_joint_{i+1}.pos", 0.0) for i in range(7)]

        pose = pose_from_xyzrpy(
            target.x, target.y, target.z,
            target.roll, target.pitch, target.yaw,
        )
        result: IKResult = solver.solve(pose, q_seed_deg_7=seed)

        if not result.converged:
            # Freeze — do not update trajectories. User sees no motion and
            # (via the UI) gets a warning. Re-try next tick in case target
            # moved back into reach.
            if not self._ik_failed[arm]:
                logger.warning(
                    f"IK failed for {arm} arm: pos_err={result.pos_err_mm:.1f}mm "
                    f"rot_err={result.rot_err_deg:.1f}° after {result.iters} iters. "
                    "Arm will freeze until target becomes reachable."
                )
                self.send_error.emit(f"{arm}: IK freeze — target unreachable")
            self._ik_failed[arm] = True
            return

        # Converged. If the solution was clamped to joint limits the TCP
        # error may be non-negligible; warn once per transition.
        if result.clamped:
            if not self._ik_failed[arm]:
                logger.warning(f"{arm}: IK solution clamped to joint limits")
                self.send_error.emit(f"{arm}: IK solution clamped to joint limits")
            self._ik_failed[arm] = True
        else:
            if self._ik_failed[arm]:
                logger.info(f"{arm}: IK recovered")
                self.send_error.emit("")  # clear warning
            self._ik_failed[arm] = False

        self._last_ik_q_deg[arm] = list(result.q_deg)

        # Route IK's joint targets through the shared helper so the
        # per-tick IK updates also get the extend-in-same-direction
        # treatment (otherwise holding a cartesian jog key would hit
        # the same accumulation bug as the joint-space path).
        speed = self.runtime.max_speed_deg_per_sec
        for i in range(7):
            joint = f"joint_{i+1}"
            key = f"{arm}_{joint}.pos"
            cur = current.get(key)
            if cur is None:
                continue
            self._set_joint_target(key, float(cur), float(result.q_deg[i]), speed)

        # Gripper is not part of IK; if target specifies one, send it too.
        if target.gripper is not None:
            key = f"{arm}_gripper.pos"
            cur = current.get(key)
            if cur is not None:
                self._set_joint_target(key, float(cur), float(target.gripper), speed)

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
                new_target = float(cmd.target_deg)
                self._set_joint_target(key, cur, new_target, speed)
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
            elif cmd.kind == "set_mode":
                arm = cmd.arm or ""
                mode = cmd.mode or "joint"
                if arm in self._mode and mode in ("joint", "cartesian"):
                    self._mode[arm] = mode
                    # Reset the IK seed and failure flag when switching
                    # so the next tick starts fresh.
                    self._last_ik_q_deg[arm] = None
                    self._ik_failed[arm] = False
                    logger.info(f"{arm}: mode set to {mode}")
            elif cmd.kind == "set_cart_target":
                if cmd.arm in self._cart_target:
                    self._cart_target[cmd.arm] = cmd.cart_target
            elif cmd.kind == "vr_enable":
                arm = cmd.arm or ""
                enabled = bool(cmd.torque_enabled)
                if arm not in self._vr_enabled:
                    continue
                self._vr_enabled[arm] = enabled
                if enabled:
                    # Switch to cartesian mode so the (future) per-tick
                    # VR pose updates have somewhere to land. Seed the
                    # cart_target from current TCP pose so the arm
                    # doesn't jump on mode switch.
                    self._mode[arm] = "cartesian"
                    self._last_ik_q_deg[arm] = None
                    self._ik_failed[arm] = False
                    fk = self.compute_fk(arm)
                    if fk is not None:
                        x, y, z, roll, pitch, yaw = pose_to_xyzrpy(fk)
                        self._cart_target[arm] = CartesianTarget(
                            x=x, y=y, z=z,
                            roll=roll, pitch=pitch, yaw=yaw,
                        )
                    logger.info(
                        f"{arm}: VR control ENABLED; arm switched to cartesian "
                        f"mode (Phase 2b-α — not yet tracking controller pose)"
                    )
                else:
                    # Drop back to joint mode so the arm stops consuming
                    # cartesian targets. Trajectories pin to current in
                    # the joint-mode per-tick path on the next tick.
                    self._mode[arm] = "joint"
                    self._cart_target[arm] = None
                    logger.info(f"{arm}: VR control DISABLED; arm back to joint mode")
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
                # Extra safety: zero gravity-comp scale so re-enabling torque
                # leaves the motors fully passive until the user deliberately
                # dials it back up from the System tab.
                self.runtime.gravity_comp_scale = 0.0
                # Drop cartesian mode — operator has to deliberately
                # re-enter it. Cleared targets too. Also drop VR hard-
                # enable so controller motion doesn't silently resume
                # the moment torque comes back.
                for arm in ("left", "right"):
                    self._mode[arm] = "joint"
                    self._cart_target[arm] = None
                    self._last_ik_q_deg[arm] = None
                    self._ik_failed[arm] = False
                    self._vr_enabled[arm] = False
                logger.warning(
                    "motion worker: e-stop consumed; torque off; trajectories pinned; "
                    "gravity_comp_scale reset to 0; arms returned to joint mode; "
                    "VR control disabled"
                )

    def _read_state(self) -> Optional[dict[str, float]]:
        """Read motor positions only. Cameras are read by the UI thread."""
        obs = self.robot.get_observation()
        if obs is None:
            return None
        # Keep only .pos scalars. Cameras and .vel/.torque are noise here.
        return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
