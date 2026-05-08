"""Thread-safe wrapper around BiOpenArmFollower for the UI.

The UI owns the single robot instance. A QMutex guards the robot object so
the poll timer and UI event handlers don't race on CAN traffic. This is
the minimum viable version: connect, disconnect, disable-torque-on-startup,
get_observation, send_action. No background control loop in M1.
"""

from __future__ import annotations

import logging
import threading

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors import MotorCalibration
from lerobot.robots.bi_openarm_follower import BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfigBase

from .bi_openarm_follower_no_auto_zero import BiOpenArmFollowerNoAutoZero

from . import config

logger = logging.getLogger(__name__)


class RobotService:
    """Serializes all access to BiOpenArmFollower behind a mutex.

    The UI never touches ``self._robot`` directly — it calls methods here. This
    keeps the locking discipline local to one file.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._robot: BiOpenArmFollowerNoAutoZero | None = None
        self._connected = False
        # Tracks whether cameras are known-dead. We log the transition ONCE
        # (not per tick) and then fall back to a state-only observation until
        # the app is restarted. No auto-reconnect by design.
        self._cameras_dead = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Construct and connect the robot. Torque is disabled immediately
        after connect — users explicitly turn it on per arm from the UI.
        """
        with self._lock:
            if self._connected:
                return

            cams = {
                name: OpenCVCameraConfig(
                    index_or_path=spec["index"],
                    width=spec["w"],
                    height=spec["h"],
                    fps=spec["fps"],
                )
                for name, spec in config.CAMERAS.items()
            }
            left_cfg = OpenArmFollowerConfigBase(
                port=config.CAN_LEFT,
                side="left" if config.USE_FULL_LIMITS else None,
                max_relative_target=config.MAX_RELATIVE_TARGET_DEG,
            )
            right_cfg = OpenArmFollowerConfigBase(
                port=config.CAN_RIGHT,
                side="right" if config.USE_FULL_LIMITS else None,
                max_relative_target=config.MAX_RELATIVE_TARGET_DEG,
            )
            cfg = BiOpenArmFollowerConfig(
                id="openarm_controller_ui",
                left_arm_config=left_cfg,
                right_arm_config=right_cfg,
                cameras=cams,
            )

            logger.info("Connecting BiOpenArmFollower…")
            # Use our subclasses that skip the automatic set_zero_position
            # call during connect — preserves the motor's factory-calibrated
            # zero across app restarts.
            self._robot = BiOpenArmFollowerNoAutoZero(cfg)

            # calibrate=False: skip the built-in `input()` prompt that blocks
            # startup asking the user to position the arms. Calibration is a
            # deliberate action from the System tab, not a startup side effect.
            self._robot.connect(calibrate=False)
            # BiOpenArmFollower.connect() re-enables torque at the end. Force
            # it back off so the user can move arms by hand before commanding.
            self._robot.left_arm.bus.disable_torque()
            self._robot.right_arm.bus.disable_torque()
            self._connected = True
            # Read calibration directly while we still hold the lock — calling
            # self.is_calibrated here would re-acquire the non-reentrant lock
            # and deadlock the startup thread.
            cal_state = "calibrated" if self._robot.is_calibrated else "NOT calibrated"
            logger.info(f"Robot connected; torque disabled on both arms; {cal_state}.")

    def disconnect(self) -> None:
        with self._lock:
            if not self._connected or self._robot is None:
                return
            try:
                self._robot.disconnect()
            except Exception as e:
                logger.warning(f"disconnect error: {e}")
            finally:
                self._robot = None
                self._connected = False
                logger.info("Robot disconnected.")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        """True iff BOTH arms have a matching calibration loaded on their motors."""
        with self._lock:
            if not self._connected or self._robot is None:
                return False
            return self._robot.is_calibrated

    def calibrate_arm(self, arm: str) -> None:
        """Run the BiOpenArmFollower per-arm calibration:
        disables torque, sets current pose as zero, writes a calibration
        file with the generic (-90, +90) safety ranges.

        Caller MUST disable torque on the other arm too and confirm the arm
        is in the 'hanging straight down, gripper closed' pose before calling.
        """
        with self._lock:
            if not self._connected or self._robot is None:
                raise RuntimeError("not connected")
            target = self._robot.left_arm if arm == "left" else self._robot.right_arm

            # OpenArmFollower.calibrate() prompts via input() when a calibration
            # file already exists. Bypass that prompt by forcing a fresh zero.
            target.bus.disable_torque()
            target.bus.set_zero_position()

            # Write the generic-range calibration file so BiOpenArmFollower
            # treats us as calibrated on next startup. Mirror the defaults
            # from OpenArmFollower.calibrate (range_min=-90, range_max=90).
            for motor_name, motor in target.bus.motors.items():
                target.calibration[motor_name] = MotorCalibration(
                    id=motor.id, drive_mode=0, homing_offset=0,
                    range_min=-90, range_max=90,
                )
            target.bus.write_calibration(target.calibration)
            target._save_calibration()
            logger.info(f"{arm} arm: calibration written to {target.calibration_fpath}")

    def set_zero(self, arm: str) -> None:
        """Re-zero the arm at its current physical pose (no calibration file write).

        Use when you've moved the arm manually and want the current pose to
        read as 0°. Does not change joint-limit ranges or drive modes.
        """
        with self._lock:
            if not self._connected or self._robot is None:
                raise RuntimeError("not connected")
            target = self._robot.left_arm if arm == "left" else self._robot.right_arm
            target.bus.disable_torque()
            target.bus.set_zero_position()
            logger.info(f"{arm} arm: zero position set at current pose.")

    # ------------------------------------------------------------------
    # Torque
    # ------------------------------------------------------------------
    def set_torque(self, arm: str, enabled: bool) -> None:
        """arm in {'left', 'right'}."""
        with self._lock:
            if not self._connected or self._robot is None:
                return
            target = self._robot.left_arm if arm == "left" else self._robot.right_arm
            if enabled:
                target.bus.enable_torque()
            else:
                target.bus.disable_torque()
            logger.info(f"torque {arm}: {'on' if enabled else 'off'}")

    def emergency_stop(self) -> None:
        with self._lock:
            if not self._connected or self._robot is None:
                return
            self._robot.left_arm.bus.disable_torque()
            self._robot.right_arm.bus.disable_torque()
            logger.warning("EMERGENCY STOP: torque disabled on both arms.")

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def get_observation(self) -> dict | None:
        """Return the robot observation, tolerating camera failures.

        Normal path: delegates to BiOpenArmFollower.get_observation, which
        reads all motor states over CAN and one frame per camera.

        If any camera raises (e.g. USB disconnect → ``RuntimeError: read
        thread is not running``), we switch to a state-only fallback that
        only reads motor positions. Cameras stay dead until app restart;
        we log the transition once to avoid flooding the console.
        """
        with self._lock:
            if not self._connected or self._robot is None:
                return None

            if not self._cameras_dead:
                try:
                    return self._robot.get_observation()
                except RuntimeError as e:
                    # Most camera failures surface as RuntimeError from the
                    # OpenCVCamera reader thread. Log once and fall through
                    # to the state-only path.
                    self._cameras_dead = True
                    logger.error(
                        f"Camera read failed ({e!s}). Switching to state-only "
                        "observations for the rest of this session."
                    )

            # State-only path (cameras dead). Read motor state directly from
            # each bus and build the same key layout BiOpenArmFollower would
            # emit, minus the camera entries.
            obs: dict = {}
            try:
                right_states = self._robot.right_arm.bus.sync_read_all_states()
                for motor, state in right_states.items():
                    obs[f"right_{motor}.pos"] = state.get("position", 0.0)
                    obs[f"right_{motor}.vel"] = state.get("velocity", 0.0)
                    obs[f"right_{motor}.torque"] = state.get("torque", 0.0)
                left_states = self._robot.left_arm.bus.sync_read_all_states()
                for motor, state in left_states.items():
                    obs[f"left_{motor}.pos"] = state.get("position", 0.0)
                    obs[f"left_{motor}.vel"] = state.get("velocity", 0.0)
                    obs[f"left_{motor}.torque"] = state.get("torque", 0.0)
            except Exception as e:
                logger.exception(f"Fallback state read failed: {e}")
                return None
            return obs

    @property
    def cameras_dead(self) -> bool:
        return self._cameras_dead

    def arm_config_snapshot(self, arm: str) -> dict | None:
        """Return a copy of the kp/kd/joint_limits the worker needs for MIT
        batches. Returned dict is safe to use without holding the lock.
        """
        with self._lock:
            if not self._connected or self._robot is None:
                return None
            src = self._robot.left_arm.config if arm == "left" else self._robot.right_arm.config
            return {
                "position_kp": list(src.position_kp)
                if isinstance(src.position_kp, list) else src.position_kp,
                "position_kd": list(src.position_kd)
                if isinstance(src.position_kd, list) else src.position_kd,
                "joint_limits": dict(src.joint_limits),
            }

    def get_motor_stats(self) -> dict | None:
        """Return the Damiao bus's cached motor stats, keyed by '{arm}_{joint}'.

        The public ``sync_read_all_states()`` only returns position/velocity/
        torque, but the Damiao driver's internal ``_last_known_states`` cache
        also holds ``temp_mos`` and ``temp_rotor`` after any recent read.
        The motion worker polls state at 30 Hz, so this cache is always
        fresh when the System tab reads it at 2 Hz.

        Reads the cache directly rather than triggering new CAN traffic.
        That keeps this cheap (no extra bus load) and avoids stepping on
        the motion worker's reads.
        """
        with self._lock:
            if not self._connected or self._robot is None:
                return None
            result: dict[str, dict[str, float]] = {}
            for arm_name, arm in (("left", self._robot.left_arm),
                                  ("right", self._robot.right_arm)):
                cache = getattr(arm.bus, "_last_known_states", {}) or {}
                for motor, stats in cache.items():
                    # Defensive copy — the worker thread may overwrite
                    # stats[...] any moment.
                    result[f"{arm_name}_{motor}"] = dict(stats)
            return result

    def send_action(self, action: dict) -> None:
        with self._lock:
            if not self._connected or self._robot is None:
                return
            self._robot.send_action(action)

    def send_mit_batch(
        self,
        arm: str,
        commands: dict[str, tuple[float, float, float, float, float]],
    ) -> None:
        """Send a direct MIT-control batch, bypassing send_action.

        Use this when you need to pass a non-zero feedforward torque —
        LeRobot's send_action hardcodes torque=0. The command tuple per
        motor is ``(kp, kd, position_deg, velocity_deg_per_sec, torque_ff_Nm)``.
        Joint-limit clipping and kp/kd selection are the caller's
        responsibility — this wrapper just forwards to the bus.
        """
        with self._lock:
            if not self._connected or self._robot is None:
                return
            target_arm = self._robot.left_arm if arm == "left" else self._robot.right_arm
            target_arm.bus._mit_control_batch(commands)
