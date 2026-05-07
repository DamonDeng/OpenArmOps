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
from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfigBase

from . import config

logger = logging.getLogger(__name__)


class RobotService:
    """Serializes all access to BiOpenArmFollower behind a mutex.

    The UI never touches ``self._robot`` directly — it calls methods here. This
    keeps the locking discipline local to one file.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._robot: BiOpenArmFollower | None = None
        self._connected = False

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
            self._robot = BiOpenArmFollower(cfg)
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
    # IO (used by later milestones — stubbed for M1)
    # ------------------------------------------------------------------
    def get_observation(self) -> dict | None:
        with self._lock:
            if not self._connected or self._robot is None:
                return None
            return self._robot.get_observation()

    def send_action(self, action: dict) -> None:
        with self._lock:
            if not self._connected or self._robot is None:
                return
            self._robot.send_action(action)
