"""Single-class subclass that disables the automatic ``set_zero_position``
call inside ``OpenArmFollower.connect()``.

Background
----------
``OpenArmFollower.connect()`` in LeRobot v0.5.1 unconditionally calls
``self.bus.set_zero_position()`` at the end of connect whenever a
calibration file exists — see
``lerobot/robots/openarm_follower/openarm_follower.py:147-148``.

For Damiao motors this is destructive: the motor's factory-set zero
(stored in the motor's internal flash) is overwritten with whatever
physical pose the arm happens to be in at connect time. We lose the
factory calibration on every app startup.

This subclass overrides ``connect()`` with a verbatim copy of the
upstream method body minus the ``set_zero_position`` line. The rest
of the LeRobot connect flow (CAN bus open, calibration file load,
camera connect, motor configure, torque enable) is unchanged.

Used together with ``BiOpenArmFollowerNoAutoZero`` — the bimanual
subclass replaces the two stock ``OpenArmFollower`` sub-arm instances
with instances of this class. ``robot_service.py`` constructs the
bimanual subclass directly.

Upstream reference
------------------
LeRobot 0.5.1. If you upgrade LeRobot and ``OpenArmFollower.connect()``
changes in ``lerobot/robots/openarm_follower/openarm_follower.py``,
diff the upstream body against ``connect()`` below and re-sync.
"""

from __future__ import annotations

import logging

from lerobot.robots.openarm_follower import OpenArmFollower
from lerobot.utils.decorators import check_if_already_connected

logger = logging.getLogger(__name__)


class OpenArmFollowerNoAutoZero(OpenArmFollower):
    """OpenArmFollower that does NOT re-zero motors on connect."""

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.

        Verbatim copy of ``OpenArmFollower.connect`` (LeRobot 0.5.1) with
        the single ``self.bus.set_zero_position()`` call removed so the
        motor's internal zero (factory-calibrated or deliberately set via
        the System-tab button) is preserved across app restarts.
        """
        # Connect to CAN bus
        logger.info(f"Connecting arm on {self.config.port} (no auto-zero)...")
        self.bus.connect()

        # Run calibration if needed
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the "
                "calibration file or no calibration file found"
            )
            self.calibrate()

        # Open cameras individually. A missing / unplugged camera must
        # not prevent the arm from being usable — the UI can still run
        # on motor state alone. We log each failure and drop the dead
        # camera from self.cameras so downstream code (get_observation,
        # bimanual.cameras aggregate, etc.) doesn't try to read from it.
        dead_cams = []
        for name, cam in self.cameras.items():
            try:
                cam.connect()
            except Exception as e:
                logger.warning(
                    f"Camera {name!r} failed to open ({e}). Continuing without it."
                )
                dead_cams.append(name)
        for name in dead_cams:
            self.cameras.pop(name, None)

        self.configure()

        # NOTE (intentionally omitted):
        #     if self.is_calibrated:
        #         self.bus.set_zero_position()
        # Skipping this is the whole point of this subclass — see module docstring.

        self.bus.enable_torque()

        logger.info(f"{self} connected (motor zero preserved).")
