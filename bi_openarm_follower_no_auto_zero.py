"""Bimanual wrapper that uses ``OpenArmFollowerNoAutoZero`` for each arm.

Stock ``BiOpenArmFollower.__init__`` constructs ``OpenArmFollower`` directly
for its ``left_arm`` / ``right_arm``. We can't just override ``connect()``
because the destructive ``set_zero_position()`` call lives inside
``OpenArmFollower.connect()`` — which this class delegates to. So the fix
has to swap the sub-arm *class* itself.

Strategy: call stock ``__init__`` (so LeRobot builds the two per-arm
``OpenArmFollowerConfig`` instances for us), then replace the two sub-arm
objects with our no-auto-zero subclass, reusing the configs the stock init
just built. Finally, rebuild ``self.cameras`` to reference cameras attached
to the new sub-arms.

Why reuse the stock ``__init__`` rather than duplicate its body: the per-arm
config construction (lines 52-86 in upstream bi_openarm_follower.py) copies
16 fields across from the bimanual config to each sub-arm config. If LeRobot
adds a new field in a future version, stock init picks it up automatically
and our subclass inherits that. A full duplicate would silently drop the
new field.

Upstream reference: LeRobot 0.5.1 lerobot/robots/bi_openarm_follower/bi_openarm_follower.py
"""

from __future__ import annotations

import logging

from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig

from .openarm_follower_no_auto_zero import OpenArmFollowerNoAutoZero

logger = logging.getLogger(__name__)


class BiOpenArmFollowerNoAutoZero(BiOpenArmFollower):
    """BiOpenArmFollower whose sub-arms do not re-zero motors on connect."""

    def __init__(self, config: BiOpenArmFollowerConfig) -> None:
        super().__init__(config)
        # Stock __init__ has stamped OpenArmFollower instances on left_arm /
        # right_arm. Swap them for our subclass, reusing the per-arm configs
        # it already built. Camera objects on the stock sub-arms are
        # discarded — fresh sub-arms build their own, which is what our
        # rebuilt self.cameras must reference.
        left_cfg = self.left_arm.config
        right_cfg = self.right_arm.config
        self.left_arm = OpenArmFollowerNoAutoZero(left_cfg)
        self.right_arm = OpenArmFollowerNoAutoZero(right_cfg)
        self.cameras = {
            **self.left_arm.cameras,
            **self.right_arm.cameras,
        }
        logger.info("BiOpenArmFollowerNoAutoZero: sub-arms swapped, motor zero will be preserved on connect.")
