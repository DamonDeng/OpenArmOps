"""Loader for ``key_bindings.json``.

Kept tiny so the UI can reload the file at runtime without restarting. Schema
validation is minimal — enough to produce a useful error message, no more.

Each row maps (key, modifier) → either a JOINT-MODE binding (a joint name +
direction) or a CARTESIAN-MODE binding (a cartesian axis + direction). The
same (key, modifier) pair can appear once in each mode — the keyboard filter
picks which row to use based on the target arm's current mode.

Cartesian axis names: "x", "y", "z" for translation (world frame) and
"roll", "pitch", "yaw" for rotation (tool frame). Gripper is "gripper"
in both modes since it's a 1-DOF axis with no cartesian analog — its
binding rows reuse the joint-mode shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config


# Currently supported modifiers. Exactly one of these describes the state of
# the user's modifier keys at keypress time. "none" = no modifier held;
# otherwise exactly one of shift/ctrl/alt is held. Multi-modifier combos
# (e.g. Ctrl+Shift) are intentionally not supported — the keyboard filter
# ignores them rather than guessing which layer was intended.
ALLOWED_MODIFIERS = ("none", "shift", "ctrl", "alt")

# Per-arm control modes. Keys reinterpret based on the arm's mode at
# keypress time — see ControllerTab._apply_key_nudge.
ALLOWED_MODES = ("joint", "cartesian")

# Valid targets per mode.
CARTESIAN_AXES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


@dataclass(frozen=True)
class Binding:
    key: str           # single lowercase character
    modifier: str      # one of ALLOWED_MODIFIERS
    mode: str          # one of ALLOWED_MODES
    arm: str
    target: str        # joint name for mode='joint'; axis for mode='cartesian'
    direction: int     # -1 or +1
    note: str = ""

    @property
    def joint(self) -> str:
        """Back-compat accessor; only valid when mode == 'joint'."""
        return self.target


@dataclass(frozen=True)
class BindingTable:
    """One lookup dict per (mode, arm). The worker picks the right dict
    based on the arm's current mode and the arm the binding targets."""
    # {(key, modifier): Binding} keyed to arm-agnostic ⇒ joint rows
    joint: dict[tuple[str, str], Binding]
    # {(key, modifier): Binding}
    cartesian: dict[tuple[str, str], Binding]

    def __len__(self) -> int:
        return len(self.joint) + len(self.cartesian)


def load_bindings(path: Path | str = config.DEFAULT_KEY_BINDINGS_PATH) -> BindingTable:
    """Parse the JSON file into per-mode lookup tables.

    Raises ValueError with a line-level message if the JSON is malformed or
    a row fails basic validation. Lookup from the UI event filter is O(1):
    pick the right mode's dict, index by (key, modifier).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("bindings")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: top-level key 'bindings' must be a list")

    joint_tbl: dict[tuple[str, str], Binding] = {}
    cart_tbl: dict[tuple[str, str], Binding] = {}

    for i, row in enumerate(rows):
        try:
            key = str(row["key"]).lower()
            arm = row["arm"]
            direction = int(row["direction"])
        except (KeyError, TypeError) as e:
            raise ValueError(f"{path} row {i}: missing/invalid field ({e})") from e

        modifier = str(row.get("modifier", "none")).lower()
        mode = str(row.get("mode", "joint")).lower()

        # In joint mode the target field is named "joint" (back-compat); in
        # cartesian mode it's named "axis". Accept either name in either
        # mode for robustness.
        target = row.get("joint") or row.get("axis") or row.get("target")
        if target is None:
            raise ValueError(f"{path} row {i}: missing 'joint'/'axis' field")
        target = str(target)

        if len(key) != 1:
            raise ValueError(f"{path} row {i}: 'key' must be a single character, got {row['key']!r}")
        if modifier not in ALLOWED_MODIFIERS:
            raise ValueError(
                f"{path} row {i}: 'modifier' must be one of {ALLOWED_MODIFIERS}, got {modifier!r}"
            )
        if mode not in ALLOWED_MODES:
            raise ValueError(
                f"{path} row {i}: 'mode' must be one of {ALLOWED_MODES}, got {mode!r}"
            )
        if arm not in config.ARM_SIDES:
            raise ValueError(f"{path} row {i}: 'arm' must be one of {config.ARM_SIDES}, got {arm!r}")
        if direction not in (-1, 1):
            raise ValueError(f"{path} row {i}: 'direction' must be -1 or +1, got {direction}")

        if mode == "joint":
            if target not in config.JOINT_NAMES:
                raise ValueError(
                    f"{path} row {i}: joint-mode target must be one of {config.JOINT_NAMES}, "
                    f"got {target!r}"
                )
            tbl = joint_tbl
        else:
            if target not in CARTESIAN_AXES:
                raise ValueError(
                    f"{path} row {i}: cartesian-mode axis must be one of {CARTESIAN_AXES}, "
                    f"got {target!r}"
                )
            tbl = cart_tbl

        lookup_key = (key, modifier)
        if lookup_key in tbl:
            raise ValueError(
                f"{path} row {i}: duplicate binding for key={key!r} modifier={modifier!r} "
                f"in mode={mode!r}"
            )

        tbl[lookup_key] = Binding(
            key=key, modifier=modifier, mode=mode, arm=arm,
            target=target, direction=direction,
            note=str(row.get("note", "")),
        )

    return BindingTable(joint=joint_tbl, cartesian=cart_tbl)
