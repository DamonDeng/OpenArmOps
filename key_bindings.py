"""Loader for ``key_bindings.json``.

Kept tiny so the UI can reload the file at runtime without restarting. Schema
validation is minimal — enough to produce a useful error message, no more.

Each row maps (key, modifier) → (arm, joint, direction). The same key may
appear multiple times with different modifiers for a multi-layer keyboard
(e.g. `e` = left shoulder forward, `shift+e` = left elbow fold).
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


@dataclass(frozen=True)
class Binding:
    key: str           # single lowercase character
    modifier: str      # one of ALLOWED_MODIFIERS
    arm: str
    joint: str
    direction: int     # -1 or +1
    note: str = ""


def load_bindings(path: Path | str = config.DEFAULT_KEY_BINDINGS_PATH) -> dict[tuple[str, str], Binding]:
    """Parse the JSON file. Returns {(lowercase_key, modifier): Binding}.

    Raises ValueError with a line-level message if the JSON is malformed or a
    row fails basic validation. Lookup is O(1) per keystroke from the UI
    event filter.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("bindings")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: top-level key 'bindings' must be a list")

    result: dict[tuple[str, str], Binding] = {}
    for i, row in enumerate(rows):
        try:
            key = str(row["key"]).lower()
            arm = row["arm"]
            joint = row["joint"]
            direction = int(row["direction"])
        except (KeyError, TypeError) as e:
            raise ValueError(f"{path} row {i}: missing/invalid field ({e})") from e

        # Modifier is optional for backward compatibility; default = "none".
        modifier = str(row.get("modifier", "none")).lower()

        if len(key) != 1:
            raise ValueError(f"{path} row {i}: 'key' must be a single character, got {row['key']!r}")
        if modifier not in ALLOWED_MODIFIERS:
            raise ValueError(
                f"{path} row {i}: 'modifier' must be one of {ALLOWED_MODIFIERS}, got {modifier!r}"
            )
        if arm not in config.ARM_SIDES:
            raise ValueError(f"{path} row {i}: 'arm' must be one of {config.ARM_SIDES}, got {arm!r}")
        if joint not in config.JOINT_NAMES:
            raise ValueError(f"{path} row {i}: 'joint' must be one of {config.JOINT_NAMES}, got {joint!r}")
        if direction not in (-1, 1):
            raise ValueError(f"{path} row {i}: 'direction' must be -1 or +1, got {direction}")

        lookup_key = (key, modifier)
        if lookup_key in result:
            raise ValueError(
                f"{path} row {i}: duplicate binding for key={key!r} modifier={modifier!r}"
            )

        result[lookup_key] = Binding(
            key=key, modifier=modifier, arm=arm, joint=joint,
            direction=direction, note=str(row.get("note", "")),
        )

    return result
