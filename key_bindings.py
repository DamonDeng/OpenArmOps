"""Loader for ``key_bindings.json``.

Kept tiny so the UI can reload the file at runtime without restarting. Schema
validation is minimal — enough to produce a useful error message, no more.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Binding:
    key: str
    arm: str
    joint: str
    direction: int
    note: str = ""


def load_bindings(path: Path | str = config.DEFAULT_KEY_BINDINGS_PATH) -> dict[str, Binding]:
    """Parse the JSON file. Returns {lowercase_key_char: Binding}.

    Raises ValueError with a line-level message if the JSON is malformed or a
    row fails basic validation. We return the dict rather than a list because
    the UI's Qt eventFilter needs O(1) lookup per keystroke.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("bindings")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: top-level key 'bindings' must be a list")

    result: dict[str, Binding] = {}
    for i, row in enumerate(rows):
        try:
            key = str(row["key"]).lower()
            arm = row["arm"]
            joint = row["joint"]
            direction = int(row["direction"])
        except (KeyError, TypeError) as e:
            raise ValueError(f"{path} row {i}: missing/invalid field ({e})") from e

        if len(key) != 1:
            raise ValueError(f"{path} row {i}: 'key' must be a single character, got {row['key']!r}")
        if arm not in config.ARM_SIDES:
            raise ValueError(f"{path} row {i}: 'arm' must be one of {config.ARM_SIDES}, got {arm!r}")
        if joint not in config.JOINT_NAMES:
            raise ValueError(f"{path} row {i}: 'joint' must be one of {config.JOINT_NAMES}, got {joint!r}")
        if direction not in (-1, 1):
            raise ValueError(f"{path} row {i}: 'direction' must be -1 or +1, got {direction}")
        if key in result:
            raise ValueError(f"{path} row {i}: duplicate binding for key {key!r}")

        result[key] = Binding(
            key=key, arm=arm, joint=joint, direction=direction,
            note=str(row.get("note", "")),
        )

    return result
