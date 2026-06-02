"""Save / load runtime-tunable motion settings.

Scope today: max-speed + gravity-comp-scale. Extensible — add a new
``RuntimeState`` field, mention it in ``PERSISTED_FIELDS``, it'll be
saved and loaded with the next file round-trip.

File: ``~/.openarm_ui_config/motion_settings.json`` (path from config.py).
Format: flat JSON object, one entry per persisted field. Extra fields in
the file are ignored (forward-compat if a newer version wrote more);
missing fields keep the RuntimeState's current value (so loading an old
file into a newer app doesn't reset new knobs to zero).

Strict parse: malformed JSON raises — the caller decides what to show
the user. No partial loads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields as dataclass_fields
from pathlib import Path

from . import config
from .runtime_state import RuntimeState

logger = logging.getLogger(__name__)


# Fields we persist to / restore from disk. Adding a field here is the
# only step needed to get it saved and loaded — the code below walks
# this tuple for both directions. Kept as a small explicit list (rather
# than "everything on RuntimeState") so we can choose which knobs are
# truly user-preferences vs transient state.
PERSISTED_FIELDS: tuple[str, ...] = (
    "max_speed_deg_per_sec",
    "max_speed_deg_per_sec_gripper",
    "gravity_comp_scale",
    "vr_pos_scale",
    "vr_rot_scale",
    "vr_receiver_backend",
    "ik_boundary_fallback_enabled",
)


def save_session(state: RuntimeState, path: Path | str = config.SESSION_CONFIG_PATH) -> Path:
    """Dump the persisted RuntimeState fields to ``path`` as JSON.

    Creates the parent directory if it doesn't exist. Returns the written
    path so the caller can show it in a status line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: getattr(state, name) for name in PERSISTED_FIELDS}
    # Pretty-print so users who open the file with a text editor get
    # something readable. Two-space indent matches key_bindings.json.
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Saved session config to {path}: {payload}")
    return path


def load_session(
    state: RuntimeState,
    path: Path | str = config.SESSION_CONFIG_PATH,
) -> dict[str, object]:
    """Load fields from ``path`` and apply them to ``state``.

    Returns the dict of fields actually applied (useful for status text
    and logging). Raises FileNotFoundError if the file is missing —
    callers who want "silently do nothing" should check path.exists()
    first. Raises ValueError on malformed JSON or on a field whose type
    doesn't match what RuntimeState expects.

    Unknown fields in the file are silently ignored (forward-compat).
    Missing fields keep ``state``'s current value (no reset).
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: malformed JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level value must be an object")

    applied: dict[str, object] = {}
    # Build a map of RuntimeState's field types so we can coerce/validate.
    type_map = {f.name: f.type for f in dataclass_fields(state)}

    for name in PERSISTED_FIELDS:
        if name not in data:
            continue  # keep current value
        value = data[name]
        expected_type = type_map.get(name, None)
        # Coerce permissively for the trivial types we persist today.
        # ``from __future__ import annotations`` turns dataclass field
        # types into string forms, so compare by name.
        type_name = expected_type if isinstance(expected_type, str) else getattr(expected_type, "__name__", "")
        if type_name == "float":
            try:
                value = float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{path}: field '{name}' must be a number, got {value!r}"
                ) from e
        elif type_name == "str":
            if not isinstance(value, str):
                raise ValueError(
                    f"{path}: field '{name}' must be a string, got {value!r}"
                )
        elif type_name == "bool":
            if not isinstance(value, bool):
                raise ValueError(
                    f"{path}: field '{name}' must be true/false, got {value!r}"
                )
        setattr(state, name, value)
        applied[name] = value

    logger.info(f"Loaded session config from {path}: {applied}")
    return applied


def reset_to_defaults(state: RuntimeState) -> dict[str, object]:
    """Restore all persisted fields to their compile-time defaults
    (``config.INITIAL_*``). Does not touch the file — caller can
    follow up with save_session() if they want the file to match.
    Returns the dict of fields actually reset.
    """
    defaults = {
        "max_speed_deg_per_sec": config.INITIAL_MAX_SPEED_DEG_PER_SEC,
        "max_speed_deg_per_sec_gripper": config.INITIAL_MAX_SPEED_DEG_PER_SEC_GRIPPER,
        "gravity_comp_scale": config.INITIAL_GRAVITY_COMP_SCALE,
        "vr_pos_scale": config.INITIAL_VR_POS_SCALE,
        "vr_rot_scale": config.INITIAL_VR_ROT_SCALE,
        "vr_receiver_backend": config.VR_RECEIVER_BACKEND,
        "ik_boundary_fallback_enabled": config.INITIAL_IK_BOUNDARY_FALLBACK_ENABLED,
    }
    applied: dict[str, object] = {}
    for name in PERSISTED_FIELDS:
        if name in defaults:
            setattr(state, name, defaults[name])
            applied[name] = defaults[name]
    logger.info(f"Reset session config to defaults: {applied}")
    return applied
