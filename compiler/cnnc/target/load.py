"""Target loading: name/alias -> `Target`, plus a JSON dump helper."""

from __future__ import annotations

from pathlib import Path

from .contract import Target, TargetError
from .discover import discover_cnn_accel

_ALIASES = {
    "cnn_accel": discover_cnn_accel,
    "cnn_accel_v1": discover_cnn_accel,
}


def load_target(name: str) -> Target:
    if name.endswith(".json"):
        path = Path(name)
        if not path.is_file():
            raise TargetError(f"target JSON file not found: {path}")
        return Target.from_json(path.read_text())

    factory = _ALIASES.get(name)
    if factory is None:
        raise TargetError(f"unknown target {name!r} (known: {sorted(_ALIASES)}, or a path to a .json file)")
    return factory()


def dump_target(target: Target, path: str | Path) -> None:
    Path(path).write_text(target.to_json())
