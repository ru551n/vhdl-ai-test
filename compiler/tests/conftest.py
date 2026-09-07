"""pytest configuration: make `cnnc` importable from the repo checkout."""

import sys
from pathlib import Path

COMPILER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPILER_ROOT.parent

if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))

import pytest  # noqa: E402

from cnnc.target.contract import Target  # noqa: E402
from cnnc.target.load import load_target  # noqa: E402


@pytest.fixture(scope="session")
def target() -> Target:
    """The real `cnn_accel_v1` target, discovered from
    `modules/cnn_accel/` (now `rounding == "half_up"` since HW milestone
    H0 landed)."""
    return load_target("cnn_accel_v1")


def half_even_target(target: Target) -> Target:
    """A synthetic variant of `target` with every unit's rescale rounding
    forced to `half_even`, built via `Target.from_dict`/`to_dict` so it
    never touches `modules/cnn_accel/`. Used to keep the rounding-gate
    rejection path covered now that the real target is half-up."""
    data = target.to_dict()
    for unit in data["units"]:
        unit["epilogue"]["rescale"]["rounding"] = "half_even"
    return Target.from_dict(data)
