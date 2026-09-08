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


def v10_target(target: Target) -> Target:
    """A synthetic ISA v1.0 (pre-H1) variant of `target`: no `output_offset`/
    `clamp_min`/`clamp_max` ISA fields, no `CLAMP_EN` flag, `output_zp:
    false` and the two legacy clamp ranges only. Built via
    `Target.to_dict`/`from_dict` so it never touches `modules/cnn_accel/`.
    The real target has been ISA v1.1 since HW milestone H1 landed; this
    keeps the MVP rejection/unfused paths (doc/tosa_compiler_plan.md §9)
    covered."""
    data = target.to_dict()
    for unit in data["units"]:
        unit["isa_version"] = "1.0"
        unit["epilogue"]["rescale"]["output_zp"] = False
        unit["epilogue"]["clamp_ranges"] = [[-128, 127], [0, 127]]
    isa = data["isa"]
    for name in ("output_offset", "clamp_min", "clamp_max"):
        isa["fields"].pop(name, None)
    isa["flags"].pop("CLAMP_EN", None)
    return Target.from_dict(data)


def half_even_target(target: Target) -> Target:
    """A synthetic variant of `target` with every unit's rescale rounding
    forced to `half_even`, built via `Target.from_dict`/`to_dict` so it
    never touches `modules/cnn_accel/`. Used to keep the rounding-gate
    rejection path covered now that the real target is half-up."""
    data = target.to_dict()
    for unit in data["units"]:
        unit["epilogue"]["rescale"]["rounding"] = "half_even"
    return Target.from_dict(data)
