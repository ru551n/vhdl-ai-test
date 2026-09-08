"""M0 tests: `cnnc.target`'s Target contract, constraint evaluator, and
`cnn_accel` discovery (proving derivation from the HW source of truth,
not hard-coded compiler constants)."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cnnc.target import CapabilityError, TargetError, load_target
from cnnc.target.constraints import ConstraintEvalError, ConstraintViolation, check, evaluate
from cnnc.target.contract import Constraint, Target
from cnnc.target.discover import discover_cnn_accel

REPO_ROOT = Path(__file__).resolve().parents[2]
ACCEL_ROOT = REPO_ROOT / "modules" / "cnn_accel"
COMPILER_ROOT = REPO_ROOT / "compiler"


def _load_accel_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ACCEL_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cnn_accel_constants():
    return _load_accel_module("cnn_accel_constants")


@pytest.fixture(scope="module")
def cnn_accel_model():
    return _load_accel_module("cnn_accel_model")


# ---------------------------------------------------------------------------
# Discovery matches the accelerator's own source of truth.
# ---------------------------------------------------------------------------


def test_load_target_succeeds():
    target = load_target("cnn_accel")
    assert target.name == "cnn_accel_v1"
    assert target.unit("conv_engine").ops == ("conv2d",)

    alias = load_target("cnn_accel_v1")
    assert alias == target


def test_derived_values_match_constants_directly(cnn_accel_constants):
    target = load_target("cnn_accel")
    unit = target.unit("conv_engine")

    assert unit.internal_tiling.cout == cnn_accel_constants.PE_ROWS
    assert unit.internal_tiling.cin == cnn_accel_constants.TILE_CHANNELS
    assert unit.dtypes["accumulator"] == f"i{cnn_accel_constants.ACCUM_WIDTH}"

    divisible = next(c for c in unit.constraints if c.kind == "divisible")
    assert divisible.by == cnn_accel_constants.PE_ROWS

    row_tile = next(c for c in unit.constraints if "MAX_ROW_TILE_WORDS" in c.source)
    assert row_tile.value == cnn_accel_constants.MAX_ROW_TILE_WORDS

    weight_buf = next(c for c in unit.constraints if "WEIGHT_BUFFER_DEPTH" in c.source)
    assert weight_buf.value == cnn_accel_constants.WEIGHT_BUFFER_DEPTH

    kernel_max = {c.expr for c in unit.constraints if c.value == cnn_accel_constants.MAX_KERNEL_SIZE}
    assert {"kernel_h", "kernel_w"} <= kernel_max


def test_rounding_and_implicit_shift_are_discovered_not_hardcoded(cnn_accel_model):
    target = load_target("cnn_accel")
    rescale = target.unit("conv_engine").epilogue.rescale

    expected_rounding = "half_even" if cnn_accel_model.round_shift_right_signed(1, 1) == 0 else "half_up"
    assert rescale.rounding == expected_rounding

    # Pins the current cnn_accel fixed-point contract (Q15 requant_scale, no
    # extra implicit shift beyond that). This is a deliberate hard assertion,
    # not a tautology against the model: if the HW/model's scale convention
    # ever changes, this must fail loudly rather than silently track it.
    assert rescale.implicit_shift == 15


def test_isa_v12_per_channel_is_discovered_from_constants(cnn_accel_constants):
    # HW milestone H2 (ISA v1.2): `scale_addr` (W14) and PER_CHANNEL_EN are
    # discovered off cnn_accel_constants, never hard-coded -- and the
    # version string follows from the fields present.
    target = load_target("cnn_accel")
    unit = target.unit("conv_engine")
    assert unit.isa_version == "1.2"
    assert unit.epilogue.rescale.per_channel is True
    expected = next(f for f in cnn_accel_constants.isa_field_offsets() if f.name == "scale_addr")
    assert target.isa.fields["scale_addr"] == (expected.offset_bytes, expected.width_bytes, expected.signed)
    assert expected.width_bytes == 4 and expected.signed is False
    assert expected.offset_bytes == 14 * 4  # W14
    assert target.isa.flags["PER_CHANNEL_EN"] == cnn_accel_constants.FLAGS["PER_CHANNEL_EN"]


def test_discovery_has_no_hardcoded_accelerator_properties(tmp_path):
    (tmp_path / "cnn_accel_model.py").write_text((ACCEL_ROOT / "cnn_accel_model.py").read_text())

    constants_src = (ACCEL_ROOT / "cnn_accel_constants.py").read_text()
    patched = constants_src.replace("PE_ROWS = 8\n", "PE_ROWS = 16\n", 1)
    # PE_ROWS_SCALED must stay a *distinct* member of PE_ROWS_LEGAL (its own
    # elaboration-time assert), so swap it to the other legal value.
    patched = patched.replace("PE_ROWS_SCALED = 16\n", "PE_ROWS_SCALED = 8\n", 1)
    patched = patched.replace("MAX_ROW_TILE_WORDS = 512\n", "MAX_ROW_TILE_WORDS = 1024\n", 1)
    assert patched != constants_src
    (tmp_path / "cnn_accel_constants.py").write_text(patched)

    fake_target = discover_cnn_accel(tmp_path)
    real_target = discover_cnn_accel()

    fake_unit = fake_target.unit("conv_engine")
    real_unit = real_target.unit("conv_engine")

    assert fake_unit.internal_tiling.cout == 16
    assert real_unit.internal_tiling.cout == 8

    fake_divisible = next(c for c in fake_unit.constraints if c.kind == "divisible")
    real_divisible = next(c for c in real_unit.constraints if c.kind == "divisible")
    assert fake_divisible.by == 16
    assert real_divisible.by == 8

    fake_row_tile = next(c for c in fake_unit.constraints if "MAX_ROW_TILE_WORDS" in c.source)
    real_row_tile = next(c for c in real_unit.constraints if "MAX_ROW_TILE_WORDS" in c.source)
    assert fake_row_tile.value == 1024
    assert real_row_tile.value == 512


# ---------------------------------------------------------------------------
# Constraint expression evaluator.
# ---------------------------------------------------------------------------


def test_evaluate_precedence():
    assert evaluate("2 + 3 * 4", {}) == 14


def test_evaluate_parentheses_override_precedence():
    assert evaluate("(2 + 3) * 4", {}) == 20


def test_evaluate_ceil_true_division():
    assert evaluate("ceil(in_channels / TILE_CHANNELS)", {"in_channels": 20, "TILE_CHANNELS": 8}) == 3


def test_evaluate_ceil_composed_with_multiplication():
    env = {"in_width": 10, "in_channels": 17, "TILE_CHANNELS": 8}
    assert evaluate("in_width * ceil(in_channels / TILE_CHANNELS)", env) == 30


def test_evaluate_exact_division_outside_ceil():
    assert evaluate("out_channels / PE_ROWS", {"out_channels": 16, "PE_ROWS": 8}) == 2


def test_check_divisible_constraint():
    constraint = Constraint(kind="divisible", expr="out_channels", by=8, source="test")
    assert check(constraint, {"out_channels": 16}) is None
    violation = check(constraint, {"out_channels": 17})
    assert isinstance(violation, ConstraintViolation)
    assert violation.actual == 17
    assert violation.constraint is constraint


def test_evaluate_unknown_identifier_raises():
    with pytest.raises(ConstraintEvalError):
        evaluate("foo + 1", {})


def test_evaluate_bad_token_raises():
    with pytest.raises(ConstraintEvalError):
        evaluate("2 + @", {})


def test_evaluate_inexact_division_outside_ceil_raises():
    with pytest.raises(ConstraintEvalError):
        evaluate("7 / 2", {})


# ---------------------------------------------------------------------------
# JSON round trip / validation.
# ---------------------------------------------------------------------------


def test_json_round_trip():
    target = load_target("cnn_accel")
    round_tripped = target.from_json(target.to_json())
    assert round_tripped == target


def _baseline_dict() -> dict:
    return copy.deepcopy(load_target("cnn_accel").to_dict())


def test_from_json_missing_field_raises():
    data = _baseline_dict()
    del data["name"]
    with pytest.raises(TargetError):
        Target.from_json(json.dumps(data))


def test_from_json_unknown_constraint_kind_raises():
    data = _baseline_dict()
    data["units"][0]["constraints"][0]["kind"] = "bogus"
    with pytest.raises(TargetError):
        Target.from_json(json.dumps(data))


def test_from_json_bad_rounding_raises():
    data = _baseline_dict()
    data["units"][0]["epilogue"]["rescale"]["rounding"] = "bogus"
    with pytest.raises(TargetError):
        Target.from_json(json.dumps(data))


def test_load_target_unknown_name_raises():
    with pytest.raises(TargetError):
        load_target("does_not_exist")


def test_capability_error_is_exported_and_is_an_exception():
    assert issubclass(CapabilityError, Exception)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_dump_produces_parseable_json():
    result = subprocess.run(
        [sys.executable, "-m", "cnnc.target", "dump", "cnn_accel"],
        cwd=COMPILER_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["name"] == "cnn_accel_v1"
