"""Derive a `Target` for `cnn_accel` from its own source of truth
(`modules/cnn_accel/cnn_accel_constants.py`, `cnn_accel_model.py`) -- never
by hard-coding accelerator properties in the compiler. Every number in the
returned `Target` traces back to one of those two files; see the
`provenance` field for the mapping and `flow_status.md`/`AGENTS.md`
("`cnn_accel` constants are generated, not hand-written") for why that
file is authoritative.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

from .contract import (
    Constraint,
    Epilogue,
    InternalTiling,
    IsaInfo,
    Memory,
    MemorySpace,
    RescaleCaps,
    Target,
    TargetError,
    Unit,
)

_CONSTANTS_MODULE_NAME = "cnn_accel_constants"
_MODEL_MODULE_NAME = "cnn_accel_model"


def _resolve_accel_root() -> Path:
    env_root = os.environ.get("CNNC_ACCEL_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[3] / "modules" / "cnn_accel"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import_fresh(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TargetError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise TargetError(f"failed importing {path}: {exc}") from exc
    return module


def _probe_implicit_shift(bias_requantize_relu) -> int:
    """Smallest `n` in `0..40` such that scaling `acc=1` by `2**n` (with
    no bias/relu) rounds to exactly `1` *and* scaling `acc=-1` by the same
    `2**n` rounds to exactly `-1`: the point at which the model's internal
    fixed-point scale denominator equals `2**n`, i.e. the ISA's implicit
    (undocumented-as-a-number) shift. Not read from any constant --
    purely behavioral, so it fails loudly if the RTL/model contract ever
    changes it.

    Checking both signs matters: at `n = shift - 1` the scaled value is
    an exact tie (`0.5` in magnude), and a "round ties away from zero"
    (half-up) convention rounds `acc=1` to `1` there too -- one step
    before the real shift. That tie is *not* symmetric (`acc=-1` rounds
    to `0`, not `-1`, under half-up), so requiring both `+1 -> 1` and
    `-1 -> -1` (both exact, non-tie divisions at the true shift) rejects
    that false positive regardless of which rounding convention the RTL
    implements.
    """
    previous = None
    for n in range(41):
        result = bias_requantize_relu(
            1, 0, bias_en=False, requant_en=True, relu_en=False, requant_scale=1 << n, requant_shift=0
        )
        negated = bias_requantize_relu(
            -1, 0, bias_en=False, requant_en=True, relu_en=False, requant_scale=1 << n, requant_shift=0
        )
        if previous is not None and result not in (0, 1) and previous not in (0, 1):
            raise TargetError("implicit_shift probe: non-monotonic response, cannot discover shift")
        if result == 1 and negated == -1:
            return n
        previous = result
    raise TargetError("implicit_shift probe: no n in 0..40 rounds acc=1 to 1 and acc=-1 to -1")


def _probe_rounding(bias_requantize_relu, implicit_shift: int) -> str:
    half_scale = 1 << (implicit_shift - 1)

    def f(acc: int) -> int:
        return bias_requantize_relu(
            acc, 0, bias_en=False, requant_en=True, relu_en=False, requant_scale=half_scale, requant_shift=0
        )

    samples = {value: f(value) for value in (1, 3, 5, -1, -5, -3)}
    half_up = {1: 1, 3: 2, 5: 3, -1: 0, -5: -2, -3: -1}
    half_even = {1: 0, 3: 2, 5: 2, -1: 0, -5: -2, -3: -2}
    if samples == half_up:
        return "half_up"
    if samples == half_even:
        return "half_even"
    raise TargetError(f"unrecognised rounding: tie-break probe samples {samples}")


def discover_cnn_accel(accel_root: Path | None = None) -> Target:
    root = Path(accel_root) if accel_root is not None else _resolve_accel_root()
    constants_path = root / "cnn_accel_constants.py"
    model_path = root / "cnn_accel_model.py"
    if not constants_path.is_file():
        raise TargetError(f"cnn_accel_constants.py not found under {root}")
    if not model_path.is_file():
        raise TargetError(f"cnn_accel_model.py not found under {root}")

    saved_modules = {
        name: sys.modules.get(name) for name in (_CONSTANTS_MODULE_NAME, _MODEL_MODULE_NAME)
    }
    try:
        constants = _import_fresh(_CONSTANTS_MODULE_NAME, constants_path)
        model = _import_fresh(_MODEL_MODULE_NAME, model_path)
        return _build_target(root, constants, model, constants_path, model_path)
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _build_target(root: Path, constants, model, constants_path: Path, model_path: Path) -> Target:
    field_specs = {
        f.name: (f.offset_bytes, f.width_bytes, f.signed) for f in constants.isa_field_offsets()
    }

    def field_width_bytes(name: str) -> int:
        return field_specs[name][1]

    def field_signed(name: str) -> bool:
        return field_specs[name][2]

    def field_max_by_width(name: str) -> int:
        return 2 ** (8 * field_width_bytes(name)) - 1

    implicit_shift = _probe_implicit_shift(model.bias_requantize_relu)
    rounding = _probe_rounding(model.bias_requantize_relu, implicit_shift)

    isa_fields = set(field_specs)
    has_output_offset = "output_offset" in isa_fields
    has_scale_addr = "scale_addr" in isa_fields
    isa_version = "1.0"
    if has_output_offset:
        isa_version = "1.1"
        if has_scale_addr:
            isa_version = "1.2"

    per_channel = "PER_CHANNEL_EN" in constants.FLAGS
    clamp_en = "CLAMP_EN" in constants.FLAGS
    relu_en = "RELU_EN" in constants.FLAGS
    if clamp_en:
        clamp_ranges: object = "any"
    elif relu_en:
        clamp_ranges = ((-128, 127), (0, 127))
    else:
        clamp_ranges = ((-128, 127),)

    rescale = RescaleCaps(
        multiplier_bits=8 * field_width_bytes("requant_scale"),
        multiplier_signed=field_signed("requant_scale"),
        implicit_shift=implicit_shift,
        shift_min=implicit_shift,
        shift_max=implicit_shift + field_max_by_width("requant_shift"),
        rounding=rounding,
        per_channel=per_channel,
        input_zp=False,
        output_zp=has_output_offset,
    )
    epilogue = Epilogue(bias="BIAS_EN" in constants.FLAGS, rescale=rescale, clamp_ranges=clamp_ranges)

    kernels = tuple(
        (kh, kw)
        for kh in range(1, constants.MAX_KERNEL_SIZE + 1)
        for kw in range(1, constants.MAX_KERNEL_SIZE + 1)
    )

    def max_constraint(expr: str, value: int, source: str) -> Constraint:
        return Constraint(kind="max", expr=expr, value=value, source=source)

    constraints = [
        Constraint(
            kind="divisible",
            expr="out_channels",
            by=constants.PE_ROWS,
            source="cnn_accel_constants.PE_ROWS: output channels are computed PE_ROWS lanes at a time",
        ),
        max_constraint(
            f"in_width * ceil(in_channels / {constants.TILE_CHANNELS})",
            constants.MAX_ROW_TILE_WORDS,
            "cnn_accel_constants.MAX_ROW_TILE_WORDS, sizes cnn_accel_window_gen's "
            "g_max_row_tile_words row-bank depth (doc/cnn_accel_window_gen.md)",
        ),
        max_constraint(
            f"kernel_h * kernel_w * ceil(in_channels / {constants.TILE_CHANNELS})",
            constants.WEIGHT_BUFFER_DEPTH,
            "cnn_accel_constants.WEIGHT_BUFFER_DEPTH",
        ),
    ]
    for name in ("in_width", "in_height", "in_channels", "out_channels"):
        constraints.append(
            max_constraint(
                name,
                field_max_by_width(name),
                f"cnn_accel_constants.ISA_LAYOUT field {name!r} width ({field_width_bytes(name)} bytes)",
            )
        )
    for name in ("pad_top", "pad_bottom", "pad_left", "pad_right"):
        constraints.append(
            max_constraint(
                name,
                field_max_by_width(name),
                f"cnn_accel_constants.ISA_LAYOUT field {name!r} width ({field_width_bytes(name)} bytes); "
                "doc/cnn_accel_arch.md W8 ('zero-padding, valid when pad_en=1')",
            )
        )
    for name in ("kernel_h", "kernel_w"):
        constraints.append(
            max_constraint(
                name,
                constants.MAX_KERNEL_SIZE,
                "cnn_accel_constants.MAX_KERNEL_SIZE, enforced by cnn_accel_window_gen's "
                "g_max_kernel_size elaboration-time assert (doc/cnn_accel_window_gen.md)",
            )
        )
    for name in ("stride_h", "stride_w"):
        constraints.append(
            max_constraint(
                name,
                field_max_by_width(name),
                f"cnn_accel_constants.ISA_LAYOUT field {name!r} width ({field_width_bytes(name)} bytes); "
                "cnn_accel_window_gen imposes no elaboration-time bound on stride (unlike "
                "g_max_kernel_size for kernel) -- only 1x1/2x2 are exercised today by "
                "tb_cnn_accel_window_gen's test_kernel_stride_shapes (doc/cnn_accel_window_gen.md "
                "'Verification notes')",
            )
        )

    unit = Unit(
        name="conv_engine",
        ops=("conv2d",),
        dtypes={
            "input": "i8",
            "weight": "i8",
            "bias": "i32",
            "accumulator": f"i{constants.ACCUM_WIDTH}",
            "output": "i8",
        },
        kernels=kernels,
        strides="any",
        dilation=((1, 1),),
        padding="zero",
        batch=1,
        internal_tiling=InternalTiling(cin=constants.TILE_CHANNELS, cout=constants.PE_ROWS),
        constraints=tuple(constraints),
        epilogue=epilogue,
        isa_version=isa_version,
        partial_sum_io="PSUM_IN" in constants.FLAGS,
    )

    ddr_size = 2 ** (8 * field_width_bytes("in_addr"))
    memory = Memory(
        spaces={
            "ddr": MemorySpace(
                name="ddr",
                size_bytes=ddr_size,
                align=64,
            )
        },
        activation_layout="HWC",
        weight_layout="OHWI",
        bias_format="i32_le",
    )

    isa = IsaInfo(
        instr_word_bytes=constants.INSTR_WORD_BYTES,
        opcodes=dict(constants.OPCODES),
        flags=dict(constants.FLAGS),
        fields=field_specs,
    )

    provenance = {
        "accel_root": str(root),
        "files": {
            "cnn_accel_constants.py": _sha256(constants_path),
            "cnn_accel_model.py": _sha256(model_path),
        },
        "notes": [
            "PE_ROWS/PE_COLS/TILE_CHANNELS/MAX_KERNEL_SIZE/MAX_ROW_TILE_WORDS/"
            "WEIGHT_BUFFER_DEPTH/BIAS_BUFFER_DEPTH/ACCUM_WIDTH read directly from "
            "cnn_accel_constants.py; changing them there changes this Target.",
            "kernels enumerated for 1..MAX_KERNEL_SIZE in both dims (square and "
            "non-square) per doc/cnn_accel_window_gen.md 'Verification notes'.",
            "strides modelled as 'any', bounded only by the 1-byte stride_h/stride_w "
            "ISA field width (see the stride_h/stride_w constraints' own source "
            "note): the RTL has no elaboration-time stride bound, but only 1x1/2x2 "
            "are regression-tested today.",
            "memory.spaces['ddr'].size_bytes derived from the in_addr ISA field "
            "width (cnn_accel_constants.ISA_LAYOUT); .align=64 is a compiler burst "
            "alignment POLICY (AXI-burst-friendly), not a value read from the HW.",
            "activation_layout/weight_layout/bias_format taken from cnn_accel_model.py's "
            "module docstring ('Tensor layout conventions').",
            f"epilogue.rescale.implicit_shift={implicit_shift} and "
            f"rounding={rounding!r} were discovered by probing "
            "cnn_accel_model.bias_requantize_relu, not hard-coded.",
        ],
    }

    return Target(
        name="cnn_accel_v1",
        program_model="instruction_stream",
        memory=memory,
        units=(unit,),
        isa=isa,
        provenance=provenance,
    )
