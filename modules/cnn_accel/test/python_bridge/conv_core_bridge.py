"""VUnit python_bridge entry point for tb_cnn_accel_conv_core and
tb_cnn_accel_pe_array_from_vectors -- the last two cnn_accel testbenches
that read pre-generated vector files (`desc.txt`/`weights_packed.txt`/
`bias.txt`/`input.txt`/`expected.txt`, doc/cnn_accel_test_vectors.md) via
VHDL's own `file_open`/`read_int_file`. See `shared/Vunit.md`'s
"python_call and python_execute" section for the general pattern this
follows.

`generate_vectors.py`'s own case-authoring functions
(`generate_conv_core_cases`/`generate_pe_array_xlang_case`) are
deliberately left untouched -- they still write real files, into a
private `tempfile.mkdtemp()` scratch directory this module owns (never
the repository, never VUnit's own `output_path`), read straight back with
`generate_vectors.read_case_from_dir`. The compiler-generated cases
(`test_bitexact_compiler_cases`) instead point that same reader at
VUnit's `output_path`, where `module_cnn_accel.py`'s
`_compiler_vectors_pre_config` still writes them via `cnnc.backend.
cnn_accel_v1.vectors.write_conv_core_vectors`, unchanged -- both sources
produce the identical directory shape, so one reader and one bridge
serve either. Either way, this module is the *only* place left that
reads a `.txt` vector file at all: no VHDL testbench does its own file
I/O any more.

Selection functions (`select_hand_case`/`select_dir_case`/
`select_pe_array_case`) set which case the `get_*` functions below
report; call one before any `get_*` call, exactly like
`top_level_bridge.py`'s `set_test_case`/`_CASE`.
"""

import sys
import tempfile
from pathlib import Path

_CNN_ACCEL_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CNN_ACCEL_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_ACCEL_DIR))

import generate_vectors
import numpy as np

# Every LayerDesc field tb_cnn_accel_conv_core.vhd's run_case actually
# reads, in the fixed order get_desc_fields() returns them -- tile_channels/
# pe_rows (D10 packing parameters, not LayerDesc fields) are always
# appended after these, so a field added here only ever grows the array,
# never reshuffles an existing index.
_DESC_FIELD_ORDER = (
    "opcode", "flags", "in_width", "in_height", "in_channels", "out_channels",
    "kernel_h", "kernel_w", "stride_h", "stride_w",
    "pad_top", "pad_bottom", "pad_left", "pad_right", "pad_value",
    "requant_scale", "requant_shift", "output_offset", "clamp_min", "clamp_max",
)

# name -> ConvCoreCase, built once per simulation process (this test
# process's own g_pe_rows never changes mid-run) into a private scratch
# directory -- see this module's own docstring.
_hand_cases: dict[str, "generate_vectors.ConvCoreCase"] = {}
_pe_array_case: "generate_vectors.ConvCoreCase | None" = None

_case: "generate_vectors.ConvCoreCase | None" = None


def _build_hand_cases(pe_rows: int) -> None:
    global _hand_cases
    if _hand_cases:
        return
    scratch = Path(tempfile.mkdtemp(prefix="cnn_accel_conv_core_vectors_"))
    hw = generate_vectors.hw_packing(scratch, pe_rows=pe_rows)
    names = generate_vectors.generate_conv_core_cases(hw)
    _hand_cases = {name: generate_vectors.read_case_from_dir(scratch / name) for name in names}


def select_hand_case(name, pe_rows):
    """Select one of `generate_vectors.generate_conv_core_cases`'s
    hand-authored cases by name, at `pe_rows`'s packing point -- for
    tb_cnn_accel_conv_core.vhd's `run_all_cases`."""
    global _case
    _build_hand_cases(int(pe_rows))
    if name not in _hand_cases:
        raise KeyError(
            f"conv_core_bridge.select_hand_case: no case {name!r} at pe_rows={pe_rows} "
            f"(known: {sorted(_hand_cases)})"
        )
    _case = _hand_cases[name]
    return 0


def select_dir_case(case_dir):
    """Select a case already written to `case_dir` by someone else --
    module_cnn_accel.py's `_compiler_vectors_pre_config`, for
    tb_cnn_accel_conv_core.vhd's `run_compiler_cases`."""
    global _case
    _case = generate_vectors.read_case_from_dir(Path(case_dir))
    return 0


def select_pe_array_case(pe_rows):
    """Select the one `pe_array_xlang_check` case, at `pe_rows`'s packing
    point -- for tb_cnn_accel_pe_array_from_vectors.vhd."""
    global _case, _pe_array_case
    if _pe_array_case is None:
        scratch = Path(tempfile.mkdtemp(prefix="cnn_accel_pe_array_vectors_"))
        hw = generate_vectors.hw_packing(scratch, pe_rows=int(pe_rows))
        generate_vectors.generate_pe_array_xlang_case(hw)
        _pe_array_case = generate_vectors.read_case_from_dir(scratch / "pe_array_xlang_check")
    _case = _pe_array_case
    return 0


def _require_case(caller: str) -> "generate_vectors.ConvCoreCase":
    if _case is None:
        raise RuntimeError(f"conv_core_bridge.{caller}: no case selected yet")
    return _case


def get_desc_fields():
    """Flat `[opcode, flags, ..., clamp_max, tile_channels, pe_rows]`
    array (`_DESC_FIELD_ORDER` plus the two packing parameters) for the
    currently selected case."""
    c = _require_case("get_desc_fields")
    values = [c.desc[name] for name in _DESC_FIELD_ORDER] + [c.tile_channels, c.pe_rows]
    return np.array(values, dtype=np.int32)


def get_weights_packed_flat():
    return np.array(_require_case("get_weights_packed_flat").weights_packed, dtype=np.int32)


def get_bias_flat():
    return np.array(_require_case("get_bias_flat").bias, dtype=np.int32)


def get_input_flat():
    return np.array(_require_case("get_input_flat").input, dtype=np.int32)


def get_expected_flat():
    return np.array(_require_case("get_expected_flat").expected, dtype=np.int32)


def get_scale_table_packed_flat():
    c = _require_case("get_scale_table_packed_flat")
    if c.scale_table_packed is None:
        raise RuntimeError(
            "conv_core_bridge.get_scale_table_packed_flat: current case has no scale table"
        )
    return np.array(c.scale_table_packed, dtype=np.int32)


def get_raw_accum_flat():
    c = _require_case("get_raw_accum_flat")
    if c.raw_accum is None:
        raise RuntimeError("conv_core_bridge.get_raw_accum_flat: current case has no raw accumulator")
    return np.array(c.raw_accum, dtype=np.int32)
