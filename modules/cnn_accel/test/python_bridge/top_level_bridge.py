"""VUnit python_bridge entry point for tb_cnn_accel_top: seeds DDR and
verifies each run entirely live, over VUnit's Python FFI (`python_call`),
from inside the running simulation. No file is read or written anywhere
in this path -- not `mem_image.csv` going in, nor `result.csv`/
`counters.csv` coming out.

Loaded into the simulator's embedded Python interpreter via
`python_execute(file_name => ...)`, once, before the run starts (so the
`_CASE` lookup below is populated before any other function here is
called). Named like a small object's public interface: `set_test_case`
picks the object, everything else is a `get_*` reading one fact off it,
except `check_result`, the one call that judges rather than reports."""

import sys
from pathlib import Path

import numpy as np

_CNN_ACCEL_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CNN_ACCEL_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_ACCEL_DIR))

from accel_v2 import (  # noqa: E402
    cases,
    cases_concat_split,
    cases_conv_pad,
    cases_error,
    cases_pool_pad,
    cases_tiling,
    cases_yolo,
)

# The same seven catalogues `module_cnn_accel.py`'s `_setup_cnn_accel_top`
# registers configs from, in one place so the two stay in sync by
# construction rather than by two lists someone has to remember to edit
# together.
_CATALOGUES = (
    cases,
    cases_pool_pad,
    cases_conv_pad,
    cases_concat_split,
    cases_yolo,
    cases_error,
    cases_tiling,
)

# Set by `set_test_case`, called once from VHDL right after
# `python_execute`-ing this file, naming which case's pre-built `TbCase`
# object every `get_*`/`check_result` call below reads from. One case per
# simulation process (VUnit forks a fresh process per config), so a
# single module-level variable is enough -- no session/threading concern.
_CASE = None


def set_test_case(name):
    """Select the case every later call in this module reads from.
    Raises (loudly, via the bridge's own exception-to-VHDL-failure path)
    if `name` is not found in any of `_CATALOGUES` -- a typo here must
    not silently check nothing."""
    global _CASE
    by_name = {case.name: case for catalogue in _CATALOGUES for case in catalogue.all_cases()}
    if name not in by_name:
        raise KeyError(f"top_level_bridge.set_test_case: no case named {name!r}")
    _CASE = by_name[name]
    # Called as VHDL's 'integer'-returning python_call overload (there is
    # no argument-taking, return-nothing overload) -- the return value
    # itself carries no meaning, only the call (and its exception path).
    return 0


def _require_test_case(caller: str) -> None:
    if _CASE is None:
        raise RuntimeError(f"top_level_bridge.{caller}: set_test_case was never called")


def get_program_start_address():
    """The selected case's program entry address (`TbCase.program.
    program_addr`), written to `PROGRAM_BASE_ADDR`."""
    _require_test_case("get_program_start_address")
    return _CASE.program.program_addr


def get_output_region():
    """`[base, num_bytes]` of the DDR window `check_result` reads back
    and checks after the run (`TbCase.export_base`/`export_bytes`)."""
    _require_test_case("get_output_region")
    return [_CASE.export_base, _CASE.export_bytes]


def get_input_region():
    """`[base, num_bytes]` of the DDR window `get_input_data` seeds
    before the run (`TbCase.input_region`)."""
    _require_test_case("get_input_region")
    return list(_CASE.input_region())


def get_expect_error():
    """Whether the selected case expects `STATUS.ERROR` (`TbCase.
    expect_error`) -- the testbench's own one pass/fail claim."""
    _require_test_case("get_expect_error")
    return _CASE.expect_error


def get_input_data():
    """The selected case's graph-input tensor bytes, covering the window
    `get_input_region` describes (`TbCase.input_bytes`). Same bytes the
    compiler's own `emit_program` already produced -- this only changes
    how they reach the simulated DDR."""
    _require_test_case("get_input_data")
    return np.frombuffer(_CASE.input_bytes(), dtype=np.uint8)


def get_program_regions():
    """`[addr0, len0, addr1, len1, ...]` -- the selected case's compiled
    program layout (`TbCase.compiled_regions`: the descriptor chain and
    the weight/bias/scale/LUT tables, the input window excluded),
    flattened so one `python_call` hands the whole layout over at once.
    VHDL loops over `length(bounds) / 2` regions, calling
    `get_program_data(index)` for each one's content."""
    _require_test_case("get_program_regions")
    flat = [value for region in _CASE.compiled_regions() for value in region]
    return np.asarray(flat, dtype=np.int32)


def get_program_data(index):
    """The bytes of program region `index` (see `get_program_regions`)."""
    _require_test_case("get_program_data")
    return np.frombuffer(_CASE.compiled_region_bytes(int(index)), dtype=np.uint8)


def check_result(
    export_bytes,
    export_base,
    status,
    busy,
    done,
    error,
    err_code,
    err_pc_low,
    hw_info,
    hw_info2,
    hw_info3,
    cmd_count,
    cycle_count,
    compute_cycles,
    stall_cycles,
    ddr_rd_bytes,
    ddr_wr_bytes,
    tensor_load_count,
    tensor_store_count,
    weight_load_bytes,
    local_bytes,
    axi_ar_count,
    axi_aw_count,
    axi_rd_beats,
    axi_wr_beats,
    axi_wr_bytes,
    axi_wr_lo_addr,
    axi_wr_hi_addr,
):
    """Called from tb_cnn_accel_top's main process right after
    STATUS.DONE/STATUS.ERROR. `export_bytes` is the raw exported DDR
    region as a VUnit integer_array_t of UNSIGNED byte values (0..255),
    read straight out of the simulator's `memory_t` model via
    `read_word`. Every other argument is one counter register/passive-
    monitor value, read straight out of the DUT/testbench -- the exact
    same set `TbCase.check_live` expects. Returns True/False for VHDL to
    `check_true` on."""
    _require_test_case("check_result")

    counters = {
        "status": int(status),
        "busy": int(busy),
        "done": int(done),
        "error": int(error),
        "err_code": int(err_code),
        "err_pc_low": int(err_pc_low),
        "hw_info": int(hw_info),
        "hw_info2": int(hw_info2),
        "hw_info3": int(hw_info3),
        "cmd_count": int(cmd_count),
        "cycle_count": int(cycle_count),
        "compute_cycles": int(compute_cycles),
        "stall_cycles": int(stall_cycles),
        "ddr_rd_bytes": int(ddr_rd_bytes),
        "ddr_wr_bytes": int(ddr_wr_bytes),
        "tensor_load_count": int(tensor_load_count),
        "tensor_store_count": int(tensor_store_count),
        "weight_load_bytes": int(weight_load_bytes),
        "local_bytes": int(local_bytes),
        "axi_ar_count": int(axi_ar_count),
        "axi_aw_count": int(axi_aw_count),
        "axi_rd_beats": int(axi_rd_beats),
        "axi_wr_beats": int(axi_wr_beats),
        "axi_rd_bytes": int(axi_rd_beats) * 8,
        "axi_wr_bytes": int(axi_wr_bytes),
        "axi_wr_lo_addr": int(axi_wr_lo_addr),
        "axi_wr_hi_addr": int(axi_wr_hi_addr),
    }
    return _CASE.check_live(counters, int(export_base), [int(b) for b in export_bytes])
