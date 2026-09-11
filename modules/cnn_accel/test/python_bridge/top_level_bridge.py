"""VUnit python_bridge entry point for the tb_cnn_accel_top LIVE-check
pilot: swap the post-simulation, CSV-file-based `post_check` for a
live call into the SAME verification code (`TbCase.check_live`, see
accel_v2/tbcase.py) from inside the running simulation, at the moment
STATUS.DONE fires.

Loaded into the simulator's embedded Python interpreter via
`python_execute(file_name => ...)`, once, before the run starts (so the
`_CASE` lookup below is populated before `check_live` is ever called).
"""

import sys
from pathlib import Path

import numpy as np

_CNN_ACCEL_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CNN_ACCEL_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_ACCEL_DIR))

from accel_v2 import cases  # noqa: E402

# Populated by `select_case`, called once from VHDL right after
# `python_execute`-ing this file, naming which case's pre-built `TbCase`
# object this run's `check_live` calls should use. One case per
# simulation process (VUnit forks a fresh process per config), so a
# single module-level variable is enough -- no session/threading concern.
_CASE = None


def select_case(name):
    """Bind `_CASE` to the named case's `TbCase` object, from the SAME
    catalogue `module_cnn_accel.py`'s `_setup_cnn_accel_top` registers
    configs from. Raises (loudly, via the bridge's own exception-to-VHDL-
    failure path) if `name` is not found -- a typo here must not silently
    check nothing."""
    global _CASE
    by_name = {case.name: case for case in cases.all_cases()}
    if name not in by_name:
        raise KeyError(
            f"top_level_bridge.select_case: no case named {name!r} in accel_v2.cases.all_cases() "
            f"(pilot only searches that one catalogue, not the other six module_cnn_accel.py registers)"
        )
    _CASE = by_name[name]


def input_bytes():
    """The selected case's DDR `INPUTS` region (see `TbCase.input_bytes`
    and `TbCase.input_region`), called live from VHDL (`ffi_seed_bytes`
    in `cnn_accel_python_ffi_pkg.vhd`) instead of being written into
    `mem_image.csv`. Same bytes the compiler's own `emit_program` already
    produced -- this only changes how they reach the simulated DDR."""
    if _CASE is None:
        raise RuntimeError("top_level_bridge.input_bytes: select_case was never called")
    return np.frombuffer(_CASE.input_bytes(), dtype=np.uint8)


def check_live_result(
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
    """Called from tb_cnn_accel_top's main process right after STATUS.DONE,
    instead of writing counters.csv/result.csv. `export_bytes` is the raw
    exported DDR region as a VUnit integer_array_t of UNSIGNED byte values
    (0..255) -- read straight out of the simulator's `memory_t` model via
    `read_word`, the same values `result.csv` would otherwise have held.
    Every other argument is one counter, in exactly counters.csv's own
    column set/order (see tb_cnn_accel_top.vhd's `put_counter` calls) so
    this dict is byte-for-byte what `read_counters(counters.csv)` would
    have produced. Returns True/False for VHDL to check_true on."""
    if _CASE is None:
        raise RuntimeError("top_level_bridge.check_live_result: select_case was never called")

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
