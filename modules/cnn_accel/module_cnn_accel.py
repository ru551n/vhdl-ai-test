from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tsfpga.module import BaseModule, get_modules

from ghdl_yosys_env import (
    resolve_ghdl_plugin_path,
    resolve_ghdl_prefix,
    resolve_vivado_path,
)

# `cnn_accel_constants.py` and `cnn_accel_isa_generator.py` are flat sibling
# modules (no `__init__.py` anywhere under `modules/`, matching every other
# module here -- see `cnn_accel_model.py`/`test_cnn_accel_model.py`'s own
# flat imports), so this directory must be on `sys.path` before a plain
# `import` of either will resolve. tsfpga loads this file itself via
# `importlib.util.spec_from_file_location` (see `system_utils.load_python_module`)
# which does not add its own directory to `sys.path`, hence the explicit
# bootstrap here rather than relying on the caller's cwd/sys.path.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import cnn_accel_constants  # noqa: E402
import generate_vectors  # noqa: E402
from cnn_accel_isa_generator import CnnAccelIsaPackageGenerator  # noqa: E402

if TYPE_CHECKING:
    from vunit.ui import VUnit

# Reference hardware configuration, from
# doc/cnn_accel_tiled_dataflow_proposal.md section 7 ("BRAM/DSP estimate").
# These are the generic values every per-entity netlist build below is sized
# for, so the recorded resource numbers are all from one coherent design
# point rather than a per-module free-for-all. D12: target XC7A100T.
#
# Values themselves now live in cnn_accel_constants.py (the single Python
# source of truth also consumed by registers_hook() below and by the
# golden model), so there is exactly one place to edit each number; these
# names are unchanged so nothing downstream in this file has to change.
_PE_ROWS = cnn_accel_constants.PE_ROWS
# The 60 fps scaling point, proven by CI/local builds and tests next to
# the default without replacing it (flow_status.md S1-S7).
_PE_ROWS_SCALED = cnn_accel_constants.PE_ROWS_SCALED
_PE_COLS = cnn_accel_constants.PE_COLS
_TILE_CHANNELS = cnn_accel_constants.TILE_CHANNELS  # = _PE_COLS, see proposal section 3.
_MAX_KERNEL_SIZE = cnn_accel_constants.MAX_KERNEL_SIZE
_MAX_ROW_TILE_WORDS = cnn_accel_constants.MAX_ROW_TILE_WORDS
_ASSEMBLY_BUFFERS = cnn_accel_constants.ASSEMBLY_BUFFERS
# = K^2 * ceil(C_max/8) = 9*32 for the target backbone's worst layer
# (3x3x256, layer 9) -- replaces the old arbitrary 512 now that weights are
# streamed per output-channel pass from DDR4 instead of double-buffered
# on-chip (see doc/cnn_accel_weight_buffer.md's depth-sizing note).
_WEIGHT_BUFFER_DEPTH = cnn_accel_constants.WEIGHT_BUFFER_DEPTH
# Independent of _WEIGHT_BUFFER_DEPTH: a real layer only ever needs
# ceil(out_channels/g_pe_rows) bias rows, far fewer than the weight region.
_BIAS_BUFFER_DEPTH = cnn_accel_constants.BIAS_BUFFER_DEPTH
_ACCUM_WIDTH = cnn_accel_constants.ACCUM_WIDTH

# Pooling is a separate kernel bound, LARGER than the convolution one:
# YOLOv8n's SPPF pools 5x5 while all its convolutions are 1x1/3x3, so the
# pool path alone is sized to 5 (see cnn_accel_constants.MAX_POOL_KERNEL_SIZE
# for the full rationale). The old `g_max_kernel_size**2 * 8 <= 128` ceiling
# is gone with the move of the pool lane window from `axi_stream_m2s_t` to
# the unconstrained `window_m2s_t` tap array.
_POOL_MAX_KERNEL_SIZE = cnn_accel_constants.MAX_POOL_KERNEL_SIZE
# Must hold `_POOL_MAX_KERNEL_SIZE**2 * 127` = 3175 without overflow.
_POOL_ACCUM_WIDTH = 16

# D12 was XC7A100T; raised to XC7A200T once the real target network's 320x320
# input was priced in. Package and speed grade are arbitrary among the
# xc7a200t family for a *netlist* build -- it is out-of-context synthesis with
# no I/O and no timing closure, so only the die's primitive mix affects the
# reported LUT/FF/BRAM/DSP counts. Revisit when an actual board is chosen.
_VIVADO_PART = "xc7a200tfbg484-2"

# --------------------------------------------------------------------------
# Two synthesis backends, two different jobs. Ratified 2026-09.
#
# Yosys (`YosysXilinxNetlistBuild`, unconditional, below): the CI-gating
# backend. It is fast (seconds to ~18 minutes for conv_core) and runs
# unconditionally in CI's `ru551n/hdl-docker` image, which has no Vivado.
# Its `build_result_checkers` are now a deliberately LOOSE structural
# regression gate, not a tight resource budget: their only job is to catch
# a *structural* collapse (block-RAM inference silently degrading to
# distributed RAM, a MAC's DSP packing lost to fabric, one leaf exploding
# by an order of magnitude), not to track the design's real resource
# footprint to the LUT. Do not tighten these back down without re-reading
# the note further below on why Yosys LUT counts are not CI-portable.
#
# Vivado (`VivadoNetlistProject`, gated on `resolve_vivado_path()`, at the
# bottom of `get_build_projects`): the AUTHORITATIVE resource backend. Real
# vendor synthesis is the only tool whose LUT/FF/BRAM/DSP numbers are worth
# trusting as "does this fit the part" -- Yosys's open-source Xilinx
# technology mapping is a reasonable stand-in for regression detection but
# is not a resource-budget oracle. Every Vivado project below carries tight
# checkers set directly from a real local measurement, and is meant to be
# re-run and re-baselined by hand after any resource-affecting change.
# It runs locally ONLY: CI's hdl-docker image has no Vivado install, so
# every `VivadoNetlistProject` below MUST be constructed only when
# `resolve_vivado_path()` returns a real path (see that function's own
# docstring in `ghdl_yosys_env.py` for why this guard is mandatory, not
# defensive -- `build_fpga.py --netlist-builds` builds every *registered*
# project with no filter, so an ungated Vivado project is a guaranteed CI
# break on the Vivado-less image). This means Vivado projects, and their
# checkers, are invisible to CI; they only ever run and gate on a
# developer's own machine. Accepted consequence of this split.
#
# The two backends' numbers are NOT directly comparable and a limit
# derived from one must never be copied onto the other:
#   - BRAM is the one resource the two backends agree on (see conv_core's
#     own cross-check comment below) -- both report the die's physical
#     RAMB18/RAMB36 primitive count, and neither tool has a reason to
#     trade BRAM for something else the same way they do LUTs/DSPs.
#   - DSP differs STRUCTURALLY, not just numerically. Vivado will pack two
#     narrow (e.g. int8 x int8) multiplies into one DSP48E1's SIMD mode
#     when the code lets it recognize that pattern, and it will just as
#     happily leave a whole bank of narrow multiplies in LUT fabric when
#     the surrounding code shape (e.g. a boolean-gated conditional
#     accumulate between the multiply and the register, as in
#     `cnn_accel_pe_array`'s `compute_partial_sums`) does not match its
#     default MACC inference template -- see that entity's own Vivado
#     comment below for a concrete, measured example of the latter. Yosys
#     makes neither optimization: every explicit `*` becomes its own
#     DSP48E1 mapping, unconditionally. Neither number is "more correct";
#     they are answers to different questions ("how many multiply
#     primitives does the RTL contain" vs. "how many DSP48E1 hard blocks
#     does a real toolchain actually spend").
#   - LUT differs for the same reason (whatever one tool maps to a DSP,
#     the other maps to LUT fabric instead), compounded by each tool's own
#     independent optimizer, retiming and technology mapping choices.
# Never copy a DSP or LUT limit from one backend's checkers to the other's.
# --------------------------------------------------------------------------


# M10 (doc/tosa_compiler_plan.md ~line 613): the real TOSA fixtures
# compiled into tb_cnn_accel_conv_core's 'test_bitexact_compiler_cases'
# config (the two M10 ones, the two M11 ones, `out_zp_relu` /
# `clamp_5_100`, which exercise the ISA v1.1 `output_offset` /
# `CLAMP_EN` epilogue fields end-to-end from compiler bytes into RTL, and
# the M12 one, `per_channel_oc8`, whose ISA v1.2 `PER_CHANNEL_EN` scale
# table -- the compiler's own SCALE_TABLE constant at `scale_addr` -- is
# streamed from `scale_table_packed.txt`; the 16-channel `per_channel`
# fixture exceeds this testbench's single output-channel tile and stays
# compiler-side only), and the fixed seed each is compiled/run with (matches
# compiler/tests/test_fixtures_m9.py's own `_seed_input` convention: one
# `np.random.default_rng(seed)` per fixture's single graph input, keyed
# "arg0" like every M8/M9 fixture). Shapes come from the compiler's own
# fixtures dir (compiler/tests/fixtures/gen_fixtures.py), not repeated by
# number here beyond what `write_conv_core_vectors` needs from the
# manifest -- see `_compiler_vectors_pre_config`.
_COMPILER_VECTORS_FIXTURES = {
    "conv_rescale_clamp": (1, 8, 8, 4),
    "first_layer_cin3": (1, 8, 8, 3),
    "out_zp_relu": (1, 8, 8, 8),
    "clamp_5_100": (1, 8, 8, 8),
    "per_channel_oc8": (1, 8, 8, 8),
}
_COMPILER_VECTORS_SEED = 0


def _compiler_vectors_pre_config(output_path: str) -> bool:
    """`pre_config` for tb_cnn_accel_conv_core's `test_bitexact_compiler_
    cases` config ONLY (doc/tosa_compiler_plan.md M10): compiles each of
    `_COMPILER_VECTORS_FIXTURES` with the `cnnc` compiler
    (`compiler/cnnc`) against the real `cnn_accel_v1` target and writes
    its per-layer test vectors -- the compiler's OWN emitted
    weights/program bytes, not `cnn_accel_model` called directly -- into
    this config's own VUnit `output_path` via
    `cnnc.backend.cnn_accel_v1.vectors.write_conv_core_vectors`, one case
    directory per fixture's conv layer, prefixed `'<fixture>_'` so the
    fixtures' case names cannot collide. Writes ONE combined
    `cases.txt` covering every case from every fixture (each
    `write_conv_core_vectors` call would otherwise overwrite the
    previous one's, since they share `output_path`).

    `compiler/` is only put on `sys.path` HERE, inside the hook -- not at
    module import time -- so importing `module_cnn_accel.py` itself
    stays side-effect-free (tsfpga imports this file just to discover
    modules/entities; it must keep working even in a checkout with no
    `compiler/` directory at all, e.g. a synthesis-only clone).

    Returns `False` (fails the config, per `add_vunit_config`'s own
    contract) if any fixture is unexpectedly skipped (all are single-
    layer, `out_channels <= PE_ROWS`, so none should ever exceed
    `max_out_channels`) or if nothing was written at all -- a silently
    empty `cases.txt` must not report a green test (see
    `tb_cnn_accel_conv_core.vhd`'s own `run_compiler_cases` assertion for
    the other half of this guarantee)."""
    repo_root = _MODULE_DIR.parent.parent
    compiler_root = repo_root / "compiler"
    if str(compiler_root) not in sys.path:
        sys.path.insert(0, str(compiler_root))

    import numpy as np

    from cnnc.backend.cnn_accel_v1 import write_conv_core_vectors
    from cnnc.driver import compile_tosa
    from cnnc.target.load import load_target

    target = load_target("cnn_accel_v1")
    fixtures_dir = compiler_root / "tests" / "fixtures"
    out_path = Path(output_path)

    ok = True
    all_written: list[str] = []
    for fixture_name, shape in sorted(_COMPILER_VECTORS_FIXTURES.items()):
        result = compile_tosa(fixtures_dir / f"{fixture_name}.mlir", target)
        rng = np.random.default_rng(_COMPILER_VECTORS_SEED)
        inputs = {"arg0": rng.integers(-128, 128, size=shape).astype(np.int8)}

        vectors = write_conv_core_vectors(
            result.program,
            inputs,
            out_path,
            target=target,
            graph=result.fused_graph,
            max_out_channels=cnn_accel_constants.PE_ROWS,
            case_prefix=f"{fixture_name}_",
        )
        if vectors.skipped:
            print(f"_compiler_vectors_pre_config: {fixture_name}: unexpectedly skipped {vectors.skipped}")
            ok = False
        if not vectors.written:
            print(f"_compiler_vectors_pre_config: {fixture_name}: wrote zero cases")
            ok = False
        all_written.extend(vectors.written)

    (out_path / generate_vectors.CONV_CORE_CASES_FILE).write_text("".join(f"{n}\n" for n in all_written))
    return ok and bool(all_written)


class Module(BaseModule):
    def registers_hook(self) -> None:
        """
        Build this module's `hdl-registers` `RegisterList` entirely from
        Python -- no `regs_cnn_accel.toml` exists (see `BaseModule.registers`:
        this hook still runs even when the TOML file is absent, which is
        what makes a 100% Python-defined register list possible). Source of
        truth for every value here is `cnn_accel_constants.py`, per that
        file's own docstring.

        Covers `doc/cnn_accel_csr_req.md` "Register map" (the milestone-M10
        `cnn_accel_csr` AXI4-Lite block's registers -- this only defines the
        map in Python and lets `create_register_synthesis_files()` /
        `create_register_simulation_files()` (inherited from `BaseModule`,
        unmodified) generate `regs_src/cnn_accel_regs_pkg.vhd`,
        `regs_src/cnn_accel_register_record_pkg.vhd`,
        `regs_src/cnn_accel_register_file_axi_lite.vhd`, and
        `regs_sim/cnn_accel_register_read_write_pkg.vhd`; `cnn_accel_csr`
        itself, instantiating the AXI-Lite wrapper, is M10, not this
        change) plus every accelerator HW property from
        `cnn_accel_constants.py` as plain `hdl-registers` constants.

        `RegisterMode` friction, flagged rather than silently approximated
        (see also this project's hdl-registers integration report):
        `hdl-registers` 8.1.0's `RegisterMode` set is `r`/`w`/`r_w`/
        `wpulse`/`r_wpulse` -- there is no register mode, and no per-field
        mode, for a *sticky, write-1-to-clear* bit. `STATUS.DONE`/
        `STATUS.ERROR` are specified as exactly that in
        `doc/cnn_accel_csr_req.md`. The mode used here for the whole
        `status` register, `r_wpulse`, is the closest faithful mapping:
        software reads the live value (hardware "up" conduit, which also
        covers the plain-`RO` `BUSY` bit sharing this register), and a
        write is presented to hardware as a one-cycle pulse of the written
        bits rather than being stored -- `cnn_accel_csr`'s own (M10, not
        yet written) RTL must AND that write-pulse against the current
        sticky bits to implement "write 1 clears" itself; `hdl-registers`/
        `axi_lite_register_file` do not do the clear-on-write-1 reduction
        for us. `CTRL.START`/`CTRL.ABORT` are genuine "write pulses the bit
        for hardware for one cycle" (no readback contract implied by
        "self-clearing"), which `r_wpulse` expresses exactly, no
        approximation needed there.

        Bit-position friction, same spirit: `hdl-registers` 8.1.0's
        `Register.append_bit`/`.append_bit_vector` always place a new field
        immediately after the previous one (`Register.bit_index` is a plain
        running counter, see `hdl_registers.register.Register._append_field`)
        -- there is no `skip_bits`/reserve-a-gap call. `STATUS.ERR_CODE`
        (bits [7:4]) and `STATUS.ERR_PC_LOW` (bits [31:16]) are fixed by
        `doc/cnn_accel_top_v2_arch.md` section 8, so the gaps before them are
        filled with plain, unused `append_bit_vector(name="reservedN", ...)`
        fields -- always reads '0', never referenced by `cnn_accel_csr`.
        """
        from hdl_registers.register_list import RegisterList
        from hdl_registers.register_modes import REGISTER_MODES

        regs = RegisterList(
            name=self.name, source_definition_file=self.path / "cnn_accel_constants.py"
        )

        # --- doc/cnn_accel_csr_req.md "Register map" -----------------------

        ctrl = regs.append_register(
            name="ctrl",
            mode=REGISTER_MODES["r_wpulse"],
            description="Control pulses. Writing a bit pulses the corresponding signal "
            "for one clock cycle; reads return the (always-zero, self-clearing) "
            "hardware value.",
        )
        ctrl.append_bit(
            name="start",
            description="Pulses 'start' for one cycle when written '1' while "
            "STATUS.BUSY='0'; launches the program at PROGRAM_BASE_ADDR.",
            default_value="0",
        )
        ctrl.append_bit(
            name="abort",
            description="Pulses 'soft_reset_pulse' for one cycle when written '1', "
            "regardless of STATUS.BUSY (ABORT/SOFT_RESET).",
            default_value="0",
        )

        program_base_addr = regs.append_register(
            name="program_base_addr",
            mode=REGISTER_MODES["r_w"],
            description="Program's first instruction byte address. Writes while "
            "STATUS.BUSY='1' are accepted (AXI4-Lite OKAY) but only take effect on "
            "the next START.",
        )
        program_base_addr.append_bit_vector(
            name="addr",
            description="Byte address, full register width.",
            width=32,
            default_value="0" * 32,
        )

        status = regs.append_register(
            name="status",
            mode=REGISTER_MODES["r_wpulse"],
            description="Status bits. BUSY is a plain hardware-provided read value; "
            "DONE/ERROR are sticky and write-1-to-clear (see this method's "
            "docstring for the RegisterMode approximation used to express that).",
        )
        status.append_bit(name="busy", description="Program running.", default_value="0")
        status.append_bit(
            name="done",
            description="Sticky: whole program halted normally (seq_done). "
            "Write-1-to-clear.",
            default_value="0",
        )
        status.append_bit(
            name="error",
            description="Sticky: fatal error, bad opcode or AXI error response "
            "(seq_error). Write-1-to-clear.",
            default_value="0",
        )
        # `hdl-registers` 8.1.0 has no explicit padding/reserved-gap facility
        # (`Register.append_bit`/`.append_bit_vector` are the only field
        # constructors, and both place their field immediately after the
        # previous one -- there is no `skip_bits`/`reserve` call to leave a
        # gap). A plain, appended `reserved` `append_bit_vector` field is
        # therefore the only way to push a later field to a specific bit
        # position; it is otherwise unused (not read, not written
        # meaningfully) and always reads back '0'. Used twice here to land
        # `err_code` on [7:4] and `err_pc_low` on [31:16] per
        # doc/cnn_accel_top_v2_arch.md section 8.
        status.append_bit_vector(
            name="reserved0",
            description="Unused, reads '0'. Padding so `err_code` lands on bits [7:4].",
            width=1,
            default_value="0",
        )
        status.append_bit_vector(
            name="err_code",
            description="Latched `ERR_CODE` (doc/cnn_accel_top_v2_arch.md section 9): "
            "the `err_code` input value at the moment `seq_error` first asserted. "
            "Holds until `STATUS.ERROR` is cleared.",
            width=4,
            default_value="0000",
        )
        status.append_bit_vector(
            name="reserved1",
            description="Unused, reads '0'. Padding so `err_pc_low` lands on bits [31:16].",
            width=8,
            default_value="00000000",
        )
        status.append_bit_vector(
            name="err_pc_low",
            description="Low 16 bits of the faulting descriptor's byte address: the "
            "`err_pc` input, bits [15:0], latched at the moment `seq_error` first "
            "asserted. Holds until `STATUS.ERROR` is cleared.",
            width=16,
            default_value="0" * 16,
        )

        irq_mask = regs.append_register(
            name="irq_mask",
            mode=REGISTER_MODES["r_w"],
            description="Per-cause IRQ enable: "
            "irq <= (STATUS.DONE and done) or (STATUS.ERROR and error). "
            "Masked (0) after reset -- host must enable explicitly.",
        )
        irq_mask.append_bit(name="done", description="Mask for STATUS.DONE.", default_value="0")
        irq_mask.append_bit(
            name="error", description="Mask for STATUS.ERROR.", default_value="0"
        )

        hw_info = regs.append_register(
            name="hw_info",
            mode=REGISTER_MODES["r"],
            description="Read-only array geometry, driven from the "
            "`g_pe_rows`/`g_pe_cols`/`g_tile_channels` generics actually elaborated "
            "into this bitstream (flow_status.md S3): the host driver reads this "
            "instead of hardcoding an array size, so the same driver binary works "
            "unmodified against the default (8-row) and scaled (16-row) builds.",
        )
        hw_info.append_bit_vector(
            name="pe_rows",
            description="Elaborated `g_pe_rows` (output-channel lanes). "
            f"One of {cnn_accel_constants.PE_ROWS_LEGAL}.",
            width=8,
            default_value=format(cnn_accel_constants.PE_ROWS, "08b"),
        )
        hw_info.append_bit_vector(
            name="pe_cols",
            description="Elaborated `g_pe_cols` (input-channel lanes, = tile_channels).",
            width=8,
            default_value=format(cnn_accel_constants.PE_COLS, "08b"),
        )
        hw_info.append_bit_vector(
            name="tile_channels",
            description="Elaborated `g_tile_channels` (= pe_cols).",
            width=8,
            default_value=format(cnn_accel_constants.TILE_CHANNELS, "08b"),
        )
        hw_info.append_bit_vector(
            name="max_kernel_size",
            description="Elaborated `g_max_kernel_size` (largest K_h/K_w this "
            "datapath supports).",
            width=8,
            default_value=format(cnn_accel_constants.MAX_KERNEL_SIZE, "08b"),
        )

        hw_info2 = regs.append_register(
            name="hw_info2",
            mode=REGISTER_MODES["r"],
            description="Second read-only hardware-info register (HW_INFO's four "
            "8-bit fields are full): ISA version and elaborated local tensor "
            "scratchpad size, per doc/cnn_accel_top_v2_arch.md section 8.",
        )
        hw_info2.append_bit_vector(
            name="isa_version",
            description="Instruction-set version this build implements, "
            "`(major << 8) | minor`; see `cnn_accel_constants.ISA_VERSION` / "
            "`accel_v2/isa.py`.",
            width=16,
            default_value=format(cnn_accel_constants.ISA_VERSION, "016b"),
        )
        hw_info2.append_bit_vector(
            name="tensor_mem_kib",
            description="Elaborated `g_tensor_bytes` (the local tensor scratchpad, "
            "doc/cnn_accel_top_v2_arch.md section 4), in KiB.",
            width=16,
            default_value="0" * 16,
        )

        hw_info3 = regs.append_register(
            name="hw_info3",
            mode=REGISTER_MODES["r"],
            description="Third read-only hardware-info register (HW_INFO's four "
            "8-bit fields and HW_INFO2's two 16-bit fields are both full): "
            "pooling-specific and DMA-tiling bounds a host/compiler cannot "
            "otherwise discover at runtime, per doc/cnn_accel_top_v2_arch.md "
            "section 8.",
        )
        hw_info3.append_bit_vector(
            name="max_pool_kernel_size",
            description="Elaborated `g_max_pool_kernel_size` (largest pooling "
            "K_h/K_w this datapath supports). Distinct from HW_INFO.MAX_KERNEL_SIZE "
            "-- pooling is sized separately (ISA v2.1, SPPF needs 5x5 while no "
            "convolution does). See cnn_accel_constants.MAX_POOL_KERNEL_SIZE.",
            width=8,
            default_value=format(cnn_accel_constants.MAX_POOL_KERNEL_SIZE, "08b"),
        )
        hw_info3.append_bit_vector(
            name="max_row_tile_words",
            description="Elaborated `g_max_row_tile_words` (per-row activation "
            "tile depth `cnn_accel_window_gen` was sized to, in words). The "
            "compiler's `in_width * ceil(in_channels / TILE_CHANNELS)` bound "
            "(both conv and pool) must not exceed this; it has changed across "
            "builds (512 -> 1920) and was previously invisible to the host/"
            "compiler at runtime. See cnn_accel_constants.MAX_ROW_TILE_WORDS.",
            width=16,
            default_value=format(cnn_accel_constants.MAX_ROW_TILE_WORDS, "016b"),
        )

        # --- Performance counters (spec section 8, CSR 0x1C-0x3C) ----------
        # Plain 'r' registers: each is a single 32-bit hardware-maintained
        # counter, pass-through from 'cnn_accel_csr's 'counters' port
        # (csr_counters_t, src/cnn_accel_v2_pkg.vhd) into 'regs_up'. Appended
        # in this exact order (offset is assigned by append order) so their
        # addresses match the spec's 0x1C..0x3C table (shifted from
        # 0x18..0x38 by HW_INFO3's addition ahead of them).
        cmd_count = regs.append_register(
            name="cmd_count",
            mode=REGISTER_MODES["r"],
            description="Number of descriptors (instructions) retired since the "
            "last START.",
        )
        cmd_count.append_bit_vector(
            name="value", description="Count, full register width.", width=32,
            default_value="0" * 32,
        )

        cycle_count = regs.append_register(
            name="cycle_count",
            mode=REGISTER_MODES["r"],
            description="Clock cycles elapsed from the accepted START to DONE "
            "(or to now, while still BUSY).",
        )
        cycle_count.append_bit_vector(
            name="value", description="Count, full register width.", width=32,
            default_value="0" * 32,
        )

        compute_cycles = regs.append_register(
            name="compute_cycles",
            mode=REGISTER_MODES["r"],
            description="Cycles during which at least one compute engine (PE array, "
            "pool, activation) was active.",
        )
        compute_cycles.append_bit_vector(
            name="value", description="Count, full register width.", width=32,
            default_value="0" * 32,
        )

        stall_cycles = regs.append_register(
            name="stall_cycles",
            mode=REGISTER_MODES["r"],
            description="Cycles during which a command was dispatched but blocked "
            "(waiting on DMA/weight-buffer/scratchpad availability).",
        )
        stall_cycles.append_bit_vector(
            name="value", description="Count, full register width.", width=32,
            default_value="0" * 32,
        )

        ddr_rd_bytes = regs.append_register(
            name="ddr_rd_bytes",
            mode=REGISTER_MODES["r"],
            description="All AXI read bytes, including descriptor fetches and "
            "weights (not just tensor payload).",
        )
        ddr_rd_bytes.append_bit_vector(
            name="value", description="Byte count, full register width.", width=32,
            default_value="0" * 32,
        )

        ddr_wr_bytes = regs.append_register(
            name="ddr_wr_bytes",
            mode=REGISTER_MODES["r"],
            description="All AXI write bytes. The decisive residency-proof counter "
            "(doc/cnn_accel_top_v2_arch.md section 8): for a multi-op local chain "
            "it must equal exactly the final STORE size.",
        )
        ddr_wr_bytes.append_bit_vector(
            name="value", description="Byte count, full register width.", width=32,
            default_value="0" * 32,
        )

        tensor_load_count = regs.append_register(
            name="tensor_load_count",
            mode=REGISTER_MODES["r"],
            description="Number of LOAD commands retired.",
        )
        tensor_load_count.append_bit_vector(
            name="value", description="Count, full register width.", width=32,
            default_value="0" * 32,
        )

        tensor_store_count = regs.append_register(
            name="tensor_store_count",
            mode=REGISTER_MODES["r"],
            description="Number of STORE commands retired.",
        )
        tensor_store_count.append_bit_vector(
            name="value", description="Count, full register width.", width=32,
            default_value="0" * 32,
        )

        weight_load_bytes = regs.append_register(
            name="weight_load_bytes",
            mode=REGISTER_MODES["r"],
            description="Bytes fetched by LOADW plus output-channel-tiling-loop "
            "weight refills.",
        )
        weight_load_bytes.append_bit_vector(
            name="value", description="Byte count, full register width.", width=32,
            default_value="0" * 32,
        )

        local_bytes = regs.append_register(
            name="local_bytes",
            mode=REGISTER_MODES["r"],
            description="KiB-granularity local tensor scratchpad traffic "
            "(doc/cnn_accel_top_v2_arch.md section 8): finer byte counts are not "
            "kept, only enough resolution to sanity-check scratchpad usage.",
        )
        local_bytes.append_bit_vector(
            name="rd_kib",
            description="KiB read from the scratchpad.",
            width=16,
            default_value="0" * 16,
        )
        local_bytes.append_bit_vector(
            name="wr_kib",
            description="KiB written to the scratchpad.",
            width=16,
            default_value="0" * 16,
        )

        # --- Accelerator HW properties, as plain constants -----------------
        # Native hdl-registers constants: each is a single scalar value with
        # no internal structure, so IntegerConstant (via add_constant's
        # automatic type dispatch on a plain `int`) is a complete, faithful
        # representation -- no custom generator needed for these.
        #
        # `pe_rows`/`pe_cols`/`tile_channels` below are NOT redundant with
        # `HW_INFO` above: these are generation-time Python constants (this
        # repo's reference/default build point, baked into generated VHDL as
        # `constant`s for e.g. static assertions), whereas `HW_INFO` is a
        # runtime AXI-readable register driven by the generics an actual
        # elaborated bitstream was built with -- the two agree for the
        # default (8-row) build and deliberately disagree for the scaled
        # (16-row) one, which is the entire point of S3.

        regs.add_constant(
            name="pe_rows",
            value=cnn_accel_constants.PE_ROWS,
            description="Number of PE array rows (output-channel lanes).",
        )
        regs.add_constant(
            name="pe_cols",
            value=cnn_accel_constants.PE_COLS,
            description="Number of PE array columns (= tile_channels).",
        )
        regs.add_constant(
            name="tile_channels",
            value=cnn_accel_constants.TILE_CHANNELS,
            description="Input-channel tile group size (= pe_cols).",
        )
        regs.add_constant(
            name="activation_plane_channels",
            value=cnn_accel_constants.ACTIVATION_PLANE_CHANNELS,
            description=(
                "T from decision S6: activations are stored in DDR as channel-tiled "
                "planes [C/T][H][W][T], so one pixel is T contiguous int8 bytes and "
                "every plane's byte address/length is a multiple of T. Numerically "
                "equal to tile_channels but a distinct concept (memory layout, not "
                "datapath width)."
            ),
        )
        regs.add_constant(
            name="max_axi_data_width",
            value=cnn_accel_constants.MAX_AXI_DATA_WIDTH,
            description=(
                "Largest legal g_axi_data_width, in bits. Every activation DMA "
                "request is a whole S6 plane, hence a multiple of "
                "activation_plane_channels bytes, so any bus at or below this width "
                "satisfies the DMA engines' word-aligned addr/length requirement "
                "unconditionally. Asserted at elaboration in cnn_accel_ofmap_dma and "
                "cnn_accel_axi_read_dma, where a violation would not error but "
                "silently hang (dma_done never fires)."
            ),
        )
        regs.add_constant(
            name="max_kernel_size",
            value=cnn_accel_constants.MAX_KERNEL_SIZE,
            description="Largest K_h/K_w this accelerator's datapath supports.",
        )
        regs.add_constant(
            name="max_pool_kernel_size",
            value=cnn_accel_constants.MAX_POOL_KERNEL_SIZE,
            description=(
                "Largest pooling K_h/K_w this accelerator supports. Deliberately "
                "separate from (and larger than) max_kernel_size: only the pool "
                "path is sized to it, so a 5x5 pool costs nothing in the conv "
                "datapath. See cnn_accel_constants.MAX_POOL_KERNEL_SIZE."
            ),
        )
        regs.add_constant(
            name="isa_version",
            value=cnn_accel_constants.ISA_VERSION,
            description="Instruction-set version, `(major << 8) | minor`; same value "
            "as read back at runtime via CSR.HW_INFO2.ISA_VERSION. Exposed as a "
            "generated constant too (`cnn_accel_constant_isa_version`) so "
            "'src/cnn_accel_v2_pkg.vhd`'s `c_isa_version` derives from this single "
            "source of truth instead of restating the `0x0200` literal.",
        )
        regs.add_constant(
            name="max_row_tile_words",
            value=cnn_accel_constants.MAX_ROW_TILE_WORDS,
            description="Weight-buffer row-tile-word sizing bound.",
        )
        regs.add_constant(
            name="assembly_buffers",
            value=cnn_accel_constants.ASSEMBLY_BUFFERS,
            description=(
                "Tap-assembly buffers in the CONV path's cnn_accel_window_gen "
                "instance (its g_assembly_buffers), i.e. how many windows may be in "
                "flight between the row banks and the PE array at once. Exposed as a "
                "generated constant so cnn_accel_conv_core derives it from this "
                "single source instead of restating a literal, exactly as "
                "cnn_accel_v2_pkg does for isa_version. The POOL instance keeps the "
                "entity default of 1. See cnn_accel_constants.ASSEMBLY_BUFFERS."
            ),
        )
        regs.add_constant(
            name="weight_buffer_depth",
            value=cnn_accel_constants.WEIGHT_BUFFER_DEPTH,
            description="cnn_accel_weight_buffer's g_weight_buffer_depth reference value.",
        )
        regs.add_constant(
            name="bias_buffer_depth",
            value=cnn_accel_constants.BIAS_BUFFER_DEPTH,
            description="Bias buffer depth (independent of weight_buffer_depth).",
        )
        regs.add_constant(
            name="accum_width",
            value=cnn_accel_constants.ACCUM_WIDTH,
            description="PE array / bias_requant accumulator width, bits.",
        )
        regs.add_constant(
            name="scale_table_entry_bytes",
            value=cnn_accel_constants.SCALE_TABLE_ENTRY_BYTES,
            description=(
                "ISA v1.2 per-channel requant table: bytes per output channel in DDR "
                "(int32 multiplier, uint8 shift, 3 zero bytes)."
            ),
        )
        regs.add_constant(
            name="scale_buffer_entry_bits",
            value=cnn_accel_constants.SCALE_BUFFER_ENTRY_BITS,
            description=(
                "Bits of each per-channel table entry kept in cnn_accel_weight_buffer's "
                "scale_buffer (multiplier + shift; the zero bytes are dropped)."
            ),
        )

        self._registers = regs

    def create_register_synthesis_files(self) -> None:
        """
        `BaseModule`'s register-artifact generation (`cnn_accel_regs_pkg.vhd`,
        the record package, the AXI-Lite wrapper -- all native `hdl-registers`
        generators, gated on `create_register_package`/`create_record_package`/
        `create_axi_lite_wrapper`, unmodified) plus this module's own custom
        generator for the ISA byte-offset table / opcodes / flags, which
        `hdl-registers`' native register/constant model cannot express as one
        coherent, structurally-checked table (see `cnn_accel_isa_generator.py`'s
        own docstring) -- so it is not just more `RegisterList.add_constant`
        calls in `registers_hook()` above.
        """
        super().create_register_synthesis_files()

        if self.registers is not None:
            # Not `.create_if_needed()`: that gates on `self.registers`'
            # `object_hash`, which never changes here since the ISA table
            # lives in `cnn_accel_constants.py`, not in any `RegisterList`
            # constant/register that would be part of that hash -- it would
            # never regenerate after the first time and would silently ship a
            # stale ISA package. Plain `.create()` is correct: the generator
            # overrides `_create_artifact()` to skip the write (and so
            # preserve the file's mtime) when the generated body is
            # unchanged. That mtime matters -- see that override's docstring
            # for the GHDL "must be reanalysed" netlist-build failure an
            # unconditional rewrite causes.
            CnnAccelIsaPackageGenerator(
                register_list=self.registers, output_folder=self.register_synthesis_folder
            ).create()

    def get_build_projects(self) -> list:
        # Local import: tsfpga.yosys.project needs a tsfpga build with Yosys
        # netlist-build support (not in the stable release this project's
        # run.py/VUnit flow uses), so this must not be imported at module
        # load time -- only build_fpga.py ever calls this method.
        from tsfpga.vivado.build_result_checker import (
            BlockRams,
            DspBlocks,
            EqualTo,
            Ffs,
            LessThan,
            LutRams,
            Ramb18,
            Ramb36,
            TotalLuts,
        )
        from tsfpga.yosys.project import YosysXilinxNetlistBuild

        # S7 (flow_status.md), option 1 ratified by the user: a fast,
        # synthesis-only (no place-and-route) 150 MHz timing estimate,
        # via tsfpga's own `analyze_synthesis_timing=True` on the entities
        # the unconstrained logic-level report flagged as hosting the deep
        # (102-105 level) paths -- `pe_array`/`conv_core` (both row counts)
        # and `bias_requant`/`weight_buffer`. This auto-detects the `clk`
        # port, runs `create_clock`+`open_run`+`report_timing -setup`
        # (`tsfpga/vivado/tcl.py`'s `_synthesis()`) and populates
        # `build_result.maximum_synthesis_frequency_hz`.
        #
        # NOT wired up as a `build_result_checker`: `VivadoNetlistProject
        # .build()` calls `self._check_size(build_result=result)` (which
        # runs every `build_result_checkers` entry) *before* it computes
        # `maximum_synthesis_frequency_hz` a few lines further down in the
        # same method (tsfpga 2026.1-era source, `vivado/project.py`
        # `build()`) -- so a checker reading that field always sees `None`,
        # regardless of the real number. Verified empirically: a checker
        # class doing exactly this raised on every entity, including ones
        # whose `timing.rpt` on disk showed a real, easily-passing slack
        # number. This is a genuine tsfpga ordering bug/limitation, not a
        # design-side mistake -- do not "fix" it by making the checker
        # swallow `None` (that would silently stop checking anything).
        #
        # Until tsfpga fixes the ordering (or this project forks
        # `VivadoNetlistProject.build()` to reorder it), the frequency
        # number is real and tool-produced but only available in the
        # printed build summary / `timing.rpt`, not as an automated gate --
        # so it is measured by hand and pinned as a comment below, the same
        # way every LUT/FF/BRAM/DSP number in this Vivado section already
        # is (see the module-level comment above `vivado_path =
        # resolve_vivado_path()` for why that is the established pattern
        # here, not a shortcut). A prior attempt at a custom
        # `STEPS.SYNTH_DESIGN.TCL.POST` hook doing `create_clock` +
        # `get_timing_paths` independently confirmed Vivado post-synthesis
        # hooks are unreliable for this (empty `get_clocks` at hook time) --
        # matching tsfpga's own maintainer comment in `tcl.py`
        # ("post-synthesis hooks ... seems to be very bugged"), which is
        # why tsfpga puts these calls directly in the build script instead
        # of a hook, and why this project does not reinvent that path.
        #
        # Full timing closure (option 2, real place-and-route) is deferred
        # until `cnn_accel_top` exists (M10-M13).
        #
        # IMPORTANT side effect discovered while measuring these six builds:
        # `analyze_synthesis_timing=True` is NOT a pure post-hoc analysis
        # switch. `VivadoNetlistProject.create()` unconditionally appends an
        # "early"-processing-order auto-clock constraint file, and only
        # populates it with a real `create_clock -period 2.000ns` (500 MHz,
        # `_clock_period_ns` in `vivado/project.py`) on the `clk` port when
        # this flag is set -- and that file is read via `read_xdc` and left
        # `USED_IN_SYNTHESIS` (default), so it is live *during synthesis
        # itself*, not just the post-synthesis `open_run`/`report_timing`
        # step. Turning this on therefore changes Vivado's actual
        # area/timing tradeoffs (retiming, BRAM-cascade choices, etc), which
        # is why the six "Measured" comments below now differ from the
        # pre-S7 unconstrained-synthesis numbers for the same RTL/generics,
        # and why three of the six needed their `checkers=[...]` values
        # re-pinned (weight_buffer's RAMB18, conv_core's LUTs/FFs/RAMB36,
        # conv_core_pe_rows_16's FFs/RAMB36/RAMB18) -- not a regression, a
        # different (and now permanent, since this flag stays on) synthesis
        # configuration. Do not silently "fix" a future checker failure here
        # by assuming the RTL changed without checking the diff first.
        #
        # First measurement round (pre-fix) came out at: bias_requant
        # 520.02 MHz, weight_buffer 257.80 MHz, pe_array 56.52/56.44 MHz
        # (8/16 rows), conv_core 46.77/45.38 MHz (8/16 rows) -- the last
        # four roughly a third of target.
        #
        # **THE 520.02 MHz FOR bias_requant WAS A LIE, AND THE LESSON IS
        # THE MOST IMPORTANT THING IN THIS COMMENT.** `report_timing
        # -setup` on an out-of-context netlist reports only
        # *register-to-register* paths. An out-of-context build gives its
        # input ports no input delay, so any combinational cone that
        # *starts at an input port* is never timed at all. Every one of
        # `cnn_accel_bias_requant`'s ~21 ns of bias-add / requant-multiply
        # / rounded-shift / saturate logic hangs off `s_accum_m2s.data` and
        # `bias_rd_data`, i.e. off input ports, so standalone synthesis saw
        # none of it and happily reported a 1-LUT `out_valid_q ->
        # out_data_q/CE` path as the worst case. Inside `conv_core`, where
        # that same cone is fed by `weight_buffer`'s registers, it was the
        # 46.77 MHz critical path of the whole composition.
        #
        # **Rule: a leaf entity's out-of-context Fmax is an upper bound and
        # nothing more. Only a composition entity (conv_core, eventually
        # cnn_accel_top) produces a number worth acting on.** Never
        # "optimize" a leaf against its own standalone figure, and never
        # conclude a leaf is fine because its standalone figure is high.
        #
        # Fixing this took three independent reworks, each of which only
        # became visible once the previous one was done -- conv_core sat at
        # 46.77 -> 46.77 -> 47.66 -> 159.72 -> 159.95 MHz, i.e. the first
        # two fixes looked like they had achieved *nothing* at the
        # composition level because a comparable path was hiding behind
        # each:
        #   1. `cnn_accel_pe_array`: the single-cycle `compute_partial_sums`
        #      MAC cone became a self-timed pipeline with one register per
        #      adder-tree level (56.52 -> 232.67 MHz standalone).
        #   2. `cnn_accel_bias_requant`: one combinational cone became a
        #      7-stage pipeline, and `round_shift_right`'s
        #      re-multiply/subtract/compare rounding was replaced by the
        #      guard/sticky round-half-to-even form (21.3 ns -> off the
        #      critical path).
        #   3. `cnn_accel_window_gen`: the per-cycle read-address
        #      arithmetic (two runtime multiplies, a modulo and four range
        #      compares feeding the BRAM address pins directly) became
        #      registered per-pixel geometry plus look-ahead accumulators;
        #      this is also why its DSP count went 4 -> 0.
        # Final: **conv_core 159.95 MHz at both 8 and 16 rows**, target met.
        # conv_core now has a *plateau* of paths around 6 ns, so expect the
        # next single-path fix here to buy almost nothing on its own.
        #
        # This is real RTL/timing work and was delegated to strong-model
        # subagents per this project's standing rule (see the
        # `vivado-gotchas` skill): always use a strong model to analyze and
        # fix timing, never a quick pass by the orchestrating agent.

        modules = get_modules(
            modules_folder=self.path.parent, names_include={self.name}
        ) + get_modules(
            modules_folder=self.path.parent.parent / "hdl-modules" / "modules",
            # "fifo" added for cnn_accel_weight_buffer's reused
            # hdl-modules 'fifo.fifo' prefetch FIFO (g_fill_fifo_depth > 0
            # by default -- see cnn_accel_weight_buffer.vhd).
            #
            # "register_file"/"axi_lite"/"axi" added because this module now
            # has registers (registers_hook() below): cnn_accel's own
            # regs_src/cnn_accel_regs_pkg.vhd unconditionally pulls in
            # register_file.register_file_pkg (mode encoding), and the
            # generated AXI-Lite wrapper additionally needs axi_lite_pkg --
            # both must be analyzable even for netlist builds of unrelated
            # cnn_accel leaves (bias_requant, pool, weight_buffer, ...),
            # since `modules` here is the shared library set for every
            # `build()` call below, one per-cnn_accel-library GHDL analysis.
            # Same three-module recipe hdl-modules' own
            # register_file/module_register_file.py uses for its own
            # netlist build (get_build_projects there:
            # `names_include=[self.name, "axi", "axi_lite", "common", "math"]`).
            names_include={
                "axi_stream",
                "common",
                "math",
                "fifo",
                "register_file",
                "axi_lite",
                "axi",
            },
        )

        def build(
            name: str, generics: dict, checkers: list, top: str | None = None
        ) -> YosysXilinxNetlistBuild:
            # 'top' defaults to the project name; pass it explicitly to
            # register the same entity twice at two different geometries
            # (see 'cnn_accel_window_gen_pool').
            return YosysXilinxNetlistBuild(
                name=name,
                modules=modules,
                top=top if top is not None else name,
                family="xc7",
                generics=generics,
                build_result_checkers=checkers,
                ghdl_plugin_path=resolve_ghdl_plugin_path(),
                ghdl_prefix=resolve_ghdl_prefix(),
                defined_at=Path(__file__),
            )

        # Yosys builds: the CI-gating structural regression gate (see the
        # module-level "Two synthesis backends" comment above for the full
        # split rationale). These `build_result_checkers` are deliberately
        # LOOSE -- roughly 1.5-2x the measured LUT/FF baseline, not a tight
        # `LessThan` set just above it -- because their only job is to
        # catch a structural collapse (block-RAM inference silently
        # degrading to distributed RAM, DSP packing lost to fabric, a leaf
        # exploding by an order of magnitude), not to track the design's
        # real resource footprint. Vivado (below, local-only) is now the
        # authoritative backend for that; do not re-tighten these back
        # toward the measured baseline.
        #
        # Baselines are measured on CI's toolchain -- Yosys v0.68 *release*,
        # as shipped in ru551n/hdl-docker:1.2.0 -- because CI is what these
        # limits actually gate. Do not re-baseline LUT counts from a local
        # Yosys: the difference is not jitter. Yosys v0.68+182 (dev) gives
        # markedly smaller LUT counts for the same RTL (window_gen 8884 vs
        # 23757, bias_requant 2348 vs 3686, pool 342 vs 468), so a locally
        # derived LUT limit fails on CI for no design reason at all -- this
        # is exactly the ~2.7x-observed variance the 1.5-2x LUT/FF headroom
        # above is sized to absorb. FFs, DSPs and block RAMs are *not*
        # subject to that particular variance: they are structural counts
        # that Yosys's optimizer cannot trade away regardless of version,
        # and this session confirmed it empirically for window_gen and
        # conv_core (M7) -- every FF/DSP/BlockRam figure measured locally
        # after the BRAM-inference fix landed exactly matches arithmetic
        # built from the old CI baselines of the untouched submodules (see
        # window_gen's and conv_core's own comments below). Their checkers
        # below are therefore kept meaningful and close to the measured
        # value rather than loosened along with LUTs/FFs -- in particular
        # `cnn_accel_window_gen`'s `BlockRams(EqualTo(3))` and
        # `cnn_accel_pe_array`'s `BlockRams(LessThan(1))` stay exact/tight,
        # since those are the checks that actually catch the two failure
        # modes this whole gate exists for (BRAM inference dying, DSP
        # packing/MAC structure lost). Local builds are still useful for
        # *relative* before/after comparisons and, per the above, for
        # FF/DSP/BRAM absolute numbers too; only LUT absolute numbers must
        # come from CI.
        projects = [
            build(
                name="cnn_accel_bias_requant",
                generics={
                    "g_accum_width": _ACCUM_WIDTH,
                    # M6 record retrofit landed: this entity's accumulator
                    # input now carries `accum_m2s_t`/`accum_s2m_t` (an
                    # array of `g_pe_rows` lanes, constrained at the
                    # declaration site) instead of the fixed 128-bit
                    # `axi_stream_m2s_t`, so the old
                    # `g_accum_width*g_pe_rows <= axi_stream_data_sz` lane
                    # ceiling is gone and this build now uses the project's
                    # real `_PE_ROWS` (8), same as every other entity here.
                    "g_pe_rows": _PE_ROWS,
                    "g_bias_addr_width": 9,
                },
                # Baseline 2026-09 (Yosys v0.68 release, CI run) at
                # `g_pe_rows`=8: 7255 LUTs, 66 FFs, 0 BRAM, 32 DSP.
                #
                # The previous 4-lane pin measured 3686 LUTs, 34 FFs, 0 BRAM,
                # 16 DSP, so doubling the lane count in the M6 record
                # retrofit roughly doubled the entity: the requant multiplies
                # (2 DSP per lane) and the per-lane datapath scale linearly
                # with `g_pe_rows`, and there is little shared logic to
                # amortize.
                #
                # As with every other entity here, only CI's Yosys v0.68
                # *release* numbers are the real baseline; a local
                # Yosys v0.68+182 (dev) build gives markedly smaller
                # netlists for the same RTL and must not be used to set
                # these limits.
                #
                # Loosened 2026-09 into a structural regression gate (see
                # module-level comment): LUT/FF given ~1.75x headroom over
                # the 7255/66 CI baseline above. BRAM/DSP kept close to
                # measured -- 0 BRAM and 32 DSP are the meaningful,
                # structural checks here (the 2-DSP-per-lane requant
                # multiply pattern, unconditionally mapped by Yosys).
                #
                # FF re-pinned 2026-09 (S7): 66 -> 2938 measured, because
                # this entity went from one combinational cone to a 7-stage
                # pipeline to make 150 MHz (module-level comment). ~1.5x
                # headroom. LUT moved 4853 (was well inside the 13000
                # already). DSP unchanged at 32 -- the requant multiply
                # pattern is untouched, which is exactly what that check is
                # for.
                checkers=[
                    TotalLuts(LessThan(13000)),
                    Ffs(LessThan(4500)),
                    BlockRams(LessThan(1)),
                    DspBlocks(LessThan(36)),
                ],
            ),
            build(
                name="cnn_accel_pool",
                generics={
                    "g_max_kernel_size": _POOL_MAX_KERNEL_SIZE,
                    "g_accum_width": _POOL_ACCUM_WIDTH,
                },
                # Baseline 2026-09 (Yosys v0.68 release): 468 LUTs, 27 FFs,
                # 0 BRAM, 0 DSP.
                # Pure combinational reduction network, must stay tiny.
                #
                # Loosened 2026-09 into a structural regression gate (see
                # module-level comment): LUT given ~1.8x headroom over the
                # 468 CI baseline. FF/BRAM/DSP already had comparable
                # headroom and are kept as the meaningful, near-measured
                # structural checks (0 BRAM, 0 DSP -- purely combinational).
                #
                # Re-pinned 2026-09 (YOLOv8n sizing pass): this build now
                # elaborates at the 5x5 pool bound (_POOL_MAX_KERNEL_SIZE
                # 3 -> 5), so the reduction network went from 9 to 25 lanes
                # and measures 1002 LUTs on a local dev Yosys (was ~342 at
                # 3x3 on the same tool) -- 2.9x growth for a 2.8x lane
                # count, i.e. exactly linear, which is the property worth
                # gating on. Applying that same 2.9x to the 468-LUT CI
                # baseline gives ~1370, and the standing ~1.8x structural
                # headroom gives the limit below. FF/BRAM/DSP are
                # unchanged: the extra lanes are pure combinational
                # reduction, so 27 FFs (the one output register), 0 BRAM
                # and 0 DSP must all stay put.
                checkers=[
                    TotalLuts(LessThan(2500)),
                    # Re-pinned 2026-09-09, timing pass. Measured 275.
                    # This gate was badly stale in BOTH directions: it
                    # predates the registered tap mask, the two-entry
                    # tagged output buffer AND the second reduction-tree
                    # register stage this pass added, and 60 has not been
                    # a plausible flip-flop count for this entity for
                    # several commits. The stage this pass is responsible
                    # for is small and exactly countable: one valid, one
                    # is_avg and one last bit, plus 'level_count(c_split2)'
                    # = 2 nodes of max (8 bits) and sum (13 bits), i.e.
                    # ~45 FFs of the 275. Re-pinned loosely, as every
                    # Yosys gate in this file is, at ~1.45x.
                    Ffs(LessThan(400)),
                    BlockRams(LessThan(1)),
                    DspBlocks(LessThan(1)),
                ],
            ),
            build(
                name="cnn_accel_weight_buffer",
                generics={
                    "g_weight_buffer_depth": _WEIGHT_BUFFER_DEPTH,
                    "g_bias_buffer_depth": _BIAS_BUFFER_DEPTH,
                    "g_pe_rows": _PE_ROWS,
                    "g_pe_cols": _PE_COLS,
                    "g_accum_width": _ACCUM_WIDTH,
                },
                # Pre-rework baseline (Yosys v0.68 release, CI run): 175
                # LUTs, 59 FFs, 72 block RAMs (8 RAMB36 + 64 RAMB18), 0 DSP
                # -- the old 2-bank, per-lane-byte-write-enable weight
                # region fragmented into 64 separate 1024x8 memories (one
                # RAMB18 each), plus an 8-RAMB36 bias memory that was 99.9%
                # dead (cnn_accel_bias_requant always reads row 0).
                #
                # Single-buffer + whole-row-write + separate small bias
                # depth rework (see cnn_accel_weight_buffer.vhd's own
                # header comment and doc/cnn_accel_weight_buffer.md):
                # measured 2026-09 (local dev Yosys, weight_buffer-only
                # netlist build, default g_fill_fifo_depth=32): 881 LUTs,
                # 1093 FFs, 15 block RAMs (0 RAMB36 + 15 RAMB18), 0 DSP --
                # BRAM count dropped 72 -> 15 as expected (one wide weight
                # region + a tiny bias region instead of 64 per-lane
                # RAMB18s), but FFs/LUTs went *up* from the pre-rework
                # baseline: the two whole-row assembly registers
                # ('weight_row_assemble_q'/'bias_row_assemble_q', 512 +
                # 256 = 768 bits between them) plus the depth-32 prefetch
                # FIFO's distributed-RAM storage (RAM32M primitives, LUT-
                # mapped since 32 entries is too shallow for Yosys to pick
                # a block RAM) now cost register/LUT area that the old
                # per-lane-byte-write design didn't pay -- an accepted
                # trade of BRAM fragmentation for FF/LUT (see proposal doc
                # section 5 / doc/cnn_accel_weight_buffer.md).
                #
                # Loosened 2026-09 into a structural regression gate (see
                # module-level comment): LUT/FF given ~1.75x headroom over
                # the measured 881/1093. BRAM kept close to measured (15) --
                # that is the meaningful check here, catching a regression
                # back toward the old 72-BRAM fragmented layout. DSP stays
                # at 0, structural (no multiply in this entity).
                checkers=[
                    TotalLuts(LessThan(1600)),
                    # Re-pinned 2026-09-09, timing pass. Measured 2304.
                    # The growth is this pass's block-RAM output register
                    # on the weight region ('weight_rd_data_p', 8 * 8 * 8
                    # = 512 bits at this geometry). NOTE the backend
                    # difference, and do not "fix" it: Vivado folds that
                    # register into the RAMB36's own DO register and its
                    # flip-flop count went DOWN (1202 -> 1156), while
                    # Yosys's xc7 flow keeps it in fabric and the count
                    # goes up by the full 512. Same RTL, same intent,
                    # different mapping -- exactly the split the
                    # module-level "Two synthesis backends" comment
                    # describes.
                    Ffs(LessThan(2700)),
                    BlockRams(LessThan(20)),
                    DspBlocks(LessThan(1)),
                ],
            ),
            build(
                name="cnn_accel_window_gen",
                generics={
                    "g_max_kernel_size": _MAX_KERNEL_SIZE,
                    "g_max_row_tile_words": _MAX_ROW_TILE_WORDS,
                    "g_tile_channels": _TILE_CHANNELS,
                    # The CONV instance's depth, i.e. what
                    # `cnn_accel_conv_core` actually elaborates -- see
                    # `cnn_accel_constants.ASSEMBLY_BUFFERS`. The POOL
                    # build below deliberately keeps the default of 1.
                    "g_assembly_buffers": _ASSEMBLY_BUFFERS,
                },
                # M7 FIXED (was: 23950 LUTs, 199 FFs, 0 BRAM, 8 DSP on CI's
                # Yosys -- the combinational random-access read across
                # `g_max_kernel_size` rows at once blocked BRAM inference, so
                # Yosys emitted 4608 RAM64M distributed-RAM cells plus the
                # LUTs to mux them, ~37% of an XC7A100T's LUTs for one small
                # block). Per `cnn_accel_window_gen_bram_proposal.md` section
                # 7.1 (Option 3a, ratified), the line buffers are now
                # `g_max_kernel_size` (3) independently-declared `bank_mem`
                # signals inside a `generate` block -- one bank per branch,
                # matching `cnn_accel_weight_buffer.vhd`'s `memory_block`
                # idiom exactly -- instead of one shared 2D array signal.
                # Yosys's `memory_collect` could never decompose that shared
                # array into per-bank cells no matter how statically-indexed
                # each access site looked (every bank's data had to be live
                # simultaneously for the tap-assembly register); giving each
                # bank its own signal object fixes that structurally.
                #
                # Measured 2026-09 post-fix (local dev Yosys v0.68+182,
                # `synth_xilinx -family xc7`): exactly 3 RAMB36E1 (one per
                # bank, confirmed via `$mem_v2` cell count before full
                # synth), 2753 LUTs, 789 FFs, 9 DSP48E1 -- an ~89% LUT cut
                # from the old distributed-RAM figure.
                #
                # FFs/BRAM/DSP are trusted directly from this local
                # measurement: they are structural counts that don't move
                # between Yosys versions (see the module-level note above,
                # and conv_core's own comment below for the cross-check).
                # LUTs are CI-sensitive, and this project's standing rule
                # (proposal doc section 7.1 item 6) is that LUT limits go in
                # from an actual CI run, never a local Yosys -- so the
                # number below is a deliberately loose, PROVISIONAL guard
                # rail, not a tight re-baseline: it only needs to sit far
                # below the old 23950/8884 (CI/local-dev) figures to catch a
                # regression back to distributed RAM, while staying clear
                # of whatever CI's real post-fix number turns out to be.
                # This entity's own historical local-dev-to-CI ratio for the
                # old RTL was ~2.7x (23950 CI vs 8884 local-dev); applying
                # that same ratio to the new local 2753 gives a rough
                # estimate of ~7430. TODO: tighten to the real CI-measured
                # number the first time this build runs on CI.
                #
                # Loosened 2026-09 into a structural regression gate (see
                # module-level comment): LUT given ~1.75x headroom over that
                # ~7430 CI estimate (was already close to this, just rounded
                # up); FF given ~1.75x headroom over the measured 789 (FFs
                # are structural, so 789 is trusted as the real CI figure
                # too, per the note above).
                #
                # ===== DOUBLE-BUFFERED TAP ASSEMBLY (2026-09) =====
                # Re-measured (same local dev Yosys) after
                # `g_assembly_buffers` went 1 -> 3 on this, the CONV,
                # instance: **7611 LUTs, 2483 FFs, 12 RAMB36, 4 DSP48E1**.
                # BRAM and DSP are unmoved and stay exact; LUT still sits
                # far inside its existing 13000 structural gate (7611, i.e.
                # 1.71x headroom) so that limit is left alone.
                #
                # FF is the one that has to move, 1400 -> 4400: three
                # tap-assembly banks instead of one is +1152 flip-flops by
                # construction (72 bytes x 8 bits x 2 extra buffers), which
                # no headroom over the old single-buffered 789 could ever
                # have absorbed. Re-pinned with the same ~1.75x headroom
                # over the new measurement that the old limit carried over
                # the old one -- still a loose structural regression gate,
                # not a tight baseline, per this file's standing rule that
                # real LUT/FF limits come from CI and not a local Yosys.
                checkers=[
                    TotalLuts(LessThan(13000)),
                    Ffs(LessThan(4400)),
                    # Exact, not an upper bound: 0 BRAM (inference silently
                    # broken again, the whole point of M7) must fail CI just
                    # as loudly as an unexpected increase would. Safe as an
                    # equality because block-RAM counts are structural and
                    # do not move between Yosys versions -- see the
                    # additivity cross-check in conv_core's comment below.
                    # This is the one check in this whole gate that is NOT
                    # loosened: it is the entire reason the gate exists (see
                    # cnn_accel_window_gen_bram_proposal.md and the
                    # module-level comment above).
                    # Re-pinned 2026-09 (YOLOv8n sizing pass) from 3 to
                    # 12: `_MAX_ROW_TILE_WORDS` went 512 -> 1920, and a
                    # bank is `g_max_row_tile_words x 64` bits, so each of
                    # the 3 banks now needs ceil(1920/512) = 4 RAMB36
                    # instead of 1. 3 x 4 = 12, measured exactly. Still an
                    # equality for the original reason: a drop to 0 means
                    # BRAM inference broke again (M7).
                    BlockRams(EqualTo(12)),
                    DspBlocks(LessThan(12)),
                ],
            ),
            build(
                # The same entity at the POOL path's geometry -- 5 banks
                # (for the 5x5 SPPF kernel) instead of 3. This is the
                # instance 'cnn_accel_top' gives the pool lanes, and the
                # conv-geometry build above cannot see its cost at all.
                name="cnn_accel_window_gen_pool",
                top="cnn_accel_window_gen",
                generics={
                    "g_max_kernel_size": _POOL_MAX_KERNEL_SIZE,
                    "g_max_row_tile_words": _MAX_ROW_TILE_WORDS,
                    "g_tile_channels": _TILE_CHANNELS,
                    # Single-buffered on purpose: at K=5 one tap-assembly
                    # bank is 200 bytes, and the pool path is nowhere near
                    # window-generator-bound. Stated explicitly rather
                    # than left to the entity default so the difference
                    # from the conv build above is visible here.
                    "g_assembly_buffers": 1,
                },
                checkers=[
                    # Structural gate with the same generous headroom as
                    # every other Yosys entry here; the authoritative
                    # resource numbers are the Vivado twin's below.
                    TotalLuts(LessThan(24000)),
                    Ffs(LessThan(4000)),
                    # 5 banks x ceil(1920/512) = 20. Exact, same rationale
                    # as the 3-bank build above.
                    BlockRams(EqualTo(20)),
                    DspBlocks(LessThan(12)),
                ],
            ),
            build(
                name="cnn_accel_pe_array",
                generics={
                    "g_pe_rows": _PE_ROWS,
                    "g_pe_cols": _PE_COLS,
                    "g_accum_width": _ACCUM_WIDTH,
                    "g_max_kernel_size": _MAX_KERNEL_SIZE,
                    "g_tile_channels": _TILE_CHANNELS,
                    "g_weight_buffer_depth": _WEIGHT_BUFFER_DEPTH,
                },
                # Baseline 2026-09 (Yosys v0.68 release, CI run): 3474 LUTs,
                # 1119 FFs, 0 BRAM, 65 DSP.
                #
                # DSP count is 64 (g_pe_rows*g_pe_cols multiply lanes) + 1 address
                # adder that Yosys folded into a DSP48E1. Broadcast activation +
                # per-lane weight means every (row, col) PE cell does its own
                # int8 x int8 multiply every cycle (proposal doc section 6), so
                # we need g_pe_rows * g_pe_cols = 8 * 8 = 64 multiplies. With
                # g_pe_cols=1 we would only need g_pe_rows DSPs; the full grid
                # needs the full g_pe_rows*g_pe_cols count. The recorded 65
                # matches the ratified proposal's predicted figure and is not
                # an anomaly.
                #
                # BlockRams stays at LessThan(1): this entity's accumulators must
                # stay in flip-flops; any block RAM here would mean something
                # has gone wrong.
                #
                # Loosened 2026-09 into a structural regression gate (see
                # module-level comment): LUT/FF given ~1.75x headroom over
                # the 3474/1119 CI baseline. BRAM and DSP are left as the
                # meaningful, near-measured structural checks -- BRAM
                # because any BRAM here means the accumulators leaked out of
                # flip-flops, and DSP because 65 is the exact expected
                # multiply-lane count and a big drop there would mean DSP
                # packing/MAC inference broke.
                #
                # Re-pinned 2026-09 (DSP48 int8 PACKING pass, see
                # cnn_accel_pe_array.vhd's "DSP48 int8 PACKING" block):
                # measured locally 2235 LUTs, 2124 FFs, 0 BRAM, **33 DSP**
                # (was 2891/1118/0/65 locally, 3474/1119/0/65 in CI).
                #
                #  * DSP 65 -> 33 is the whole point of the change and the
                #    only checker here that is meant to be tight-ish. The
                #    MAC array is now `ceil(g_pe_rows/2) * g_pe_cols = 32`
                #    packed multiplies (two int8 MACs per multiply) instead
                #    of `g_pe_rows * g_pe_cols = 64` plain ones, plus the
                #    same 1 address adder Yosys has always folded into a
                #    DSP48E1: 32 + 1 = 33, measured exactly. Yosys maps
                #    every `*` to its own DSP48E1 unconditionally, so this
                #    number counts *multiply operators*, which is precisely
                #    what the packing halves. `LessThan(40)` is therefore
                #    the regression gate that matters now: losing the
                #    packing puts it straight back at 65.
                #  * LUT 2891 -> 2235 and FF 1118 -> 2124. The FF rise is
                #    not new registers in the dataflow -- no pipeline stage
                #    was added, `c_mac_latency` is unchanged -- it is the
                #    packed product register being 33 bits wide per row
                #    *pair* where the old one was 16 bits per row, plus
                #    Yosys not having a DSP M/P register to absorb it into
                #    the way Vivado does (Vivado's FF count goes *down*,
                #    3212 -> 2124; see the Vivado twin below). LUT/FF keep
                #    the module-level ~1.75x structural headroom.
                #
                # ========= CROSS-POSITION PIPELINING (2026-09) =======
                # Re-measured after cnn_accel_pe_array.vhd removed the
                # per-output-position 'drain'/accept overhead (see that
                # file's "Cross-position pipelining" block): **1981 LUTs,
                # 2134 FFs, 0 BRAM, 33 DSP** (was 2235 / 2124 / 0 / 33).
                # DSP is unchanged at 33 -- the packing is untouched --
                # and the +10 FFs are the three new one-bit flag
                # pipelines ('first'/'lastout', carried per stage), the
                # 'first_tile_q' latch and the 6-bit 'held_addr_q'. LUTs
                # fell because two FSM states and the per-pixel
                # accumulator-clear mux are gone. No checker moves: every
                # number is well inside the limits below.
                checkers=[
                    TotalLuts(LessThan(4000)),
                    Ffs(LessThan(3800)),
                    BlockRams(LessThan(1)),
                    DspBlocks(LessThan(40)),
                ],
            ),
            build(
                name="cnn_accel_conv_core",
                generics={
                    "g_pe_rows": _PE_ROWS,
                    "g_pe_cols": _PE_COLS,
                    "g_accum_width": _ACCUM_WIDTH,
                    "g_max_kernel_size": _MAX_KERNEL_SIZE,
                    "g_tile_channels": _TILE_CHANNELS,
                    "g_max_row_tile_words": _MAX_ROW_TILE_WORDS,
                    "g_weight_buffer_depth": _WEIGHT_BUFFER_DEPTH,
                    "g_bias_buffer_depth": _BIAS_BUFFER_DEPTH,
                },
                # M6b composition entity (window_gen -> pe_array ->
                # bias_requant, weight_buffer -> pe_array), no new datapath
                # logic of its own -- see cnn_accel_conv_core.vhd's own
                # header comment.
                #
                # Pre-M7 baseline (Yosys v0.68 release, CI run, window_gen's
                # distributed-RAM blowup still present): 35056 LUTs, 1443
                # FFs, 72 BRAM (8 RAMB36 + 64 RAMB18), 105 DSP -- almost
                # exactly the sum of the individually-measured submodules
                # (LUTs 23950 + 3474 + 7255 = 34679 vs 35056; DSPs
                # 8 + 65 + 32 = 105 exactly), i.e. a pure-wiring composition
                # entity, with the LUT figure dominated by window_gen's
                # known blowup (23950 of the 35056).
                #
                # M7 fixed window_gen's BRAM inference and M7b reworked
                # weight_buffer (see both builds above). Measured after both,
                # 2026-09 (local dev Yosys v0.68+182): 11116 LUTs, 3066 FFs,
                # 18 BRAM (3 RAMB36 + 15 RAMB18), 106 DSP -- BRAM down 72 ->
                # 18, a 75% cut, which was the whole point of M7/M7b.
                #
                # FFs/BRAM/DSP are trusted directly -- and this composition
                # entity is exactly what confirmed, twice, that those three
                # resources are Yosys-version-insensitive: each number is the
                # arithmetic sum of the four leaves as measured individually
                # above (window_gen / weight_buffer / pe_array /
                # bias_requant) --
                #   DSP:  9 + 0 + 65 + 32       = 106  (measured 106, exact)
                #   BRAM: 3 + 15 + 0 + 0        = 18   (measured 18,  exact)
                #   FF:   789 + 1093 + 1118 + 66 = 3066 (measured 3066, exact)
                #   LUT:  2753 + 881 + 2891 + 4658 = 11183 (measured 11116,
                #                    -67 from cross-boundary optimization)
                # (all four leaf figures above are the *local* measurements
                # from the same session, not the CI baselines quoted in each
                # leaf's own comment -- pe_array for instance reads 2891/1118
                # locally against its 3474/1119 CI baseline.)
                # so those three limits below are re-baselined straight from
                # this measurement. Note the FF limit had to move a long way
                # (2200 -> 3200): M7b traded ~1030 FFs for 57 BRAM in
                # weight_buffer, and this entity inherits all of it.
                #
                # Cross-checked against real vendor synthesis -- see the
                # `cnn_accel_conv_core_vivado` project registered below
                # (xc7a200tfbg484-2, out-of-context, now with its own
                # authoritative checkers -- see the module-level "Two
                # synthesis backends" comment at the top of this file):
                # 15358 LUTs, 2824 FFs, 14 RAMB36 + 2 RAMB18 (16 total),
                # 36 DSP, 0 LUTRAM. The BRAM story agrees closely (16 vs
                # Yosys's 18, both far below the old 72), which is what
                # mattered here.
                #
                # SUPERSEDED 2026-09 by the DSP48 int8 PACKING pass: the
                # long paragraph below is kept because its reasoning is
                # still instructive, but its conclusion no longer holds.
                # `cnn_accel_pe_array` no longer synthesizes to 0 DSP under
                # Vivado -- it now synthesizes to exactly 32 (8 rows) / 64
                # (16 rows), two int8 MACs per DSP48E1. The real reason
                # Vivado had put the array in fabric was NOT the
                # `is_valid`-gated accumulate guessed at below (that gate
                # was already gone with S7's pipeline, and the count stayed
                # 0): an 8x8 multiply is simply below Vivado's DSP
                # inference size threshold. Packing two weights into one
                # 25-bit operand makes it a 25x8 multiply, comfortably
                # above it. See cnn_accel_pe_array.vhd's "DSP48 int8
                # PACKING" block and the re-pinned Vivado entries below.
                #
                # DSP 36 vs 106 differs STRUCTURALLY, and -- once this
                # session added a standalone Vivado build per leaf (see
                # below) -- turned out to differ in a more interesting way
                # than first assumed. The earlier version of this comment
                # guessed the 36 came from `cnn_accel_pe_array`'s 64-lane
                # MAC array packing two int8 multiplies per DSP48E1 (32
                # DSPs). A real per-leaf Vivado measurement disproves that:
                # `cnn_accel_pe_array_vivado` alone synthesizes to *0* DSP
                # (all 64 multiplies land in LUT fabric -- see that
                # project's own comment for why). The 36 DSPs actually come
                # from `cnn_accel_bias_requant_vivado` (32, its requant
                # multiplies map straightforwardly to DSP48E1 for both
                # tools) and `cnn_accel_window_gen_vivado` (4, address
                # arithmetic); `cnn_accel_weight_buffer_vivado` contributes
                # 0. 32 + 4 + 0 + 0 = 36 exactly. Yosys, by contrast,
                # unconditionally maps every `*` operator to its own
                # DSP48E1, so it does spend 65 DSPs on `pe_array`'s MAC
                # array (see that build's own comment) -- Yosys and Vivado
                # are not just using a different packing ratio, they are
                # making entirely different LUT-vs-DSP tradeoffs for the
                # same RTL. Since xc7a200t has 740 DSPs and DSP is not the
                # scarce resource here, Vivado's choice is headroom, not a
                # problem, but it means "32 DSPs for 64 lanes" was never
                # correct as a per-entity claim about `pe_array` -- see that
                # project's own comment below.
                #   - LUT 15358 vs 11116 is the other side of that same
                #     trade (pe_array_vivado alone is 6778 LUTs, more than
                #     double Yosys's 2891, precisely because its MAC array
                #     is fabric- rather than DSP-implemented under Vivado).
                #   - Vivado puts weight_buffer's prefetch FIFO in block RAM
                #     (hence 0 LUTRAM) where Yosys uses 49 RAM32M.
                # Leaf additivity under Vivado (measured leaf sum vs. this
                # composed measurement): LUT 4906+848+2900+6778=15432 vs
                # 15358 (-74, cross-boundary optimization, same phenomenon
                # as Yosys's -67); FF 66+820+815+1126=2827 vs 2824 (-3);
                # BRAM (0+13+3+0)=16 vs 16 (exact); DSP (32+0+4+0)=36 vs 36
                # (exact). So BRAM/DSP additivity is exact under Vivado too,
                # FF is near-exact, and only LUT sees the same small
                # cross-boundary optimization gap seen under Yosys.
                #
                # LUTs remain CI-sensitive under Yosys (see window_gen's own
                # comment and the module-level note above). Composing
                # window_gen's own ~2.7x local-to-CI LUT estimate
                # (2753 -> ~7430) with pe_array's and bias_requant's
                # *unchanged* CI-measured baselines (3474 + 7255),
                # weight_buffer's new but local-only 881, and ~380 of glue,
                # gives a composed CI estimate of ~19400 LUTs -- roughly a
                # 45% reduction from the old 35056.
                #
                # Loosened 2026-09 into a structural regression gate (see
                # module-level comment): LUT given ~1.75x headroom over the
                # ~19400 CI estimate; FF given ~1.75x headroom over the
                # measured 3066 (structural, trusted directly per the note
                # above). BRAM/DSP are left as the meaningful, near-measured
                # structural checks -- 18/106 are the exact expected sums of
                # the four leaves, and either dropping sharply (BRAM
                # inference regressing) or DSP dropping sharply (MAC
                # structure lost) is exactly what this gate must still
                # catch.
                #
                # FF re-pinned 2026-09 (S7): 3066 -> 6806 measured, from
                # pipelining pe_array, bias_requant and window_gen to make
                # 150 MHz (module-level comment). ~1.5x headroom. LUT came
                # *down* to 10054 and BRAM held at 18. DSP moved 106 -> 104
                # because window_gen's two runtime address multiplies are
                # gone (its 4 DSPs -> 2 under Yosys, which unconditionally
                # maps every `*`); still far inside the limit, and still
                # catching a collapse of pe_array's 65-DSP MAC structure,
                # which is what the check is for.
                checkers=[
                    TotalLuts(LessThan(34000)),
                    # Re-pinned 2026-09-09, timing pass. Measured 10560,
                    # i.e. the old gate was passed by 60 FFs and is now
                    # exceeded by 60. The additions are all structural
                    # timing fixes with countable cost: two boundary skid
                    # buffers on this entity's 's_stream'/'m_out' ports
                    # (~2 x 130 bits of data plus control), pe_array's
                    # 'tap2_q' alignment stage, and weight_buffer's
                    # fabric-mapped weight output register (see that
                    # entity's own Yosys gate above). Re-pinned at the
                    # same loose ~1.14x this gate had.
                    Ffs(LessThan(12000)),
                    # Re-pinned 2026-09 (YOLOv8n sizing pass): measured 27
                    # (12 window_gen + 15 weight_buffer), up from 18,
                    # entirely from `_MAX_ROW_TILE_WORDS` 512 -> 1920
                    # quadrupling each window_gen row bank -- see that
                    # entity's own BlockRams checker above.
                    BlockRams(LessThan(40)),
                    # Re-pinned 2026-09 (DSP48 int8 packing pass): measured
                    # 69 (was 101). pe_array's multiply count went 65 -> 33
                    # (see its own entry above); bias_requant's 32 and
                    # window_gen's ~2-4 are untouched. `LessThan(80)` still
                    # passes at 69 but fails the ~101 an unpacked MAC array
                    # would put back, which is what this gate is for.
                    DspBlocks(LessThan(80)),
                ],
                # ========= CROSS-POSITION PIPELINING (2026-09) =======
                # Re-measured after the pe_array change above: **12045
                # LUTs, 8627 FFs, 12 RAMB36 + 15 RAMB18 (27 total), 69
                # DSP**. DSP and block RAM are bit-for-bit unchanged; the
                # LUT/FF deltas are pe_array's own, unchanged limits.
            ),
        ]

        # Vendor-accurate, AUTHORITATIVE resource backend -- one
        # `VivadoNetlistProject` per entity that has a Yosys netlist build
        # above (`cnn_accel_axi_read_dma`/`cnn_accel_ofmap_dma` excluded:
        # under concurrent development elsewhere, not yet registered here),
        # using the same generics as the matching Yosys build so the two are
        # directly comparable, plus the two `_pe_rows_16` scaled-point
        # entries at the end, which are Vivado-only by decision (this
        # project synthesizes with Vivado; the Yosys set is not extended). Registered ONLY when this machine actually
        # has Vivado -- see `ghdl_yosys_env.resolve_vivado_path()`'s
        # docstring for why that guard is mandatory rather than defensive
        # (CI's hdl-docker image has no Vivado, and
        # `build_fpga.py --netlist-builds` there builds every *registered*
        # project with no filter -- an ungated entry here would break CI on
        # a machine that was never meant to have this tool). This means the
        # whole block below, and everything it registers, is invisible to
        # CI; see the module-level "Two synthesis backends" comment at the
        # top of this file for the full split rationale, and do not read
        # "no CI coverage" as "not gating" -- these checkers are real and
        # tight, they just only ever run by hand, locally, with real Vivado.
        #
        # Every entry's `build_result_checkers` below is set directly from
        # a real local measurement (`build_fpga.py --netlist-builds
        # <name>_vivado`, 2026-09, Vivado 2026.1, this part), with a small
        # (roughly 5-10%) margin for LUT/FF to absorb minor Vivado-version
        # jitter, not a loose regression gate -- Vivado is the authoritative
        # backend, so unlike the Yosys checkers above these are meant to
        # track the real footprint closely. Re-run and re-baseline every
        # entry below by hand after any resource-affecting RTL or generic
        # change; there is no CI job that will do it for you.
        vivado_path = resolve_vivado_path()
        if vivado_path is not None:
            from tsfpga.vivado.project import VivadoNetlistProject

            def vivado_build(
                name: str,
                top: str,
                generics: dict,
                checkers: list,
                analyze_synthesis_timing: bool = False,
            ) -> VivadoNetlistProject:
                return VivadoNetlistProject(
                    name=f"{name}_vivado",
                    modules=modules,
                    part=_VIVADO_PART,
                    top=top,
                    generics=generics,
                    build_result_checkers=checkers,
                    vivado_path=vivado_path,
                    analyze_synthesis_timing=analyze_synthesis_timing,
                    defined_at=Path(__file__),
                )

            projects += [
                vivado_build(
                    "cnn_accel_bias_requant",
                    "cnn_accel_bias_requant",
                    {
                        "g_accum_width": _ACCUM_WIDTH,
                        "g_pe_rows": _PE_ROWS,
                        "g_bias_addr_width": 9,
                    },
                    # Measured 2026-09 with `analyze_synthesis_timing=True`
                    # (real 500 MHz `create_clock` active during synthesis,
                    # not just post-hoc analysis -- see the S7 module-level
                    # comment above): 4906 LUTs, 66 FFs, 0 BRAM, 32 DSP,
                    # unchanged from the unconstrained baseline (this entity
                    # was already tight enough that the extra clock
                    # constraint did not shift its mapping). DSP count
                    # matches Yosys's own 32 exactly -- this entity's
                    # requant multiply is wide enough (not a tiny int8 x
                    # int8) that both tools map it to DSP48E1 the same way,
                    # unlike pe_array (see that entry below).
                    # 150 MHz timing estimate: 520.02 MHz -- **A MEANINGLESS
                    # NUMBER, DO NOT TRUST IT.** Read the S7 module-level
                    # comment above: this entity's whole datapath cone hangs
                    # off input ports, which an out-of-context build never
                    # times, and the same logic was conv_core's 46.77 MHz
                    # critical path. Only conv_core's figure means anything
                    # for this entity.
                    #
                    # Re-measured 2026-09 after the S7 7-stage pipelining:
                    # 3163 LUTs, 2817 FFs, 0 BRAM, 32 DSP, 206.40 MHz
                    # standalone (still an upper bound, but at least the
                    # register-to-register paths it reports are now the real
                    # ones). LUTs came *down* by ~1700 because the
                    # guard/sticky rounding replaced a re-multiply, a
                    # 65-bit subtract and two 66-bit comparators; FFs are up
                    # from 66 because there are now 7 stages where there was
                    # one combinational cone. DSP held at 32, i.e. the
                    # requant multiply pattern survived the rework, which is
                    # what that `EqualTo` is for.
                    checkers=[
                        TotalLuts(LessThan(5200)),
                        # Re-pinned 2026-09-09, timing pass. Measured
                        # 3388. NOT caused by this pass:
                        # 'cnn_accel_bias_requant.vhd' is byte-identical
                        # to what it was before it (see the pass's diff),
                        # so this gate was already failing on 'main'. It
                        # predates H2 (886f896), which gave this entity
                        # 'cfg_per_channel_en' plus a per-lane
                        # (multiplier, shift) mux fed from the weight
                        # buffer's scale region -- real, feature-driven
                        # state that was never re-pinned. Measured Fmax
                        # 186.67 MHz, comfortably over 150.
                        Ffs(LessThan(3700)),
                        Ramb36(LessThan(1)),
                        Ramb18(LessThan(1)),
                        DspBlocks(EqualTo(32)),
                    ],
                    analyze_synthesis_timing=True,
                ),
                vivado_build(
                    "cnn_accel_pool",
                    "cnn_accel_pool",
                    {
                        "g_max_kernel_size": _POOL_MAX_KERNEL_SIZE,
                        "g_accum_width": _POOL_ACCUM_WIDTH,
                    },
                    # Measured 2026-09 at the 3x3 bound: 293 LUTs, 27 FFs,
                    # 0 BRAM, 0 DSP -- smaller than Yosys's 342/27/0/0 for
                    # the same RTL, a rare case where Vivado's mapping is
                    # the tighter one.
                    #
                    # Re-measured 2026-09 (YOLOv8n sizing pass) at the 5x5
                    # bound this entity now elaborates with: **899 LUTs**,
                    # 27 FFs, 0 BRAM, 0 DSP. 3.1x the LUTs for 2.8x the
                    # taps (9 -> 25 lanes), and the FF count is *unchanged*
                    # -- which is the real check here: the extra taps are
                    # pure combinational reduction feeding the same single
                    # output register, so anything that moved FFs, BRAM or
                    # DSP would mean the reduction had stopped being
                    # combinational.
                    #
                    # Re-measured 2026-09 after the 150 MHz rework: **708
                    # LUTs**, **227 FFs**, 0 BRAM, 0 DSP.
                    #
                    # The FF bound had to move, and by a lot, because the
                    # premise of the sentence above is exactly what the
                    # rework deleted. The reduction is no longer one
                    # combinational stage feeding one output register --
                    # `desc_q -> pool_lane/out_max_q` was 82 logic levels
                    # and -49.5 ns of setup slack at 150 MHz, the worst
                    # path in the whole accelerator. It is now a registered
                    # tap mask, a balanced tree split across a pipeline
                    # register, and a two-entry output buffer. The 227
                    # account for themselves exactly:
                    #
                    #   25  tap_mask_q
                    #   16  cfg_kernel_h_q / cfg_kernel_w_q (the shadow the
                    #       one-cycle `cfg_match` bubble compares against)
                    #  147  pipeline stage 1: 7 max nodes x 8 bits +
                    #       7 sum nodes x 13 bits
                    #   36  output buffer: 2 entries x (8-bit max +
                    #       16-bit sum + is_avg + last)
                    #    3  buf_count_q + p1_valid_q + the stage-1 tag/last
                    #  ---
                    #  227
                    #
                    # LUTs went DOWN, 899 -> 708, because the balanced tree
                    # is cheaper than the 25-deep linear reduce it replaces
                    # and the runtime `kernel_h * kernel_w` multiply is gone
                    # from the datapath entirely.
                    #
                    # BRAM and DSP stay pinned at zero for the original
                    # reason: this entity is compare-and-add logic, and any
                    # movement there would mean something unintended
                    # happened to the reduction.
                    checkers=[
                        TotalLuts(LessThan(900)),
                        # Re-pinned 2026-09-09, timing pass. Measured
                        # 279, +19 over the previous measurement: the
                        # second reduction-tree register stage
                        # ('p2_*_q', see 'c_split2' in cnn_accel_pool.vhd)
                        # is 2 max nodes + 2 sum nodes + 3 sidebands at
                        # this geometry. That stage is what took
                        # 'p1_max_q -> out_max_q' from -1.964 ns to off
                        # the failing list in the top-level routed build.
                        Ffs(LessThan(310)),
                        Ramb36(LessThan(1)),
                        Ramb18(LessThan(1)),
                        DspBlocks(LessThan(1)),
                    ],
                ),
                vivado_build(
                    "cnn_accel_weight_buffer",
                    "cnn_accel_weight_buffer",
                    {
                        "g_weight_buffer_depth": _WEIGHT_BUFFER_DEPTH,
                        "g_bias_buffer_depth": _BIAS_BUFFER_DEPTH,
                        "g_pe_rows": _PE_ROWS,
                        "g_pe_cols": _PE_COLS,
                        "g_accum_width": _ACCUM_WIDTH,
                    },
                    # Measured 2026-09 with `analyze_synthesis_timing=True`
                    # (real 500 MHz `create_clock` active during synthesis --
                    # see the S7 module-level comment above): 880 LUTs,
                    # 855 FFs, 11 RAMB36 + *1* RAMB18 (12 total), 0 DSP.
                    # RAMB18 dropped from the unconstrained baseline's 2 to
                    # 1 -- turning on the clock constraint changed Vivado's
                    # BRAM-cascade decision for the depth-32 prefetch FIFO,
                    # not a regression in this project's RTL. LUT/FF also
                    # moved up slightly (were 848/820) for the same reason.
                    # BRAM is close to Yosys's 15 (both far below the
                    # pre-rework 72) but not identical: Vivado also puts the
                    # depth-32 prefetch FIFO in block RAM here, where
                    # Yosys's RAM32M-based distributed-RAM mapping for that
                    # same FIFO shows up as LUTs instead (see
                    # conv_core_vivado's own comment).
                    # 150 MHz timing estimate: PASS, 257.80 MHz.
                    # ===== RE-PINNED 2026-09-09 (timing pass) ==========
                    # Measured: **1201 LUTs, 1156 FFs, 16 RAMB36 +
                    # 2 RAMB18, 0 DSP, 247.83 MHz** (gate was 950 / 900 /
                    # 11 / 1, i.e. failing on all four memory/logic
                    # checks).
                    #
                    # WHERE THE GROWTH CAME FROM -- it is legitimate, and
                    # almost all of it predates this pass. The checkers
                    # above were pinned at 587eb1f (S7). Since then
                    # 886f896 ("per-channel requant scale region ... H2")
                    # gave this entity a THIRD memory region: 'scale_mem',
                    # 'g_bias_buffer_depth' rows of
                    # 'c_scale_entry_width (40) * g_pe_rows' bits, with
                    # its own write pointer, lane counter and row-assembly
                    # register, and it widened the prefetch FIFO's payload
                    # from 32 to 40 bits plus a second region-select bit.
                    # That is a feature the ISA needs (per-channel
                    # requantization, which YOLOv8n uses throughout), not
                    # a lost inference or a duplicated memory: the region
                    # is read by 'cnn_accel_bias_requant' through
                    # 'scale_rd_data' and filled through the same fill
                    # stream with 'fill_is_scale'. The gate was simply
                    # never refreshed, and 'main' has been failing this
                    # build since that commit -- measured at 1238 / 1202 /
                    # 15 / 2 before this pass touched anything.
                    #
                    # WHAT THIS PASS CHANGED, and it is a net improvement
                    # on three of the four numbers: the prefetch FIFO now
                    # uses 'enable_output_register' and the weight region
                    # reads through the block RAM's own output register
                    # (see cnn_accel_weight_buffer.vhd's read process for
                    # the -1.273 ns / -0.985 ns paths that motivated it).
                    # LUTs 1238 -> 1201 and FFs 1202 -> 1156: the 512-bit
                    # weight output register did NOT land in fabric, it
                    # went into the RAMB36's own DO register, which is the
                    # whole point. RAMB36 15 -> 16 is the one increase --
                    # the prefetch FIFO's depth goes from 32 to 33 words
                    # ('fifo.fifo' reserves one for its output register
                    # and then requires the RAM depth to be a power of
                    # two), so it no longer shares a tile.
                    #
                    # Leaf additivity now holds EXACTLY at the conv_core
                    # level, which it did not before: conv_core's 28
                    # RAMB36 = window_gen's 12 + this entity's 16. The
                    # previous 12 + 15 = 27 never matched conv_core's
                    # measured 28, because Vivado had been giving the
                    # composed instance one tile more than the standalone
                    # one for the same FIFO.
                    checkers=[
                        TotalLuts(LessThan(1350)),
                        Ffs(LessThan(1300)),
                        Ramb36(EqualTo(16)),
                        Ramb18(EqualTo(2)),
                        DspBlocks(LessThan(1)),
                    ],
                    analyze_synthesis_timing=True,
                ),
                vivado_build(
                    "cnn_accel_window_gen",
                    "cnn_accel_window_gen",
                    {
                        "g_max_kernel_size": _MAX_KERNEL_SIZE,
                        "g_max_row_tile_words": _MAX_ROW_TILE_WORDS,
                        "g_tile_channels": _TILE_CHANNELS,
                        "g_assembly_buffers": _ASSEMBLY_BUFFERS,
                    },
                    # Measured 2026-09 (pre-S7): 2900 LUTs, 815 FFs,
                    # 3 RAMB36 + 0 RAMB18, 4 DSP.
                    # Re-measured 2026-09 after the S7 read-address timing
                    # rework: 2750 LUTs, 1171 FFs, 3 RAMB36 + 0 RAMB18,
                    # 0 DSP. Both moved checkers moved for one reason:
                    #
                    #  * DSP 4 -> 0, exact. The 4 DSP48E1s were the two
                    #    *runtime* multiplies S7 deleted -- the read
                    #    address `input_col * n_tiles + rd_tile` and the
                    #    `out_col * stride_w` / `out_row * stride_h` window
                    #    origins -- which sat combinationally in front of
                    #    the row-bank address pins and were the design's
                    #    critical path. What is left multiplies only at
                    #    `start` on narrow values, which Vivado maps to
                    #    LUTs. Kept exact (`LessThan(1)`, i.e. zero) and in
                    #    the opposite direction to before: a DSP
                    #    *reappearing* here means a runtime multiply has
                    #    crept back into the address path and the timing
                    #    fix has regressed.
                    #  * FF 815 -> 1171. S7 precomputes the whole
                    #    per-output-position geometry (window origin, real
                    #    row/column extents, per-bank kernel-row validity)
                    #    and the read address into registers advanced by
                    #    plain increments, plus one-output-row/-column
                    #    look-ahead accumulators. That is registers traded
                    #    for logic depth, and it shows up here (LUTs went
                    #    *down* 150 in the same trade).
                    #
                    # The 3-RAMB36 check is untouched and stays exact: one
                    # per kernel-row bank, agreeing with Yosys's own
                    # `BlockRams(EqualTo(3))` below -- both tools confirm
                    # the M7 BRAM-inference fix, so this checker is kept
                    # just as exact here, on the authoritative backend, as
                    # it is on the CI-gating one.
                    #
                    # Re-measured 2026-09 (YOLOv8n sizing pass, three
                    # changes at once -- `_MAX_ROW_TILE_WORDS` 512 -> 1920,
                    # the new `cfg_pad_value`, and the `kr_base_q` timing
                    # rework): **3314 LUTs, 1202 FFs, 12 RAMB36 + 0 RAMB18,
                    # 4 DSP, 183.92 MHz** (was 2785/1171/3/0/0 at
                    # 165.73 MHz).
                    #
                    #  * RAMB36 3 -> 12, exact. A bank is
                    #    `g_max_row_tile_words x 8 * g_tile_channels` bits;
                    #    1920 x 64 needs 4 RAMB36 where 512 x 64 needed 1.
                    #    3 banks x 4 = 12. Kept exact for the M7 reason.
                    #  * DSP 0 -> 4. This checker was previously an
                    #    *exactly zero* guard whose stated meaning was "a
                    #    runtime multiply has crept back into the read
                    #    address path". It has not: Vivado's synthesis log
                    #    names the four DSPs as `col_step_words_q`,
                    #    `row_start_words_q` and `rd_base_q` -- the S7
                    #    look-ahead accumulators' *seed* values, computed
                    #    from `stride * n_tiles` at `start` and at output-
                    #    row boundaries, never per read cycle. They became
                    #    DSPs only because the deeper row buffer widened
                    #    `word_t` from 9 to 11 bits. The max-frequency
                    #    estimate going *up* (165.73 -> 183.92 MHz) is the
                    #    corroborating evidence that nothing moved onto the
                    #    critical path. Pinned exactly at 4 so a fifth --
                    #    which would be a genuinely new multiply -- fails.
                    #  * Fmax up despite the bigger memory: the `kr_base_q`
                    #    rework took the runtime `kr * kernel_w` out of the
                    #    per-cycle tap-index decode (see that signal's
                    #    declaration in cnn_accel_window_gen.vhd).
                    #
                    # ===== DOUBLE-BUFFERED TAP ASSEMBLY (2026-09) =====
                    # Re-measured after `g_assembly_buffers` went 1 -> 3 on
                    # this (the CONV) instance: **5108 LUTs, 2451 FFs,
                    # 12 RAMB36 + 0 RAMB18, 4 DSP, 183.39 MHz** (was
                    # 3314 / 1202 / 12 + 0 / 4 / 183.92 MHz).
                    #
                    #  * FF 1202 -> 2451, +1249. Two extra tap-assembly
                    #    banks at `MAX_KERNEL_SIZE**2 * TILE_CHANNELS` = 72
                    #    bytes = 576 FFs each is +1152; the remaining ~92
                    #    are the per-buffer `full_q`/`meta_*_q` sidebands,
                    #    the `buf_*_q` pointers, the `n_res_q`/
                    #    `anchor_res_q` reservation queue and the
                    #    `kr_base_walk_q`/`row_ok_walk_q` per-walk
                    #    snapshots. This is the whole cost of the change
                    #    and it is exactly what was predicted.
                    #  * LUT 3314 -> 5108, +1794. Dominated by the 3:1
                    #    output mux over 576 bits of tap data (~576 LUT6,
                    #    a 4:1 mux being one LUT6 per bit) plus the
                    #    per-buffer write-enable decode the capture stage
                    #    now needs on all 72 lanes x 3 buffers.
                    #  * **Fmax essentially unmoved**, 183.92 ->
                    #    183.39 MHz (-0.3%, i.e. inside the noise of this
                    #    estimate), which is the load-bearing number here:
                    #    the extra buffers are pure fan-out on already-
                    #    registered paths, and deleting the
                    #    `kc_q = kernel_w_q` drain state plus the
                    #    pending-window term of `s_stream_s2m.ready` took
                    #    logic *out* of the control cone, paying for the
                    #    wider output mux. 22% above the 150 MHz target.
                    #  * RAMB36 and DSP deliberately still EXACT and
                    #    unmoved -- the row banks and the `start`-time
                    #    look-ahead seeds are untouched by this rework, so
                    #    any movement there would mean something
                    #    unintended happened to the memory or the address
                    #    arithmetic.
                    #
                    # LUT/FF re-pinned at the same ~1.27x headroom over the
                    # measurement that the previous pair carried (4200 over
                    # 3314, 1500 over 1202).
                    checkers=[
                        TotalLuts(LessThan(6500)),
                        Ffs(LessThan(3100)),
                        Ramb36(EqualTo(12)),
                        Ramb18(LessThan(1)),
                        DspBlocks(EqualTo(4)),
                    ],
                    # Added 2026-09 (S7): this entity had never been timed,
                    # which is precisely why its 20.4 ns / 21-CARRY4
                    # address path went unnoticed until it surfaced as
                    # conv_core's critical path after pe_array and
                    # bias_requant were fixed. Same caveat as every other
                    # leaf here: the number it prints is an upper bound (its
                    # geometry cone is driven from `start`-time input
                    # ports), so it is worth watching for regressions but
                    # conv_core's figure is the one that decides.
                    analyze_synthesis_timing=True,
                ),
                vivado_build(
                    # The SAME entity as above, elaborated at the POOL
                    # path's geometry instead of the conv path's: this is
                    # the instance 'cnn_accel_top' gives the pool lanes,
                    # and it is where the 5x5 pool kernel is actually paid
                    # for (banks scale as K, and each bank is
                    # 'g_max_row_tile_words x 64' bits). It exists as its
                    # own netlist build because the conv-geometry build
                    # above cannot see that cost at all, and a resource
                    # regression nobody measures is a resource regression
                    # nobody notices.
                    "cnn_accel_window_gen_pool",
                    "cnn_accel_window_gen",
                    {
                        "g_max_kernel_size": _POOL_MAX_KERNEL_SIZE,
                        "g_max_row_tile_words": _MAX_ROW_TILE_WORDS,
                        "g_tile_channels": _TILE_CHANNELS,
                        "g_assembly_buffers": 1,
                    },
                    # Measured 2026-09 (first measurement of this build):
                    # **9169 LUTs, 2316 FFs, 20 RAMB36 + 0 RAMB18, 4 DSP,
                    # 185.19 MHz**.
                    #
                    # This is what the 5x5 pool kernel costs, and it is
                    # worth being explicit that the cost is superlinear in
                    # LUTs: the tap-assembly capture stage decodes a write
                    # enable for every one of `g_max_kernel_size**2 *
                    # g_tile_channels` lanes (200 at K=5, 72 at K=3), from
                    # each of `g_max_kernel_size` banks -- so ~2.8x the
                    # LUTs of the conv-geometry build for 1.67x the banks.
                    # That is exactly why `g_max_pool_kernel_size` is a
                    # separate generic and the conv path was NOT raised to
                    # 5: this cost is paid once, by the one instance that
                    # needs it.
                    #
                    # 185.19 MHz is after the `kr_base_q` rework. Before
                    # it, this build was the one entity in the whole design
                    # that MISSED the 150 MHz target -- 139.55 MHz, with
                    # the critical path running `kr_capture_q` -> the
                    # runtime `kr * kernel_w` multiply -> all 200 assembly
                    # lanes' clock enables. See that signal's declaration
                    # in cnn_accel_window_gen.vhd.
                    checkers=[
                        TotalLuts(LessThan(11000)),
                        Ffs(LessThan(2800)),
                        # One RAMB36 per kernel-row bank per 512 words of
                        # depth: 5 banks x ceil(1920/512) = 5 x 4 = 20.
                        # Exact for the same reason the conv-geometry
                        # build's 12 is: a drop to 0 means block-RAM
                        # inference broke again (M7), and an increase means
                        # the bank geometry changed without anyone saying so.
                        Ramb36(EqualTo(20)),
                        Ramb18(LessThan(1)),
                        # The same four look-ahead-seed DSPs the
                        # conv-geometry build has, and for the same reason
                        # -- they are row-count-independent.
                        DspBlocks(EqualTo(4)),
                    ],
                    # ===== DOUBLE-BUFFERED TAP ASSEMBLY (2026-09) =====
                    # Re-measured after that rework, which this instance
                    # deliberately does NOT buy into (`g_assembly_buffers`
                    # stays 1 here -- a second 200-byte tap bank at K=5 for
                    # a path that is not window-generator-bound): **8522
                    # LUTs, 2367 FFs, 20 RAMB36 + 0 RAMB18, 4 DSP,
                    # 180.96 MHz** (was 9169 / 2316 / 20 + 0 / 4 /
                    # 185.19 MHz).
                    #
                    # LUTs went *down* 647 (-7%) and FFs up only 51 (+2%):
                    # the single-buffered path still gains the parts of the
                    # rework that cost nothing -- the `kc_q = kernel_w_q`
                    # drain state and the idle re-arm cycle are gone, and
                    # `s_stream_s2m.ready` lost its pending-window term --
                    # while paying only for the reservation queue and the
                    # per-walk geometry snapshots. Fmax 185.19 -> 180.96
                    # MHz, -2.3%, still 21% clear of the 150 MHz target.
                    # Checkers left as they were (8522 < 11000,
                    # 2367 < 2800): both moved in the safe direction, and
                    # re-pinning a gate downward on a build this rework was
                    # meant NOT to touch would only make it fragile.
                    analyze_synthesis_timing=True,
                ),
                vivado_build(
                    "cnn_accel_pe_array",
                    "cnn_accel_pe_array",
                    {
                        "g_pe_rows": _PE_ROWS,
                        "g_pe_cols": _PE_COLS,
                        "g_accum_width": _ACCUM_WIDTH,
                        "g_max_kernel_size": _MAX_KERNEL_SIZE,
                        "g_tile_channels": _TILE_CHANNELS,
                        "g_weight_buffer_depth": _WEIGHT_BUFFER_DEPTH,
                    },
                    # Measured 2026-09 with `analyze_synthesis_timing=True`
                    # (real 500 MHz `create_clock` active during synthesis --
                    # see the S7 module-level comment above): 7068 LUTs,
                    # 1136 FFs (were 6778/1126 unconstrained -- both still
                    # comfortably inside the `LessThan` margins below, so no
                    # checker change needed here), 0 BRAM, *0* DSP. This is
                    # SUPERSEDED 2026-09 (DSP48 int8 packing) -- the
                    # paragraph below records why this entity USED to
                    # synthesize to 0 DSP and why that guess was wrong; the
                    # current numbers are in the block after it.
                    # the one genuinely surprising number in this whole set:
                    # constraint/prior assumption said
                    # Vivado would pack the 64 int8 x int8 MAC lanes two per
                    # DSP48E1 (32 DSPs), by analogy with how it packs
                    # `bias_requant`'s multiply. A real measurement says
                    # otherwise -- Vivado's default synthesis puts the
                    # entire MAC array in LUT fabric here (hence the LUT
                    # count more than double Yosys's 2891 for the same
                    # generics). The likely reason is structural, not a
                    # fluke: `compute_partial_sums` in
                    # `cnn_accel_pe_array.vhd` gates each lane's accumulate
                    # with an `is_valid` boolean between the multiply and
                    # the add (`if is_valid then result(r) := result(r) +
                    # resize(product, g_accum_width)`), which does not match
                    # Vivado's default DSP48E1 MACC inference template as
                    # cleanly as a plain unconditional multiply-accumulate
                    # does. Do NOT "fix" this checker by assuming DSP usage
                    # should be higher -- 0 is the real, reproducible
                    # number (confirmed via conv_core_vivado's own additive
                    # DSP breakdown below: 32 + 4 + 0 + 0 = 36, no room left
                    # for a hidden pe_array contribution). Yosys, which
                    # unconditionally maps every `*` to its own DSP48E1,
                    # does not share this behavior (65 DSPs there) -- see
                    # that entity's own comment above and the module-level
                    # "Two synthesis backends" comment for why neither
                    # number is wrong, just answering a different question.
                    # The zero-BRAM check (Ramb36/Ramb18, both LessThan(1))
                    # is unchanged from the Yosys entry's own
                    # `BlockRams(LessThan(1))` rule: this entity's
                    # accumulators must stay in flip-flops on both backends.
                    # 150 MHz timing estimate: was **FAIL**, 56.52 MHz --
                    # roughly a third of target, the deep MAC-array
                    # accumulate chain. Fixed by S7 fix 1 (see the
                    # module-level comment): `compute_partial_sums` became a
                    # self-timed pipeline with one register per adder-tree
                    # level, now **232.67 MHz**. Re-measured: 6297 LUTs,
                    # 3212 FFs, 0 BRAM, 0 DSP. FFs nearly tripled (1118 ->
                    # 3212) buying 4.1x the clock -- that is the trade this
                    # entity exists to make, so the FF limit below is a
                    # deliberate ceiling, not an accident.
                    #
                    # Note the intermediate data point, worth keeping: a
                    # single balanced adder tree with no internal registers
                    # only reached 110 MHz. Both the rebalance *and* the
                    # per-level register were needed.
                    #
                    # ================= DSP48 int8 PACKING (2026-09) ======
                    # Re-measured after `cnn_accel_pe_array.vhd` started
                    # packing TWO int8 MACs into one DSP48E1 -- see that
                    # file's "DSP48 int8 PACKING" block for the design and
                    # the (exhaustively checked) overflow bound. This is a
                    # pure MAPPING change: no pipeline stage was added or
                    # removed, `c_mac_latency` is unchanged, and every
                    # simulation expected value is bit-identical.
                    #
                    # **1852 LUTs, 2124 FFs, 0 BRAM, 32 DSP, 205.97 MHz**
                    # (was 6297 / 3212 / 0 / **0** / 232.67 MHz).
                    #
                    #  * DSP 0 -> 32, EXACTLY `g_pe_rows/2 * g_pe_cols`.
                    #    The comment above used to say Vivado would not
                    #    put this MAC array in DSPs; the reason was simply
                    #    that an 8x8 multiply is below Vivado's DSP
                    #    inference size threshold. The packed operand is
                    #    25x8, well above it, and the Vivado synthesis log
                    #    reports mode `((D+A)*B2)` per cell -- so the
                    #    pack's own adder is absorbed into the DSP48E1
                    #    PRE-ADDER and costs no fabric at all, and
                    #    `tap_q`/`dsp_q` are absorbed as the DSP's own
                    #    input/M registers. 2 MACs per DSP, as intended.
                    #    Pinned with `EqualTo`: this number IS the
                    #    packing's structural signature -- 64 would mean
                    #    the packing was lost and every MAC got its own
                    #    DSP, 0 would mean it fell back to fabric.
                    #  * LUT 6297 -> 1852 (-71%), which is the entire
                    #    point: LUTs, not DSPs, are what limits how wide
                    #    this array can grow on an XC7A200T (134600 LUT /
                    #    740 DSP).
                    #  * FF 3212 -> 2124 (-34%), because the product
                    #    register moved inside the DSP.
                    #  * Fmax 232.67 -> 205.97 MHz, an 11% drop and still
                    #    37% above the 150 MHz target, so no extra
                    #    pipeline stage was added to buy it back (the
                    #    DSP48E1's combinational A/B -> MREG path is
                    #    simply longer than a fabric 8x8 multiply's, and
                    #    the entity is no longer anywhere near critical --
                    #    conv_core, the figure that actually decides, went
                    #    *up*, 170.33 -> 178.57 MHz).
                    #  * A deliberate non-result: with the multiply in a
                    #    DSP, Vivado also wanted to absorb the reduction
                    #    tree into the DSPs' post-adder/PCIN cascade (60
                    #    DSPs, 1360 LUTs, same 205.97 MHz). That mapping
                    #    is correct but spends 28 extra DSP48E1s of the
                    #    budget this change exists to protect, so
                    #    `reduce_q` carries `use_dsp = "no"` -- see its
                    #    declaration in cnn_accel_pe_array.vhd. The 492
                    #    LUTs that costs buy back 28 DSPs.
                    #
                    # ===== CROSS-POSITION PIPELINING (2026-09) =======
                    # Re-measured after cnn_accel_pe_array.vhd removed the
                    # per-output-position pipeline overhead (its
                    # "Cross-position pipelining" block): the array now
                    # sustains one group per cycle across output-pixel
                    # boundaries instead of paying 'T + c_mac_latency'
                    # extra cycles per pixel -- 3x3 conv went 15 -> 9
                    # cycles per output position, the exact
                    # 'T*num_groups' ideal.
                    #
                    # **1860 LUTs, 2142 FFs, 0 BRAM, 32 DSP,
                    # 205.97 MHz** (was 1852 / 2124 / 0 / 32 / 205.97,
                    # re-measured on this machine in the same session, not
                    # quoted from the comment above).
                    #
                    #  * DSP **exactly unchanged at 32**, which is the
                    #    point of the `EqualTo` below: the throughput
                    #    rework had to leave the int8 packing alone, and
                    #    it did -- `tap_q`/`dsp_q` are still absorbed as
                    #    the DSP48E1's own input/M registers even though
                    #    they now carry a clock enable (`pipe_en` maps to
                    #    the DSP's CE, not to fabric).
                    #  * FF +18: the 'first'/'lastout' flag pipelines
                    #    (2 bits per MAC stage), 'first_tile_q' and the
                    #    6-bit 'held_addr_q'. LUT +8, i.e. the two
                    #    removed FSM states very nearly pay for the
                    #    global clock enable.
                    #  * **Fmax bit-identical at 205.97 MHz** -- the
                    #    enable fan-out and the late-select address mux
                    #    cost nothing measurable, and the entity stays
                    #    37% above the 150 MHz target.
                    #  * Every simulation expected value is unchanged
                    #    (141/141 `cnn_accel.*`): this is a scheduling
                    #    change, so any shift would be a bug.
                    #
                    # Checkers below are NOT moved: 1860 < 2050 and
                    # 2142 < 2350 still, so the existing ~10% margins
                    # absorb this change without being re-slackened.
                    checkers=[
                        TotalLuts(LessThan(2050)),
                        Ffs(LessThan(2350)),
                        Ramb36(LessThan(1)),
                        Ramb18(LessThan(1)),
                        DspBlocks(EqualTo(32)),
                    ],
                    analyze_synthesis_timing=True,
                ),
                vivado_build(
                    "cnn_accel_conv_core",
                    "cnn_accel_conv_core",
                    {
                        "g_max_kernel_size": _MAX_KERNEL_SIZE,
                        "g_pe_rows": _PE_ROWS,
                        "g_pe_cols": _PE_COLS,
                        "g_accum_width": _ACCUM_WIDTH,
                        "g_tile_channels": _TILE_CHANNELS,
                        "g_max_row_tile_words": _MAX_ROW_TILE_WORDS,
                        "g_weight_buffer_depth": _WEIGHT_BUFFER_DEPTH,
                        "g_bias_buffer_depth": _BIAS_BUFFER_DEPTH,
                    },
                    # Measured 2026-09 with `analyze_synthesis_timing=True`
                    # (real 500 MHz `create_clock` active during synthesis --
                    # see the S7 module-level comment above): 16115 LUTs,
                    # 3080 FFs, 10 RAMB36 + 2 RAMB18 (12 total), 36 DSP,
                    # 0 LUTRAM -- the system-footprint number ("does the
                    # accelerator fit an XC7A200T") this whole Vivado
                    # backend exists for. LUT/FF grew and RAMB36 shrank
                    # (were 15358/2824/14 unconstrained) for the same reason
                    # as weight_buffer above: the real clock constraint
                    # changes Vivado's area/timing tradeoffs, it is not a
                    # regression. DSP additivity is unaffected by the
                    # constraint; see the Yosys `cnn_accel_conv_core` entry
                    # above for the full leaf-additivity cross-check.
                    # 150 MHz timing estimate: was **FAIL**, 46.77 MHz.
                    # **This is the only entity in this file whose timing
                    # number was ever trustworthy** -- see the S7
                    # module-level comment. It took all three S7 reworks
                    # (pe_array, bias_requant, window_gen), and after the
                    # first two this number had moved only 46.77 -> 47.66,
                    # because each fix merely exposed the next comparable
                    # path.
                    #
                    # Final S7 measurement: **159.95 MHz** (PASS), 12951
                    # LUTs, 7760 FFs, 14 RAMB36 + 2 RAMB18, 32 DSP.
                    #
                    # Both moved memory/DSP checkers are now *exactly*
                    # leaf-additive, which is the real evidence the rework
                    # did not quietly break inference:
                    #   DSP    32 = 32 (bias_requant) + 0 (window_gen, was
                    #                4 runtime address multiplies) + 0 + 0
                    #   RAMB36 14 = 11 (weight_buffer) + 3 (window_gen)
                    # RAMB36 went 10 -> 14 purely from the bias_requant
                    # rework, which touches no memory at all: `bias_rd_data`
                    # now lands on a clean stage-1 register instead of
                    # feeding a 21 ns combinational cone, so Vivado no
                    # longer partially dissolves weight_buffer's bias
                    # memory into fabric. The old 10 was the anomaly; 14 is
                    # the leaf-additive truth. Do not "restore" it.
                    #
                    # Re-measured 2026-09 (YOLOv8n sizing pass): **14196
                    # LUTs, 8727 FFs, 27 RAMB36 + 2 RAMB18, 36 DSP,
                    # 170.33 MHz** (was 14279/8708/18/2/32 at 159.95 MHz
                    # when this pass started -- note the FF and RAMB36
                    # checkers below were ALREADY failing at that starting
                    # point, i.e. they were stale before this change, not
                    # broken by it).
                    #
                    # Leaf additivity still holds for the memory and DSP
                    # counts, measured hierarchically:
                    #   RAMB36 27 = 12 (window_gen, 3 banks x 4) + 15
                    #                (weight_buffer)
                    #   DSP    36 = 32 (bias_requant) + 4 (window_gen
                    #                look-ahead seeds) + 0 + 0
                    # weight_buffer contributes 15 RAMB36 here against its
                    # own standalone build's 11 -- the same "Vivado
                    # dissolves part of the bias memory into fabric,
                    # depending on what feeds it" variability the previous
                    # comment above already documents in the other
                    # direction. The window_gen delta (3 -> 12) is the
                    # whole of the intended `_MAX_ROW_TILE_WORDS` cost.
                    #
                    # Timing went UP, 159.95 -> 170.33 MHz, despite the
                    # bigger row buffers, for the `kr_base_q` reason in
                    # window_gen's own comment above. This is still the
                    # only timing number in this file that is worth
                    # trusting, and it is comfortably above 150 MHz.
                    #
                    # Re-measured 2026-09 after conv started honouring the
                    # ISA v2.1 `pad_value` (this entity's `cfg_pad_value`
                    # is now a real input instead of a tie-off): **14487
                    # LUTs, 8735 FFs, 27 RAMB36 + 2 RAMB18, 36 DSP,
                    # 170.33 MHz**, against 14196/8727/27/2/36 at the same
                    # 170.33 MHz before it. That is +291 LUTs and +8 FFs
                    # and nothing else, and both halves are exactly what
                    # the change predicts: the 8 FFs are the `pad_value_q`
                    # register that latches the field at `start`, and the
                    # LUTs are the per-window clear of the tap-assembly
                    # register, which used to be a constant-0 synchronous
                    # reset on `g_max_kernel_size**2 * g_tile_channels`
                    # (72) byte lanes and is now a fill from a runtime
                    # value. Memory, DSP and Fmax are all bit-for-bit
                    # unchanged -- the fill sits on the per-window restart
                    # path, not on the per-cycle read or capture path, and
                    # `window_gen`'s own netlist build is byte-identical
                    # (its RTL was not touched at all: it already had the
                    # port).
                    #
                    # ================= DSP48 int8 PACKING (2026-09) ======
                    # Re-measured after `cnn_accel_pe_array.vhd` started
                    # packing two int8 MACs per DSP48E1 (see that file and
                    # the pe_array Vivado entry above):
                    # **10179 LUTs, 7647 FFs, 27 RAMB36 + 2 RAMB18,
                    # 68 DSP, 178.57 MHz** (was 14487 / 8735 / 27 + 2 /
                    # 36 / 170.33 MHz). This entity is where the change is
                    # actually paid for and collected, so these are the
                    # numbers that count:
                    #
                    #  * LUT 14487 -> 10179, **-29.7%**. The saving is
                    #    exactly pe_array's own (6297 -> 1852) plus a
                    #    little cross-boundary noise, and it is a saving of
                    #    the resource that limits array width.
                    #  * DSP 36 -> 68, still exact leaf additivity:
                    #    32 (pe_array, NEW -- 64 MACs at 2 per DSP)
                    #    + 32 (bias_requant, unchanged)
                    #    + 4 (window_gen address arithmetic, unchanged)
                    #    = 68. 9.2% of the XC7A200T's 740.
                    #  * FF 8735 -> 7647: the MAC product registers moved
                    #    inside the DSP48E1s.
                    #  * BRAM unchanged at 27 + 2, as it must be -- this
                    #    change touches nothing but the multiply's mapping.
                    #  * Fmax 170.33 -> **178.57 MHz**, i.e. timing did not
                    #    just hold, it improved, and no extra pipeline
                    #    stage was spent to get there.
                    #
                    # Every conv testbench value is bit-identical (141/141
                    # `cnn_accel.*` VUnit tests unchanged) -- the packing
                    # is a mapping change, so an expected-value shift here
                    # would be a bug, never a new baseline.
                    #
                    # ===== CROSS-POSITION PIPELINING (2026-09) =======
                    # Re-measured after the pe_array throughput rework
                    # (see the pe_array Vivado entry above): **10197
                    # LUTs, 7668 FFs, 27 RAMB36 + 2 RAMB18, 68 DSP,
                    # 178.57 MHz** (was 10179 / 7647 / 27 + 2 / 68 /
                    # 178.57). +18 LUTs and +21 FFs -- pe_array's own
                    # delta and nothing else -- with DSP, block RAM and
                    # **Fmax all bit-identical**. What it buys, measured
                    # from a VCD of `test_bitexact_full_throughput` at
                    # zero stall (accumulator output beats per output
                    # position): 3x3 conv 15.0 -> **9.0** cycles/position
                    # (the `T*num_groups` ideal, PE array 100% busy),
                    # 3-tile 3x3 35.0 -> **27.25**, 1x1 conv 7.0 ->
                    # **4.0** -- and 1x1 is no longer pe_array-bound at
                    # all (it needs only 1 cycle/position now; the 4 is
                    # cnn_accel_window_gen's per-window column walk, the
                    # next bottleneck to attack).
                    # Checkers unchanged: 10197 < 11200, 7668 < 8400.
                    #
                    # ===== DOUBLE-BUFFERED TAP ASSEMBLY (2026-09) =====
                    # Re-measured after `cnn_accel_window_gen` gained
                    # `g_assembly_buffers` (= 3 here, see that entry
                    # above): **12039 LUTs, 8921 FFs, 27 RAMB36 +
                    # 2 RAMB18, 68 DSP, 178.57 MHz** (was 10197 / 7668 /
                    # 27 + 2 / 68 / 178.57).
                    #
                    # +1842 LUTs and +1253 FFs, against window_gen's own
                    # standalone delta of +1794 / +1249 -- i.e. the whole
                    # composition's growth IS the window generator's, to
                    # within 48 LUTs and 4 FFs of boundary optimization.
                    # Nothing else moved: block RAM, **DSP (68 = 32
                    # pe_array + 32 bias_requant + 4 window_gen) and Fmax
                    # are all bit-identical**, which is the check that
                    # matters -- the PE array's DSP packing and the
                    # critical path are untouched by a change that is
                    # purely window_gen-internal scheduling.
                    #
                    # What it buys, measured the same way as the
                    # cross-position pipelining entry above (VCD of
                    # `test_bitexact_full_throughput` at zero stall,
                    # accumulator output beats per output position):
                    # **1x1 conv 4.0 -> 1.0 cycles/position** (the PE
                    # array's own ideal for a 1x1 window, so the array is
                    # now 100% busy there too), 3x3 **9.0 -> 9.0** and
                    # 3-tile 3x3 **27.25 -> 27.25** (both already PE-array-
                    # bound, so unchanged by construction).
                    #
                    # Re-pinned at the same ~1.10x headroom the previous
                    # pair carried (11200 over 10197, 8400 over 7668).
                    # ===== RE-PINNED 2026-09-09 (timing pass) ==========
                    # Measured: **11314 LUTs, 9361 FFs, 28 RAMB36 +
                    # 2 RAMB18, 68 DSP**. Only the RAMB36 gate moves,
                    # 27 -> 28, and it was already failing on 'main'
                    # before this pass.
                    #
                    # 28 is what leaf additivity actually predicts now
                    # that 'cnn_accel_weight_buffer_vivado' is measured
                    # honestly: window_gen 12 + weight_buffer 16 = 28.
                    # The old 27 came from adding window_gen's 12 to a
                    # STALE weight_buffer figure of 15 (see that build's
                    # own comment); the composed instance has been at 28
                    # all along. Nothing about this entity's block-RAM
                    # inference changed -- no region moved to distributed
                    # RAM, no memory is duplicated.
                    #
                    # DSP stays exactly 68 = 64 (bias_requant, 4 per
                    # requant lane x 16... at 8 rows: 4 x 8 = 32, plus
                    # pe_array's 32 packed int8 MACs) + 4 (window_gen
                    # address arithmetic), i.e. the DSP48 int8 packing is
                    # intact after the 'tap2_q' alignment stage and the
                    # bounded 'c_kernel_bits' geometry arithmetic.
                    # ===== SCALE REGION MOVED TO LUTRAM (round-3 timing
                    # pass, 175 MHz exploration) =======================
                    # 'weight_buffer_inst/scale_mem' is no longer mapped to
                    # block RAM in THIS COMPOSED build. Vivado says so
                    # explicitly, once, in the synthesis log:
                    #
                    #   [Synth 8-5584] The signal
                    #   "i_0/weight_buffer_inst/scale_mem_reg" is
                    #   implemented as distributed LUT RAM for the
                    #   following reason(s): The timing constraints
                    #   suggest that the chosen mapping will yield better
                    #   timing results.
                    #
                    # It is the ONLY signal that moved -- there is exactly
                    # one 8-5584 message in the log -- and the arithmetic
                    # checks out: 'scale_mem' is 'g_bias_buffer_depth' (8)
                    # rows of 'c_scale_entry_width' (40) x 'g_pe_rows'
                    # bits, i.e. 8 x 320 at 8 rows, which is exactly the
                    # 4 RAMB36 + 1 RAMB18 that disappeared (and 8 x 640,
                    # 9 RAMB36, at 16 rows).
                    #
                    # THIS IS THE DOCUMENTED DESIGN INTENT, not drift.
                    # See 'g_bias_buffer_depth' on the entity: the bias/
                    # scale region is sized independently of the weight
                    # region precisely "so it can fall out of block RAM
                    # into LUTRAM/registers". Spending five block RAMs to
                    # hold 2 560 bits uses ~1.4 % of them; 216 LUTRAMs is
                    # the better mapping and Vivado picked it on the
                    # merits. The WEIGHT region -- the one that must stay
                    # block RAM -- is untouched, and 'pe_array's and
                    # 'window_gen's own inference is unchanged (both leaf
                    # builds still pass their own exact RAMB pins).
                    #
                    # What made Vivado re-decide: the round-3 setup-stage
                    # split in 'cnn_accel_window_gen' (the DSP product and
                    # its fixup arithmetic no longer share a cycle). That
                    # shortened window_gen's worst cone, which changes
                    # which paths look critical under the SYNTHETIC
                    # 500 MHz ('create_clock -period 2.000') constraint
                    # tsfpga puts on a netlist build -- a target the design
                    # misses by several ns, so the mapping heuristic is
                    # running at its most aggressive. Bisected: pristine
                    # HEAD gives 28/2 and 0 LUTRAMs, HEAD + the window_gen
                    # change alone already gives 24/1 and 216.
                    #
                    # It does NOT happen in any real build. The
                    # 'cnn_accel_top_build' place-and-route run at both
                    # 6.667 ns and 5.714 ns reports 62 RAMB36 + 2 RAMB18
                    # and **0 LUT as Distributed RAM**, with
                    # 'weight_buffer_inst' holding its full 16 + 2 -- i.e.
                    # at any constraint the design can actually meet,
                    # 'scale_mem' stays in block RAM. Only the
                    # out-of-context estimate vehicle flips.
                    #
                    # Re-pinned rather than widened: the numbers below are
                    # exact measurements, and a 'LutRams' pin is ADDED so
                    # the region cannot drift any further unnoticed -- if
                    # it ever fell out of LUTRAM into flip-flops (the
                    # failure shared/TimingAndResources.md section 7 is
                    # really about) that pin fires. Leaf additivity still
                    # holds for everything that stayed: window_gen 12 +
                    # weight_buffer 16 = 28, less the 4 that scale_mem
                    # took with it = 24.
                    checkers=[
                        TotalLuts(LessThan(13200)),
                        Ffs(LessThan(9800)),
                        Ramb36(EqualTo(24)),
                        Ramb18(EqualTo(1)),
                        LutRams(EqualTo(216)),
                        DspBlocks(EqualTo(68)),
                    ],
                    analyze_synthesis_timing=True,
                ),
                # Local tensor scratchpad (doc/cnn_accel_top_v2_arch.md
                # section 4). No Yosys twin, by the same standing decision
                # as the two scaled points below (this project synthesizes
                # cnn_accel with Vivado only) -- and this entity had NO
                # netlist build at all until now, so its cost (in
                # particular the per-channel landing/skid register and the
                # 2:1 output mux added in front of each read channel) had
                # never been measured. Default generics (`g_num_banks`=2,
                # `g_bank_words`=1024, `g_data_width`=64): no project-wide
                # constant exists for this entity's sizing the way
                # `_PE_ROWS`/`_TILE_CHANNELS`/etc. do for the PE-array
                # family, so there is no "different convention" to follow.
                vivado_build(
                    "cnn_accel_tensor_mem",
                    "cnn_accel_tensor_mem",
                    {},
                    # First-ever measurement of this entity (it had no
                    # netlist build at all before now), and it took two
                    # real RTL fixes to get a meaningful number -- see
                    # 'bank_gen's own header comment in
                    # cnn_accel_tensor_mem.vhd for the full story:
                    #   1. The old 'bank_ram' was ONE shared 2D array
                    #      signal (`bank_mem_arr_t`, an array of
                    #      `bank_mem_t`) indexed by bank number, not one
                    #      independent per-bank signal the way
                    #      cnn_accel_window_gen.vhd's/
                    #      cnn_accel_weight_buffer.vhd's own 'bank_mem'/
                    #      'memory_block' idiom already does. Vivado
                    #      cannot map that shape to a RAMB36 at all ("
                    #      Potential Runtime issue for 3D-RAM or RAM from
                    #      Record/Structs") and fell back to one flip-flop
                    #      per bit: 76421 LUTs, 131664 FFs, 0 BRAM for the
                    #      default 2-bank/1024-word/64-bit point --
                    #      obviously not "what the mux costs".
                    #   2. Splitting `bank_ram` into a per-bank 'bank_mem'
                    #      fixed the FF blowup (789 FFs) but Vivado still
                    #      would not use block RAM: 11030 LUTs (2774 logic
                    #      + 8256 LUTRAM), 0 BRAM, because the write/read
                    #      processes had two textually-different
                    #      'bank_mem(addr)' accesses (one per w0/w1 or
                    #      r0/r1 branch) instead of one pre-selected
                    #      address/enable per port -- UG901's canonical
                    #      simple-dual-port template needs exactly one
                    #      textual access per port to be recognized, even
                    #      though only one branch ever executes per cycle.
                    #      Rewriting both processes to select the
                    #      address/data/enable into a variable first, then
                    #      doing the single 'bank_mem(addr)' access
                    #      unconditionally on that variable, fixed it.
                    # Final measured 2026-09 (Vivado 2026.1) with
                    # `analyze_synthesis_timing=True`: 783 LUTs, 534 FFs,
                    # 4 RAMB36 (2 per bank -- each bank is 1024 x 64 =
                    # 65536 bits, which needs 2 cascaded RAMB36 (36864 bits
                    # each) since a single one is too small; NOT 1 per
                    # bank as this build was first pinned before the real
                    # measurement) + 0 RAMB18, 0 DSP, 0 LUTRAM.
                    # 150 MHz timing estimate: PASS, 221.63 MHz -- an
                    # upper bound like every other leaf here (its
                    # geometry/arbiter cone starts at registers, not input
                    # ports, so this one is closer to trustworthy than
                    # most, but conv_core-style composition is still the
                    # real test). Comfortably clear, as expected: the only
                    # combinational depth is the per-bank round-robin
                    # arbiter plus the landing-register/output-mux read
                    # path added since this entity was last touched, none
                    # of the deep arithmetic cones pe_array/bias_requant/
                    # window_gen had. 0 DSP matches the header comment (no
                    # multiply anywhere in this design). The new
                    # per-channel landing/skid register and its 2:1 output
                    # mux are FFs/LUTs, not a new resource class, and
                    # there is no prior baseline to compare against (this
                    # is the first measurement) -- but 783 LUTs/534 FFs is
                    # a small fraction of an XC7A200T (134600 LUTs,
                    # 365 RAMB36, 740 DSP), so the mux does not cost
                    # anything meaningful.
                    checkers=[
                        TotalLuts(LessThan(1000)),
                        Ffs(LessThan(700)),
                        Ramb36(EqualTo(4)),
                        Ramb18(LessThan(1)),
                        DspBlocks(LessThan(1)),
                    ],
                    analyze_synthesis_timing=True,
                ),
                # The 60 fps scaling point (flow_status.md S1-S7:
                # `PE_ROWS_SCALED`, THE single scaling knob doubled, every
                # other generic identical to the shipped default above).
                # Proven here on the authoritative backend only -- this
                # project synthesizes with Vivado, there is deliberately no
                # Yosys twin of these two -- so the "does 16 rows still fit
                # the XC7A200T" question has a real answer without changing
                # the default. The scaled point exists to be measured, so
                # both entries below are sized against a real run at
                # g_pe_rows=16, not extrapolated from the 8-row numbers:
                # pe_array's MAC array and bias_requant's requant lanes both
                # scale linearly with rows, weight_buffer's lane count and
                # bias width too, window_gen not at all.
                vivado_build(
                    f"cnn_accel_pe_array_pe_rows_{_PE_ROWS_SCALED}",
                    "cnn_accel_pe_array",
                    {
                        "g_pe_rows": _PE_ROWS_SCALED,
                        "g_pe_cols": _PE_COLS,
                        "g_accum_width": _ACCUM_WIDTH,
                        "g_max_kernel_size": _MAX_KERNEL_SIZE,
                        "g_tile_channels": _TILE_CHANNELS,
                        "g_weight_buffer_depth": _WEIGHT_BUFFER_DEPTH,
                    },
                    # Measured 2026-09 (Vivado 2026.1) with
                    # `analyze_synthesis_timing=True` (real 500 MHz
                    # `create_clock` active during synthesis -- see the S7
                    # module-level comment above): 13696 LUTs, 1657 FFs,
                    # 0 BRAM, 0 DSP (were 13136/1642 unconstrained -- both
                    # still comfortably inside the `LessThan` margins below,
                    # no checker change needed here) -- LUTs 1.94x the 8-row
                    # 7068, i.e. the MAC array scales linearly with rows as
                    # it must (128 int8 lanes instead of 64), FFs less than
                    # 2x because the control/address side does not scale at
                    # all. 0 DSP for the same reason as the 8-row entry
                    # above (the `is_valid`-gated accumulate keeps the whole
                    # MAC array in LUT fabric), and the accumulators still
                    # live in flip-flops (0 BRAM), both unchanged by the
                    # knob.
                    # 150 MHz timing estimate: **FAIL**, 56.44 MHz --
                    # essentially identical to the 8-row 56.52 MHz, meaning
                    # the critical path does not scale with `g_pe_rows` (it
                    # is inside one PE lane's MAC chain, not across rows).
                    # See flow_status.md S7.
                    #
                    # ================= DSP48 int8 PACKING (2026-09) ======
                    # Re-measured after `cnn_accel_pe_array.vhd` started
                    # packing TWO int8 MACs into one DSP48E1 -- see that
                    # file's "DSP48 int8 PACKING" block for the design and
                    # the (exhaustively checked) overflow bound. This is a
                    # pure MAPPING change: no pipeline stage was added or
                    # removed, `c_mac_latency` is unchanged, and every
                    # simulation expected value is bit-identical.
                    #
                    # **3359 LUTs, 3638 FFs, 0 BRAM, 64 DSP, 205.97 MHz**
                    # (was 12345 / 5846 / 0 / 0 / 232.67 MHz -- note the
                    # FF checker below was already stale and FAILING at
                    # 5846 against its 1750 limit before this change; it is
                    # re-pinned here against a real measurement rather than
                    # left broken).
                    #
                    # DSP is exactly `g_pe_rows/2 * g_pe_cols = 64`, i.e.
                    # 2 MACs per DSP at the scaled point too, and LUTs
                    # scale linearly with rows (3359 ~= 2 x 1852) as they
                    # must. Fmax is identical to the 8-row build, so the
                    # critical path still does not scale with `g_pe_rows`.
                    # Headroom check the scaled point exists to answer:
                    # 64 DSP of the XC7A200T's 740 -- the 128-MAC array now
                    # costs 8.6% of the DSPs and 2.5% of the LUTs, where
                    # before it cost 0% and 9.2%.
                    #
                    # ===== CROSS-POSITION PIPELINING (2026-09) =======
                    # Re-measured after the same change as the 8-row
                    # entry above: **3374 LUTs, 3662 FFs, 0 BRAM,
                    # 64 DSP, 205.97 MHz** (was 3359 / 3638 / 0 / 64 /
                    # 205.97). DSP is exactly `g_pe_rows/2 * g_pe_cols`
                    # still, i.e. the packing survives at the scaled
                    # point too; LUT +15 / FF +24 is the same fixed
                    # control-side cost as at 8 rows (the flag pipelines
                    # are per-stage, not per-row, so it does not double),
                    # and Fmax is bit-identical. Checkers unchanged:
                    # 3374 < 3700, 3662 < 4000.
                    checkers=[
                        TotalLuts(LessThan(3700)),
                        Ffs(LessThan(4000)),
                        Ramb36(LessThan(1)),
                        Ramb18(LessThan(1)),
                        DspBlocks(EqualTo(64)),
                    ],
                    analyze_synthesis_timing=True,
                ),
                vivado_build(
                    f"cnn_accel_conv_core_pe_rows_{_PE_ROWS_SCALED}",
                    "cnn_accel_conv_core",
                    {
                        "g_max_kernel_size": _MAX_KERNEL_SIZE,
                        "g_pe_rows": _PE_ROWS_SCALED,
                        "g_pe_cols": _PE_COLS,
                        "g_accum_width": _ACCUM_WIDTH,
                        "g_tile_channels": _TILE_CHANNELS,
                        "g_max_row_tile_words": _MAX_ROW_TILE_WORDS,
                        "g_weight_buffer_depth": _WEIGHT_BUFFER_DEPTH,
                        "g_bias_buffer_depth": _BIAS_BUFFER_DEPTH,
                    },
                    # Measured 2026-09 (Vivado 2026.1) with
                    # `analyze_synthesis_timing=True` (real 500 MHz
                    # `create_clock` active during synthesis -- see the S7
                    # module-level comment above): 29027 LUTs, 4725 FFs,
                    # 17 RAMB36 + 2 RAMB18 (19 total), 68 DSP, 0 LUTRAM --
                    # the 60 fps system-footprint number, the whole reason
                    # this entry exists. FF grew and RAMB36/18 shrank from
                    # the unconstrained baseline (were 4219/24/3) for the
                    # same area/timing-tradeoff reason as every other entry
                    # in this set; LUTs stayed close (29253 -> 29027). Still
                    # well inside the XC7A200T (134600 LUTs, 365 RAMB36, 740
                    # DSP), so the scaled point fits with room for the DMAs,
                    # CSR and sequencer still to come. Leaf additivity holds
                    # exactly for DSP: 64 (bias_requant, 4 per requant lane
                    # x 16 lanes, was 32 at 8 rows) + 4 (window_gen, address
                    # arithmetic, row-independent) + 0 + 0 = 68.
                    #
                    # 150 MHz timing estimate: **FAIL**, 45.38 MHz --
                    # essentially identical to the 8-row conv_core's
                    # 46.77 MHz, confirming (together with the pe_array
                    # pair above) that the critical path lives inside one
                    # PE lane's MAC/accumulate chain and does not scale with
                    # `g_pe_rows`. See flow_status.md S7 for the full
                    # six-entity summary and the timing-fix delegation.
                    #
                    # Re-measured 2026-09 (YOLOv8n sizing pass): **25413
                    # LUTs, 15499 FFs, 43 RAMB36 + 2 RAMB18, 68 DSP,
                    # 170.33 MHz** -- the 16-row point now also clears
                    # 150 MHz, matching the 8-row build exactly (the
                    # critical path is inside one PE lane and does not
                    # scale with `g_pe_rows`, as the pair of measurements
                    # above already showed for the failing case).
                    #
                    # Two of these checkers were badly stale before this
                    # pass and are re-pinned here rather than left as
                    # known-failing: FF was still at the pre-S7 4900 while
                    # the entity really has ~15.5k (the 8-row build has
                    # 8727, and FFs scale with `g_pe_rows`), and RAMB36 was
                    # 17 against a real 43. Of that 43, the increase this
                    # change is responsible for is window_gen's 3 -> 12.
                    #
                    # ================= DSP48 int8 PACKING (2026-09) ======
                    # Re-measured after the pe_array DSP48 int8 packing
                    # (see the 8-row conv_core entry above for the full
                    # rationale): **16989 LUTs, 13348 FFs, 42 RAMB36 +
                    # 2 RAMB18, 132 DSP, 178.57 MHz** (was 25705 / 15507 /
                    # 43 + 2 / 68 / 170.33 MHz).
                    #
                    #  * LUT 25705 -> 16989, **-33.9%**: the 60 fps
                    #    system-footprint number, and the answer to "does
                    #    16 rows fit the XC7A200T" is now a much easier
                    #    yes -- 12.6% of its LUTs and 17.8% of its DSPs.
                    #  * DSP 68 -> 132, exact leaf additivity again:
                    #    64 (pe_array, 128 MACs at 2 per DSP) + 64
                    #    (bias_requant, 4 per requant lane x 16 lanes) + 4
                    #    (window_gen) = 132.
                    #  * RAMB36 43 -> 42 with LUTRAMs 0 -> 28: Vivado moved
                    #    one small weight_buffer prefetch FIFO out of block
                    #    RAM into LUTRAM now that there are LUTs going
                    #    spare next to it. A mapping side effect of the LUT
                    #    pressure dropping, not a storage change -- pinned
                    #    at the measured 42 rather than assumed back to 43.
                    #  * Fmax 170.33 -> 178.57 MHz, matching the 8-row
                    #    build exactly, so the critical path still does not
                    #    scale with `g_pe_rows`.
                    #
                    # ===== CROSS-POSITION PIPELINING (2026-09) =======
                    # Re-measured after the pe_array throughput rework
                    # (see the 8-row conv_core entry above for what it
                    # buys in cycles/position): **17033 LUTs, 13379 FFs,
                    # 42 RAMB36 + 2 RAMB18, 132 DSP, 178.57 MHz** (was
                    # 16989 / 13348 / 42 + 2 / 132 / 178.57). +44 LUTs,
                    # +31 FFs, everything else bit-identical including
                    # Fmax, so the 60 fps scaled point still clears
                    # 150 MHz with the same 19% margin. Checkers
                    # unchanged: 17033 < 18700, 13379 < 14700.
                    #
                    # ===== DOUBLE-BUFFERED TAP ASSEMBLY (2026-09) =====
                    # Re-measured after the same change: **18863 LUTs,
                    # 14620 FFs, 42 RAMB36 + 2 RAMB18, 132 DSP,
                    # 178.57 MHz** (was 17033 / 13379 / 42 + 2 / 132 /
                    # 178.57). +1830 LUTs, +1241 FFs -- the same absolute
                    # delta as the 8-row build (+1842 / +1253), which is
                    # the expected result and worth stating: the tap
                    # assembly is sized by `g_max_kernel_size` and
                    # `g_tile_channels`, not by `g_pe_rows`, so this cost
                    # does NOT scale with the array. Block RAM, DSP and
                    # Fmax again bit-identical; the 60 fps scaled point
                    # still clears 150 MHz with a 19% margin. Re-pinned at
                    # the same ~1.10x headroom as before.
                    # ===== RE-PINNED 2026-09-09 (timing pass) ==========
                    # Measured: **18159 LUTs, 15193 FFs, 43 RAMB36 +
                    # 2 RAMB18, 132 DSP**. As with the 8-row build above,
                    # only the RAMB36 gate moves (42 -> 43) and it was
                    # already failing on 'main' before this pass -- the
                    # 42 was pinned from a measurement taken when the
                    # weight buffer's scale region (H2) did not exist.
                    # DSP is bit-identical at 132, so the int8 packing is
                    # intact at the scaled point too.
                    # ===== SCALE REGION MOVED TO LUTRAM (round-3 timing
                    # pass) ===========================================
                    # Same single cause as the 8-row build above -- read
                    # that comment for the evidence, the bisect and why
                    # this is the entity's documented design intent
                    # ('g_bias_buffer_depth' exists so the bias/scale
                    # region "can fall out of block RAM into LUTRAM/
                    # registers") rather than drift.
                    #
                    # At 16 rows 'scale_mem' is 8 rows x (40 x 16) = 640
                    # bits wide, which is the 9 RAMB36 that left: 43 -> 34.
                    # RAMB18 stays 2 here (the 16-row geometry packs the
                    # remainder differently from the 8-row one, where the
                    # single RAMB18 went too), and DSP is bit-identical at
                    # 132, so the int8 packing is intact at the scaled
                    # point as well.
                    checkers=[
                        TotalLuts(LessThan(20700)),
                        Ffs(LessThan(16100)),
                        Ramb36(EqualTo(34)),
                        Ramb18(EqualTo(2)),
                        LutRams(EqualTo(428)),
                        DspBlocks(EqualTo(132)),
                    ],
                    analyze_synthesis_timing=True,
                ),
            ]

            # ================================================================
            # TOP-LEVEL BUILDS (synthesis + place-and-route + bitstream).
            #
            # Everything above this line -- Yosys and Vivado alike -- is an
            # out-of-context *netlist* build of one entity. Those give real
            # resource counts but only an unrouted, register-to-register
            # timing *estimate*, and only of the entity they build; nothing
            # above measures 'cnn_accel_top', and until this block existed
            # 'build_fpga.py --list-only' (full builds) listed nothing at
            # all. The blocks that the system total was previously only
            # estimated for -- 'cmd_proc', 'cmd_fetch', 'csr', the three
            # 'axi_read_dma', 'ofmap_dma', 'axi_mux', 'elementwise',
            # 'pool_requant' -- are measured for the first time here.
            #
            # Top level is 'cnn_accel_top_build', NOT 'cnn_accel_top': see
            # that entity's header for why a full build cannot have the
            # accelerator itself on top (~840 port bits versus this part's
            # 285 I/O, and tsfpga's '-no_iobuf' out-of-context mode is
            # synthesis-only by construction). The wrapper terminates the
            # whole bus boundary in flip-flops inside the fabric and exposes
            # five pins, so every accelerator path is a real
            # register-to-register path and the hierarchical utilization
            # report still attributes the accelerator separately.
            #
            # Constraints: 'tcl/cnn_accel_top_build_pinning.xdc', attached
            # project-wide the same way tsfpga's own 'artyz7' example
            # attaches 'tcl/artyz7_pinning.tcl'. It pins the five ports and
            # creates the 150 MHz target clock. NOT a scoped constraint:
            # 'create_clock' on a top-level port is a project-level
            # constraint, and scoping it to a '-ref' entity would give it no
            # port to attach to.
            from tsfpga.constraint import Constraint
            from tsfpga.vivado.project import VivadoProject

            # A superset of the netlist builds' 'modules': the top pulls in
            # 'dma_axi_write_simple' (via 'cnn_accel_ofmap_dma', which has no
            # netlist build of its own), which in turn pulls in 'resync' and
            # 'ring_buffer'. Rather than tracking that closure by hand, take
            # all of hdl-modules with the same two exclusions 'run.py' uses:
            # 'hard_fifo' is Xilinx-unisim-only, 'bfm' is simulation-only.
            top_modules = get_modules(
                modules_folder=self.path.parent, names_include={self.name}
            ) + get_modules(
                modules_folder=self.path.parent.parent / "hdl-modules" / "modules",
                names_avoid={"hard_fifo", "bfm"},
            )

            pinning = Constraint(self.path / "tcl" / "cnn_accel_top_build_pinning.xdc")

            def top_build(name: str, generics: dict) -> VivadoProject:
                return VivadoProject(
                    name=name,
                    modules=top_modules,
                    part=_VIVADO_PART,
                    top="cnn_accel_top_build",
                    generics=generics,
                    constraints=[pinning],
                    vivado_path=vivado_path,
                    defined_at=Path(__file__),
                )

            # ONE design point only, deliberately. There is no
            # 'cnn_accel_top_build_pe_rows_16' to match the '_pe_rows_16'
            # netlist builds above, because at the top level
            # '_PE_ROWS_SCALED' does not elaborate:
            #
            #   ERROR: [Synth 8-63] RTL assertion: "cnn_accel_top: g_pe_rows
            #   (16) must equal the generated activation-plane channel count
            #   (8): one OT pass writes exactly one output activation plane"
            #   (cnn_accel_top.vhd, the 'g_pe_rows' assert)
            #
            # That assertion is a real contract of the rev-2 memory layout,
            # not a build-script detail: one OT pass writes exactly one
            # activation plane, and the plane's channel count 'T' is baked
            # into the ISA/DDR layout via 'cnn_accel_constants.py'. So the
            # 16-row point that 'cnn_accel_pe_array'/'cnn_accel_conv_core'
            # are measured at is a *datapath* scaling point that the
            # integrated accelerator cannot currently be built at; taking it
            # to the top would mean scaling 'ACTIVATION_PLANE_CHANNELS' (and
            # therefore 'MAX_AXI_DATA_WIDTH', the layout and the compiler)
            # with it. Registering a project that can only ever fail
            # elaboration would be worse than not registering one, and
            # relaxing the assert to make it build is exactly the kind of
            # "weaken the checker until it passes" this file warns against
            # everywhere else. Add the variant back the day the top supports
            # it.
            projects += [top_build("cnn_accel_top_build", {"g_pe_rows": _PE_ROWS})]

        return projects

    def setup_vunit(self, vunit_proj: VUnit, **kwargs) -> None:
        library = vunit_proj.library(self.library_name)

        self._setup_cnn_accel_bias_requant(library)
        self._setup_cnn_accel_window_gen(library)
        self._setup_cnn_accel_pool(library)
        self._setup_cnn_accel_pe_array(library)
        self._setup_cnn_accel_pe_array_from_vectors(library)
        self._setup_cnn_accel_conv_core(library)
        self._setup_cnn_accel_tensor_mem(library)
        self._setup_cnn_accel_top(library)
        self._setup_cnn_accel_elementwise_pyffi_pilot(library)
        library.test_bench("tb_python_ffi_throughput_pilot")

    def _setup_cnn_accel_elementwise_pyffi_pilot(self, library) -> None:
        # PILOT ONLY (branch feat/vunit-python-ffi-pilot): needs the
        # venv-pyffi install of ru551n/vunit@feature/python-ffi and
        # run.py's `add_vhdl_builtins(python=True)` -- not part of the
        # project's normal shared venv/requirements.txt yet. See
        # tb_cnn_accel_elementwise_pyffi_pilot.vhd's header.
        library.test_bench("tb_cnn_accel_elementwise_pyffi_pilot")

    def _setup_cnn_accel_window_gen(self, library) -> None:
        """Run the whole window-generator suite at BOTH shipped tap-assembly
        buffer depths.

        `g_assembly_buffers` is a pure scheduling knob -- how many windows
        may be in flight between the row banks and `m_window` at once --
        so every golden-window expectation in `tb_cnn_accel_window_gen` is
        identical for both, and running the suite twice is a real check
        rather than a duplicated one. The two values are the two the
        design actually instantiates: 1 for `cnn_accel_top`'s POOL
        instance (K=5, where a second 200-byte tap register bank would be
        paid for nothing) and `_ASSEMBLY_BUFFERS` for the CONV instance
        inside `cnn_accel_conv_core`, where a 1x1 convolution is
        window-generator-bound and every cycle counts.
        """
        tb = library.test_bench("tb_cnn_accel_window_gen")
        for buffers in (1, _ASSEMBLY_BUFFERS):
            for test in tb.get_tests():
                self.add_vunit_config(
                    test=test,
                    name=f"g_assembly_buffers_{buffers}",
                    generics={"g_assembly_buffers": buffers},
                )

    def _setup_cnn_accel_top(self, library) -> None:
        """The rev-2 top-level integration testbench (arch doc section 11).

        `tb_cnn_accel_top` is the project's ONE top-level testbench and it
        is completely generic: it loads a DDR image from CSV, starts the
        DUT through the CSR, waits for DONE/ERROR, then exports the CSR
        counters plus a byte region of the DDR model back to CSV. Every
        decision about what to run, and all numerical verification, lives
        in `accel_v2/cases.py` + `accel_v2/tbcase.py`, so one VUnit config
        per case is the whole registration -- adding a test never touches
        VHDL.

        `pre_config` writes that case's `mem_image.csv` into VUnit's own
        per-config `output_path`; `post_check` reads `result.csv` (the
        bytes the DUT itself wrote over AXI) and `counters.csv` back out
        of it. Nothing is checked into the repository.
        """
        # Imported here rather than at module scope: `accel_v2` is only
        # needed to register these configs, and `module_cnn_accel.py` is
        # also loaded by `build_fpga.py` (the synthesis env), where
        # dragging in the whole model/planner/reference stack buys nothing.
        from accel_v2 import (  # noqa: PLC0415
            cases,
            cases_concat_split,
            cases_conv_pad,
            cases_error,
            cases_pool_pad,
            cases_tiling,
            cases_yolo,
        )

        tb = library.test_bench("tb_cnn_accel_top")

        # Seven catalogues, one registration loop. `cases_pool_pad.py`
        # holds the ISA v2.1 pooling cases (padding, the zero-point pad
        # value, the 5x5 SPPF kernel); `cases_conv_pad.py` holds the
        # convolution half of that same `pad_value` field (a padded conv
        # fills with the tensor's zero-point, not 0 -- see that file's
        # docstring for why the data has to be clamped for the difference
        # to be observable at all); `cases_concat_split.py` holds the
        # channel CONCAT/SPLIT cases (which add no opcode at all -- they
        # are buffer aliasing, see that file's docstring); `cases_yolo.py`
        # tests by *topology* rather than by feature -- Bottleneck, C2f,
        # SPPF, backbone stage, FPN/PAN merge, the three-scale head
        # boundary and a small end-to-end YOLOv8n-shaped network;
        # `cases_tiling.py` holds the spatially tiled cases -- the only
        # ones whose program the tiler produced, and therefore the only
        # ones with a pinned tensor, a row copy, a plane-confined buffer
        # or a `space_wgt = LOCAL_TENSOR` convolution in them (see its
        # own docstring, and `tests/test_tiling_guard.py` for why the
        # other six must stay free of all of that);
        # `cases_error.py` holds the ISA v2.1 error-model coverage (each
        # `c_err_*` code that can be provoked deterministically, by
        # mutating one field of an otherwise-well-formed program's first
        # descriptor -- see that file's docstring for which codes it
        # deliberately does not attempt and why). All follow exactly the
        # same contract as `cases.py` and are separate files only so the
        # seven can be edited independently. Case names are unique across
        # all of them.
        for case in (
            cases.all_cases()
            + cases_pool_pad.all_cases()
            + cases_conv_pad.all_cases()
            + cases_concat_split.all_cases()
            + cases_yolo.all_cases()
            + cases_error.all_cases()
            + cases_tiling.all_cases()
        ):
            # VUnit's own `add_config` rather than tsfpga's
            # `add_vunit_config` helper: the helper always appends every
            # generic to the config name, and these configs carry ten of
            # them, which would bury the one identifier that matters (the
            # case name) in a 200-character test name. The case name is
            # already the reproducible handle -- `cases.py` maps it to its
            # seed and geometry -- so the generics add nothing here.
            #
            # `case.pre_config`/`case.post_check` are bound methods of that
            # one case object, so each config carries its own state; a
            # closure over the loop variable would instead run every hook
            # against the last case built.
            tb.add_config(
                name=case.name,
                generics=case.generics(),
                pre_config=case.pre_config,
                post_check=case.post_check,
            )

        # PILOT (branch feat/vunit-python-ffi-pilot): a second config for
        # ONE existing case, 'single_conv' (the catalogue's own reference
        # point for "does the minimal end-to-end path work at all" -- see
        # that case's docstring), checked LIVE via python_call instead of
        # counters.csv/result.csv -- see top_level_bridge.py and
        # TbCase.check_live. `pre_config` is unchanged (mem_image.csv is
        # still how the program gets INTO the simulation; only the
        # OUTPUT side moves to a python_call). `post_check` is
        # deliberately omitted: the live path already asserted everything
        # from inside the simulation via `check_true`, so there is
        # nothing left to check afterward -- an unset `post_check` is
        # VUnit's own "nothing to run" default, not a gap.
        single_conv_case = next(c for c in cases.all_cases() if c.name == "single_conv")
        inputs_base, inputs_bytes = single_conv_case.input_region()
        tb.add_config(
            name="single_conv_live_check_pilot",
            generics={
                **single_conv_case.generics(),
                "g_check_live": True,
                "g_case_name": "single_conv",
                "g_inputs_base": inputs_base,
                "g_inputs_bytes": inputs_bytes,
            },
            # 'live_pre_config' (not 'pre_config'): the compiler's own
            # output -- descriptors, weight/bias/scale/LUT tables -- still
            # goes into mem_image.csv exactly as before; only the graph's
            # input tensors are left out of that file and seeded live
            # instead (see cnn_accel_python_ffi_pkg.vhd's
            # 'ffi_seed_bytes' and top_level_bridge.py's 'input_bytes').
            pre_config=single_conv_case.live_pre_config,
        )

    def _setup_cnn_accel_bias_requant(self, library) -> None:
        tb = library.test_bench("tb_cnn_accel_bias_requant")

        for test in tb.get_tests():
            # Zero stall on both links only for the dedicated
            # full-throughput test (its check_relation timing check requires
            # back-to-back beats); randomized independent per-link
            # backpressure otherwise.
            stall = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={
                    "stall_probability_percent_in": stall,
                    "stall_probability_percent_out": stall,
                },
            )

            # The M6 record retrofit (accum_m2s_t/s2m_t replacing the fixed
            # 128-bit axi_stream_m2s_t on 's_accum') removed the old
            # g_accum_width*g_pe_rows<=128 lane ceiling. Add explicit
            # coverage at g_pe_rows=8 -- the width the rest of the design
            # actually uses -- as an extra config alongside (not instead
            # of) the directed g_pe_rows=4 default above, for the two
            # tests whose behavior is genuinely lane-count-sensitive
            # (per-beat parallel work, and thus throughput/backpressure
            # timing; the directed golden-model tests exercise per-lane
            # datapath correctness, which g_pe_rows=4 already covers fully
            # since lanes are independent).
            if "full_throughput" in test.name or "backpressure" in test.name:
                self.add_vunit_config(
                    test=test,
                    generics={
                        "stall_probability_percent_in": stall,
                        "stall_probability_percent_out": stall,
                        "g_pe_rows": 8,
                    },
                )

    def _setup_cnn_accel_pe_array_from_vectors(self, library) -> None:
        # Cross-language (Python packer -> real RTL) bit-exactness guard for
        # the D10 weight-lane-order defect, see
        # tb_cnn_accel_pe_array_from_vectors.vhd's own header comment.
        # The vectors are generated into VUnit's own per-test 'output_path'
        # by the pre_config hook right before simulation (the testbench
        # reads '<output_path>/pe_array_xlang_check'); nothing is checked
        # in and nothing is read from the repository.
        tb = library.test_bench("tb_cnn_accel_pe_array_from_vectors")

        def pre_config(output_path: str) -> bool:
            generate_vectors.generate_pe_array_xlang_case(
                generate_vectors.hw_packing(Path(output_path))
            )
            return True

        for test in tb.get_tests():
            self.add_vunit_config(test=test, pre_config=pre_config)

    def _setup_cnn_accel_conv_core(self, library) -> None:
        # Cross-language (Python golden model -> composed RTL) bit-exactness
        # guard for the full window_gen -> pe_array -> bias_requant
        # composition, see tb_cnn_accel_conv_core.vhd's own header comment.
        # Vectors are generated into VUnit's own per-config 'output_path'
        # by each config's pre_config hook right before simulation, at that
        # config's `g_pe_rows` packing point; nothing is checked in and
        # nothing is read from the repository.
        tb = library.test_bench("tb_cnn_accel_conv_core")

        def make_pre_config(pe_rows: int):
            def pre_config(output_path: str) -> bool:
                generate_vectors.generate_conv_core_cases(
                    generate_vectors.hw_packing(Path(output_path), pe_rows=pe_rows)
                )
                return True

            return pre_config

        for test in tb.get_tests():
            if test.name == "test_bitexact_compiler_cases":
                # M10 (doc/tosa_compiler_plan.md ~line 613): a single
                # config, not one per `PE_ROWS_LEGAL` -- the compiler
                # always packs weights at its discovered target's own
                # `internal_tiling` (== `PE_ROWS`/`TILE_CHANNELS`, see
                # `cnnc.target.discover`), it is not parameterizable by a
                # `pe_rows` argument the way `generate_vectors.hw_packing`
                # is, so there is no `g_pe_rows_16` variant of this test.
                self.add_vunit_config(
                    test=test,
                    generics={
                        "stall_probability_percent_in": 20,
                        "stall_probability_percent_out": 20,
                        "g_pe_rows": cnn_accel_constants.PE_ROWS,
                    },
                    pre_config=_compiler_vectors_pre_config,
                )
                continue

            # Zero stall on both links only for the dedicated
            # full-throughput test (proves sustained back-to-back
            # operation); randomized independent per-link backpressure
            # otherwise. Matches every other cnn_accel testbench's
            # identical `_setup_*` precedent.
            stall = 0 if "full_throughput" in test.name else 20

            # One config per legal `g_pe_rows` (S1: the single scaling
            # knob). The default (`PE_ROWS`, 8) and the CI-proven scaled
            # point (`PE_ROWS_SCALED`, 16) each get their own generated
            # root, packed at that many lanes per weight row; the scaled
            # one carries the extra 16-output-channel case that only
            # exists there. The testbench cross-checks each case's
            # desc.txt `pe_rows` against its generic, so a root/generic
            # mix-up here fails by name rather than as a data mismatch.
            # The default is deliberately not renamed: its test names are
            # unchanged, the scaled config is the one that grows a
            # `.g_pe_rows_16` suffix.
            for pe_rows in cnn_accel_constants.PE_ROWS_LEGAL:
                self.add_vunit_config(
                    test=test,
                    name=None if pe_rows == cnn_accel_constants.PE_ROWS else f"g_pe_rows_{pe_rows}",
                    generics={
                        "stall_probability_percent_in": stall,
                        "stall_probability_percent_out": stall,
                        "g_pe_rows": pe_rows,
                    },
                    pre_config=make_pre_config(pe_rows),
                )

    def _setup_cnn_accel_tensor_mem(self, library) -> None:
        # Plain default-generic testbench (tb_cnn_accel_tensor_mem.vhd's own
        # header comment has the verification plan): every test case picks
        # its own read-consumer mode/stall percentage internally via live
        # signals, so there is nothing to sweep here -- matches
        # tb_cnn_accel_weight_buffer.vhd's identical no-generics-to-vary
        # precedent for this project's other non-PE_ROWS-parametric leaf
        # modules.
        tb = library.test_bench("tb_cnn_accel_tensor_mem")

        # 'test_zero_length_request_asserts' deliberately triggers the
        # entity's zero-length-request assertions (severity 'error',
        # matching cnn_accel_tensor_mem.vhd's existing bank-crossing/
        # out-of-range-bank style exactly). VUnit's GHDL backend defaults
        # the 'vhdl_assert_stop_level' sim option to 'error'
        # (vunit/sim_if/ghdl.py, via '--assert-level'), which aborts the
        # whole simulation the instant the first assertion fires --
        # contradicting this entity's own header comment, which assumes
        # severity 'error' lets the run continue (true of the VHDL LRM's
        # default, but not of VUnit's stricter default). Lowering the
        # stop level to 'failure' for this ONE test case -- not project-
        # wide, so every other test here still catches a real severity-
        # 'failure' bug (e.g. a misaligned request) -- is VUnit's own
        # supported knob for exactly this, and is what actually makes
        # severity 'error' behave the way the RTL comments promise,
        # rather than weakening the assertion itself or muting the
        # message.
        tb.test("test_zero_length_request_asserts").add_config(
            name="default", sim_options={"vhdl_assert_stop_level": "failure"}
        )

        # 'test_bank_crossing_request_is_detected' deliberately issues a
        # request that runs past the end of the bank its address decodes
        # to. That assertion is severity 'failure' in every real
        # instantiation (cnn_accel_tensor_mem.vhd's header explains why it
        # was raised from 'error': no bank-aware caller can produce one,
        # and the hardware's response -- clamping -- silently truncates
        # the transfer, so the run must stop). The test's whole job,
        # though, is to check the two things the assertion itself cannot
        # state: that the clamp really is to the addressed bank, and that
        # the neighbouring bank is untouched. So this ONE config lowers
        # the DUT's assertion to 'error' via the testbench's
        # 'g_illegal_request_severity_error' generic and lowers VUnit's
        # stop level to 'failure' to match, exactly as the zero-length
        # test above does -- the shipped severity is unchanged, and every
        # other config (including tb_cnn_accel_top's) still dies
        # immediately on a bank crossing.
        tb.test("test_bank_crossing_request_is_detected").add_config(
            name="default",
            generics={"g_illegal_request_severity_error": True},
            sim_options={"vhdl_assert_stop_level": "failure"},
        )

    def _setup_cnn_accel_pool(self, library) -> None:
        tb = library.test_bench("tb_cnn_accel_pool")

        for test in tb.get_tests():
            # Zero stall on all three links only for the dedicated
            # full-throughput test; randomized independent per-link
            # backpressure otherwise.
            stall = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={
                    "stall_probability_percent_in": stall,
                    "stall_probability_percent_max": stall,
                    "stall_probability_percent_avgsum": stall,
                },
            )

    def _setup_cnn_accel_pe_array(self, library) -> None:
        tb = library.test_bench("tb_cnn_accel_pe_array")

        for test in tb.get_tests():
            # Zero stall on both links only for the dedicated
            # full-throughput test (its check_relation timing check requires
            # back-to-back beats); randomized independent per-link
            # backpressure otherwise.
            stall = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={
                    "stall_probability_percent_in": stall,
                    "stall_probability_percent_out": stall,
                },
            )
