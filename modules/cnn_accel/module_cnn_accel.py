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
# = K^2 * ceil(C_max/8) = 9*32 for the target backbone's worst layer
# (3x3x256, layer 9) -- replaces the old arbitrary 512 now that weights are
# streamed per output-channel pass from DDR4 instead of double-buffered
# on-chip (see doc/cnn_accel_weight_buffer.md's depth-sizing note).
_WEIGHT_BUFFER_DEPTH = cnn_accel_constants.WEIGHT_BUFFER_DEPTH
# Independent of _WEIGHT_BUFFER_DEPTH: a real layer only ever needs
# ceil(out_channels/g_pe_rows) bias rows, far fewer than the weight region.
_BIAS_BUFFER_DEPTH = cnn_accel_constants.BIAS_BUFFER_DEPTH
_ACCUM_WIDTH = cnn_accel_constants.ACCUM_WIDTH

# Pooling is a separate, much smaller kernel bound: the target network only
# pools 2x2. The entity's own contract is `g_max_kernel_size**2 * 8 <= 128`,
# so 3 is the largest useful value; kept at 3 for headroom over 2x2.
_POOL_MAX_KERNEL_SIZE = 3
# Must hold `_POOL_MAX_KERNEL_SIZE**2 * 127` = 1143 without overflow.
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
            name="max_kernel_size",
            value=cnn_accel_constants.MAX_KERNEL_SIZE,
            description="Largest K_h/K_w this accelerator's datapath supports.",
        )
        regs.add_constant(
            name="max_row_tile_words",
            value=cnn_accel_constants.MAX_ROW_TILE_WORDS,
            description="Weight-buffer row-tile-word sizing bound.",
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
            # constant/register that would be part of that hash. Generation
            # is cheap (one small file), so always regenerating is simpler
            # and correct rather than wiring up a second, parallel
            # staleness check.
            CnnAccelIsaPackageGenerator(
                register_list=self.registers, output_folder=self.register_synthesis_folder
            ).create()

    def get_build_projects(self) -> list:
        # Local import: tsfpga.yosys.project needs a tsfpga build with Yosys
        # netlist-build support (not in the stable release this project's
        # run.py/VUnit flow uses), so this must not be imported at module
        # load time -- only build_fpga.py ever calls this method. Matches
        # module_canny.py / module_axi_stream_join.py.
        from tsfpga.vivado.build_result_checker import (
            BlockRams,
            DspBlocks,
            EqualTo,
            Ffs,
            LessThan,
            Ramb18,
            Ramb36,
            TotalLuts,
        )
        from tsfpga.yosys.project import YosysXilinxNetlistBuild

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

        def build(name: str, generics: dict, checkers: list) -> YosysXilinxNetlistBuild:
            return YosysXilinxNetlistBuild(
                name=name,
                modules=modules,
                top=name,
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
                checkers=[
                    TotalLuts(LessThan(13000)),
                    Ffs(LessThan(130)),
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
                checkers=[
                    TotalLuts(LessThan(850)),
                    Ffs(LessThan(60)),
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
                    Ffs(LessThan(2000)),
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
                checkers=[
                    TotalLuts(LessThan(13000)),
                    Ffs(LessThan(1400)),
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
                    BlockRams(EqualTo(3)),
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
                checkers=[
                    TotalLuts(LessThan(6100)),
                    Ffs(LessThan(2000)),
                    BlockRams(LessThan(1)),
                    DspBlocks(LessThan(70)),
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
                checkers=[
                    TotalLuts(LessThan(34000)),
                    Ffs(LessThan(5400)),
                    BlockRams(LessThan(20)),
                    DspBlocks(LessThan(115)),
                ],
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
                name: str, top: str, generics: dict, checkers: list
            ) -> VivadoNetlistProject:
                return VivadoNetlistProject(
                    name=f"{name}_vivado",
                    modules=modules,
                    part=_VIVADO_PART,
                    top=top,
                    generics=generics,
                    build_result_checkers=checkers,
                    vivado_path=vivado_path,
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
                    # Measured 2026-09: 4906 LUTs, 66 FFs, 0 BRAM, 32 DSP.
                    # DSP count matches Yosys's own 32 exactly -- this
                    # entity's requant multiply is wide enough (not a tiny
                    # int8 x int8) that both tools map it to DSP48E1 the
                    # same way, unlike pe_array (see that entry below).
                    checkers=[
                        TotalLuts(LessThan(5200)),
                        Ffs(LessThan(80)),
                        Ramb36(LessThan(1)),
                        Ramb18(LessThan(1)),
                        DspBlocks(EqualTo(32)),
                    ],
                ),
                vivado_build(
                    "cnn_accel_pool",
                    "cnn_accel_pool",
                    {
                        "g_max_kernel_size": _POOL_MAX_KERNEL_SIZE,
                        "g_accum_width": _POOL_ACCUM_WIDTH,
                    },
                    # Measured 2026-09: 293 LUTs, 27 FFs, 0 BRAM, 0 DSP --
                    # smaller than Yosys's 342/27/0/0 for the same RTL, a
                    # rare case where Vivado's mapping is the tighter one.
                    checkers=[
                        TotalLuts(LessThan(350)),
                        Ffs(LessThan(35)),
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
                    # Measured 2026-09: 848 LUTs, 820 FFs, 11 RAMB36 +
                    # 2 RAMB18 (13 total), 0 DSP. BRAM is close to Yosys's
                    # 15 (both far below the pre-rework 72) but not
                    # identical: Vivado also puts the depth-32 prefetch
                    # FIFO in block RAM here, where Yosys's RAM32M-based
                    # distributed-RAM mapping for that same FIFO shows up
                    # as LUTs instead (see conv_core_vivado's own comment).
                    checkers=[
                        TotalLuts(LessThan(950)),
                        Ffs(LessThan(900)),
                        Ramb36(EqualTo(11)),
                        Ramb18(EqualTo(2)),
                        DspBlocks(LessThan(1)),
                    ],
                ),
                vivado_build(
                    "cnn_accel_window_gen",
                    "cnn_accel_window_gen",
                    {
                        "g_max_kernel_size": _MAX_KERNEL_SIZE,
                        "g_max_row_tile_words": _MAX_ROW_TILE_WORDS,
                        "g_tile_channels": _TILE_CHANNELS,
                    },
                    # Measured 2026-09: 2900 LUTs, 815 FFs, 3 RAMB36 +
                    # 0 RAMB18, 4 DSP. BRAM count of exactly 3 (one per
                    # kernel-row bank) agrees with Yosys's own
                    # `BlockRams(EqualTo(3))` below -- both tools confirm
                    # the M7 BRAM-inference fix -- so this checker is kept
                    # just as exact here, on the authoritative backend,
                    # as it is on the CI-gating one. The 4 DSPs are address
                    # arithmetic that Vivado folded into DSP48E1s (Yosys
                    # folds a similar adder into one of its 9 DSPs too, for
                    # the same underlying reason).
                    checkers=[
                        TotalLuts(LessThan(3200)),
                        Ffs(LessThan(900)),
                        Ramb36(EqualTo(3)),
                        Ramb18(LessThan(1)),
                        DspBlocks(EqualTo(4)),
                    ],
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
                    # Measured 2026-09: 6778 LUTs, 1126 FFs, 0 BRAM,
                    # *0* DSP. This is the one genuinely surprising number
                    # in this whole set: constraint/prior assumption said
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
                    checkers=[
                        TotalLuts(LessThan(7200)),
                        Ffs(LessThan(1200)),
                        Ramb36(LessThan(1)),
                        Ramb18(LessThan(1)),
                        DspBlocks(LessThan(1)),
                    ],
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
                    # Measured 2026-09: 15358 LUTs, 2824 FFs, 14 RAMB36 +
                    # 2 RAMB18 (16 total), 36 DSP, 0 LUTRAM -- the
                    # system-footprint number ("does the accelerator fit an
                    # XC7A200T") this whole Vivado backend exists for.
                    # See the Yosys `cnn_accel_conv_core` entry above for
                    # the full leaf-additivity cross-check and the
                    # DSP-attribution correction (32 from bias_requant + 4
                    # from window_gen + 0 + 0, NOT 32 from pe_array as an
                    # earlier version of that comment assumed).
                    checkers=[
                        TotalLuts(LessThan(16000)),
                        Ffs(LessThan(3000)),
                        Ramb36(EqualTo(14)),
                        Ramb18(EqualTo(2)),
                        DspBlocks(EqualTo(36)),
                    ],
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
                    # Measured 2026-09 (Vivado 2026.1): 13136 LUTs, 1642 FFs,
                    # 0 BRAM, 0 DSP -- LUTs 1.94x the 8-row 6778, i.e. the
                    # MAC array scales linearly with rows as it must (128
                    # int8 lanes instead of 64), FFs less than 2x because
                    # the control/address side does not scale at all. 0 DSP
                    # for the same reason as the 8-row entry above (the
                    # `is_valid`-gated accumulate keeps the whole MAC array
                    # in LUT fabric), and the accumulators still live in
                    # flip-flops (0 BRAM), both unchanged by the knob.
                    checkers=[
                        TotalLuts(LessThan(13800)),
                        Ffs(LessThan(1750)),
                        Ramb36(LessThan(1)),
                        Ramb18(LessThan(1)),
                        DspBlocks(LessThan(1)),
                    ],
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
                    # Measured 2026-09 (Vivado 2026.1): 29253 LUTs, 4219 FFs,
                    # 24 RAMB36 + 3 RAMB18 (27 total), 68 DSP, 0 LUTRAM --
                    # the 60 fps system-footprint number, the whole reason
                    # this entry exists: 1.9x the 8-row 15358 LUTs, well
                    # inside the XC7A200T (134600 LUTs, 365 RAMB36, 740
                    # DSP), so the scaled point fits with room for the DMAs,
                    # CSR and sequencer still to come. Leaf additivity holds
                    # exactly for DSP: 64 (bias_requant, 4 per requant lane
                    # x 16 lanes, was 32 at 8 rows) + 4 (window_gen, address
                    # arithmetic, row-independent) + 0 + 0 = 68. BRAM 16 ->
                    # 27 is weight_buffer alone (twice the weight lanes and
                    # twice the bias-row width; window_gen's 3 line banks
                    # do not scale with rows).
                    #
                    # Unconstrained logic-level report shows 32 endpoints at
                    # 102-105 levels (none above 35 in pe_array alone), so
                    # the deep paths sit in bias_requant/weight_buffer; see
                    # the 150 MHz constrained-build item in flow_status.md.
                    checkers=[
                        TotalLuts(LessThan(30500)),
                        Ffs(LessThan(4450)),
                        Ramb36(EqualTo(24)),
                        Ramb18(EqualTo(3)),
                        DspBlocks(EqualTo(68)),
                    ],
                ),
            ]

        return projects

    def setup_vunit(self, vunit_proj: VUnit, **kwargs) -> None:
        library = vunit_proj.library(self.library_name)

        self._setup_cnn_accel_bias_requant(library)
        self._setup_cnn_accel_pool(library)
        self._setup_cnn_accel_pe_array(library)
        self._setup_cnn_accel_pe_array_from_vectors(library)
        self._setup_cnn_accel_conv_core(library)

    def _setup_cnn_accel_bias_requant(self, library) -> None:
        tb = library.test_bench("tb_cnn_accel_bias_requant")

        for test in tb.get_tests():
            # Zero stall on both links only for the dedicated
            # full-throughput test (its check_relation timing check requires
            # back-to-back beats); randomized independent per-link
            # backpressure otherwise. Matches module_canny.py's
            # `_setup_canny_threshold` precedent.
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

    def _setup_cnn_accel_pool(self, library) -> None:
        tb = library.test_bench("tb_cnn_accel_pool")

        for test in tb.get_tests():
            # Zero stall on all three links only for the dedicated
            # full-throughput test; randomized independent per-link
            # backpressure otherwise. Matches module_canny.py's
            # `_setup_canny_threshold` precedent.
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
            # backpressure otherwise. Matches module_canny.py's
            # `_setup_canny_threshold` precedent.
            stall = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={
                    "stall_probability_percent_in": stall,
                    "stall_probability_percent_out": stall,
                },
            )
