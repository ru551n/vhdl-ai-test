from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tsfpga.module import BaseModule, get_modules

from ghdl_yosys_env import (
    resolve_ghdl_plugin_path,
    resolve_ghdl_prefix,
    resolve_vivado_path,
)

if TYPE_CHECKING:
    from vunit.ui import VUnit

# Reference hardware configuration, from
# doc/cnn_accel_tiled_dataflow_proposal.md section 7 ("BRAM/DSP estimate").
# These are the generic values every per-entity netlist build below is sized
# for, so the recorded resource numbers are all from one coherent design
# point rather than a per-module free-for-all. D12: target XC7A100T.
_PE_ROWS = 8
_PE_COLS = 8
_TILE_CHANNELS = 8  # = _PE_COLS, see proposal section 3.
_MAX_KERNEL_SIZE = 3
_MAX_ROW_TILE_WORDS = 512
# = K^2 * ceil(C_max/8) = 9*32 for the target backbone's worst layer
# (3x3x256, layer 9) -- replaces the old arbitrary 512 now that weights are
# streamed per output-channel pass from DDR4 instead of double-buffered
# on-chip (see doc/cnn_accel_weight_buffer.md's depth-sizing note).
_WEIGHT_BUFFER_DEPTH = 288
# Independent of _WEIGHT_BUFFER_DEPTH: a real layer only ever needs
# ceil(out_channels/g_pe_rows) bias rows, far fewer than the weight region.
_BIAS_BUFFER_DEPTH = 8
_ACCUM_WIDTH = 32

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


class Module(BaseModule):
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
            names_include={"axi_stream", "common", "math", "fifo"},
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

        # Resource limits are `LessThan` guard rails set a little above the
        # measured baseline, not exact targets: they exist to make CI shout
        # when a change unexpectedly blows up an entity's size, while
        # tolerating the small jitter that comes with Yosys version bumps.
        #
        # Baselines are measured on CI's toolchain -- Yosys v0.68 *release*,
        # as shipped in ru551n/hdl-docker:1.2.0 -- because CI is what these
        # limits actually gate. Do not re-baseline LUT counts from a local
        # Yosys: the difference is not jitter. Yosys v0.68+182 (dev) gives
        # markedly smaller LUT counts for the same RTL (window_gen 8884 vs
        # 23757, bias_requant 2348 vs 3686, pool 342 vs 468), so a locally
        # derived LUT limit fails on CI for no design reason at all. FFs,
        # DSPs and block RAMs are *not* subject to this: they are structural
        # counts that Yosys's optimizer cannot trade away regardless of
        # version, and this session confirmed it empirically for window_gen
        # and conv_core (M7) -- every FF/DSP/BlockRam figure measured
        # locally after the BRAM-inference fix landed exactly matches
        # arithmetic built from the old CI baselines of the untouched
        # submodules (see window_gen's and conv_core's own comments below),
        # so those three resources' limits below *are* re-baselined directly
        # from local numbers. Local builds are still useful for *relative*
        # before/after comparisons and, per the above, for FF/DSP/BRAM
        # absolute numbers too; only LUT absolute numbers must come from CI.
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
                checkers=[
                    TotalLuts(LessThan(7800)),
                    Ffs(LessThan(100)),
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
                checkers=[
                    TotalLuts(LessThan(550)),
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
                checkers=[
                    TotalLuts(LessThan(1000)),
                    Ffs(LessThan(1200)),
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
                # estimate of ~7430, so 12000 leaves comfortable headroom
                # above that estimate without being anywhere near the old
                # figures. TODO: tighten to the real CI-measured number the
                # first time this build runs on CI.
                checkers=[
                    TotalLuts(LessThan(12000)),
                    Ffs(LessThan(900)),
                    # Exact, not an upper bound: 0 BRAM (inference silently
                    # broken again, the whole point of M7) must fail CI just
                    # as loudly as an unexpected increase would. Safe as an
                    # equality because block-RAM counts are structural and
                    # do not move between Yosys versions -- see the
                    # additivity cross-check in conv_core's comment below.
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
                checkers=[
                    TotalLuts(LessThan(4100)),
                    Ffs(LessThan(1300)),
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
                #   FF:   789 + 1093 + 1119 + 66 = 3067 (measured 3066, -1)
                #   LUT:  2753 + 881 + 2888 + 4658 = 11180 (measured 11116,
                #                    -64 from cross-boundary optimization)
                # so those three limits below are re-baselined straight from
                # this measurement. Note the FF limit had to move a long way
                # (2200 -> 3200): M7b traded ~1030 FFs for 57 BRAM in
                # weight_buffer, and this entity inherits all of it.
                #
                # Cross-checked against real vendor synthesis -- see the
                # `cnn_accel_conv_core_vivado` project registered below
                # (xc7a200tfbg484-2, out-of-context): 15358 LUTs, 2824 FFs,
                # 14 RAMB36 + 2 RAMB18, 36 DSP, 0 LUTRAM. The BRAM story
                # agrees (15 RAMB36-equivalents vs Yosys's 18, both far
                # below the old 72), which is what mattered here. The other
                # three differ structurally rather than noisily, and neither
                # tool's number belongs in the other's limit:
                #   - DSP 36 vs 106: Vivado packs the 8x8 MAC array two
                #     8-bit multiplies per DSP48E1 (32 DSPs for 64 lanes,
                #     confirmed from its own DSP report), which Yosys does
                #     not do. Since xc7a200t has 740 DSPs and DSP is not the
                #     scarce resource here, this is headroom, not a problem.
                #   - LUT 15358 vs 11116 is the other side of that trade.
                #   - Vivado puts weight_buffer's prefetch FIFO in block RAM
                #     (hence 0 LUTRAM) where Yosys uses 49 RAM32M.
                #
                # LUTs remain CI-sensitive (see window_gen's own comment and
                # the module-level note above). Composing window_gen's own
                # ~2.7x local-to-CI LUT estimate (2753 -> ~7430) with
                # pe_array's and bias_requant's *unchanged* CI-measured
                # baselines (3474 + 7255), weight_buffer's new but
                # local-only 881, and ~380 of glue, gives a composed CI
                # estimate of ~19400 LUTs -- roughly a 45% reduction from
                # the old 35056. Per this project's standing rule (see
                # window_gen's own comment above and proposal doc section
                # 7.1 item 6) that estimate is not itself a CI measurement,
                # so the limit below is a deliberately loose PROVISIONAL
                # guard rail (far below the old 35056/37000, comfortable
                # headroom above the ~19400 estimate) rather than a tight
                # re-baseline. TODO: tighten to the real CI-measured number
                # the first time this build runs on CI.
                checkers=[
                    TotalLuts(LessThan(24000)),
                    Ffs(LessThan(3200)),
                    BlockRams(LessThan(20)),
                    DspBlocks(LessThan(115)),
                ],
            ),
        ]

        # Vendor-accurate cross-check of the composition entity, registered
        # ONLY when this machine actually has Vivado -- see
        # ghdl_yosys_env.resolve_vivado_path()'s docstring for why that guard
        # is mandatory rather than defensive (CI's hdl-docker image has no
        # Vivado, and CI builds every registered project with no filter).
        #
        # Why Vivado for this one entity and Yosys for the rest: conv_core is
        # the only build big enough for the Yosys run to take ~14 minutes,
        # and it is also the only one whose numbers are a *system* footprint
        # claim ("does the accelerator fit an XC7A200T") rather than a
        # relative before/after guard rail. Vendor synthesis is the right
        # tool for the former; Yosys, which is fast and already wired into
        # CI, is the right tool for the latter. The Yosys conv_core build
        # above is NOT replaced by this -- it stays the CI-gating one.
        #
        # Deliberately no build_result_checkers here. The Yosys and Vivado
        # numbers are not comparable (different technology mapping), so a
        # limit derived from one must never be attached to the other, and
        # this project's baselines are all Yosys-derived. This project exists
        # to be read, not to gate.
        vivado_path = resolve_vivado_path()
        if vivado_path is not None:
            from tsfpga.vivado.project import VivadoNetlistProject

            projects.append(
                VivadoNetlistProject(
                    name="cnn_accel_conv_core_vivado",
                    modules=modules,
                    part=_VIVADO_PART,
                    top="cnn_accel_conv_core",
                    generics={
                        "g_max_kernel_size": _MAX_KERNEL_SIZE,
                        "g_pe_rows": _PE_ROWS,
                        "g_pe_cols": _PE_COLS,
                        "g_accum_width": _ACCUM_WIDTH,
                        "g_tile_channels": _TILE_CHANNELS,
                        "g_max_row_tile_words": _MAX_ROW_TILE_WORDS,
                        "g_weight_buffer_depth": _WEIGHT_BUFFER_DEPTH,
                        "g_bias_buffer_depth": _BIAS_BUFFER_DEPTH,
                    },
                    vivado_path=vivado_path,
                    defined_at=Path(__file__),
                )
            )

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
        # 'vectors_path' must be an absolute path (the simulator's own
        # working directory is not this module's concern) -- self.path is
        # tsfpga's own per-module root, so this is correct regardless of
        # where run.py/vunit-mcp actually invokes the simulator from.
        tb = library.test_bench("tb_cnn_accel_pe_array_from_vectors")
        vectors_path = self.path / "test" / "vectors" / "pe_array_xlang_check"

        for test in tb.get_tests():
            self.add_vunit_config(
                test=test,
                generics={"vectors_path": str(vectors_path)},
            )

    def _setup_cnn_accel_conv_core(self, library) -> None:
        # Cross-language (Python golden model -> composed RTL) bit-exactness
        # guard for the full window_gen -> pe_array -> bias_requant
        # composition, see tb_cnn_accel_conv_core.vhd's own header comment.
        # 'vectors_root' must be an absolute path (the simulator's own
        # working directory is not this module's concern) -- self.path is
        # tsfpga's own per-module root, so this is correct regardless of
        # where run.py/vunit-mcp actually invokes the simulator from.
        tb = library.test_bench("tb_cnn_accel_conv_core")
        vectors_root = self.path / "test" / "vectors"

        for test in tb.get_tests():
            # Zero stall on both links only for the dedicated
            # full-throughput test (proves sustained back-to-back
            # operation); randomized independent per-link backpressure
            # otherwise. Matches every other cnn_accel testbench's
            # identical `_setup_*` precedent.
            stall = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={
                    "stall_probability_percent_in": stall,
                    "stall_probability_percent_out": stall,
                    "vectors_root": str(vectors_root),
                },
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
