from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tsfpga.module import BaseModule, get_modules

from ghdl_yosys_env import resolve_ghdl_plugin_path, resolve_ghdl_prefix

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
_WEIGHT_BUFFER_DEPTH = 512
_ACCUM_WIDTH = 32

# Pooling is a separate, much smaller kernel bound: the target network only
# pools 2x2. The entity's own contract is `g_max_kernel_size**2 * 8 <= 128`,
# so 3 is the largest useful value; kept at 3 for headroom over 2x2.
_POOL_MAX_KERNEL_SIZE = 3
# Must hold `_POOL_MAX_KERNEL_SIZE**2 * 127` = 1143 without overflow.
_POOL_ACCUM_WIDTH = 16


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
            Ffs,
            LessThan,
            TotalLuts,
        )
        from tsfpga.yosys.project import YosysXilinxNetlistBuild

        modules = get_modules(
            modules_folder=self.path.parent, names_include={self.name}
        ) + get_modules(
            modules_folder=self.path.parent.parent / "hdl-modules" / "modules",
            names_include={"axi_stream", "common", "math"},
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
        # limits actually gate. Do not re-baseline from a local Yosys: the
        # difference is not jitter. Yosys v0.68+182 (dev) gives markedly
        # smaller netlists for the same RTL (window_gen 8884 vs 23757 LUTs,
        # bias_requant 2348 vs 3686, pool 342 vs 468), so a locally derived
        # limit fails on CI for no design reason at all. Local builds are
        # still useful for *relative* before/after comparisons; only the
        # absolute numbers below are CI's.
        return [
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
                # Baseline 2026-09 (Yosys v0.68 release) at the old 4-lane
                # pin (`g_pe_rows`=4): 3686 LUTs, 34 FFs, 0 BRAM, 16 DSP
                # (the 4 lanes' requant multiplies).
                # TODO: re-baseline from CI now that `g_pe_rows` is 8 -- the
                # limits below are provisional (roughly double the 4-lane
                # numbers, with headroom), not yet a measured CI baseline.
                # As with every other entity here, only CI's Yosys v0.68
                # *release* numbers are the real baseline; a local
                # Yosys v0.68+182 (dev) build gives markedly smaller
                # netlists for the same RTL and must not be used to set
                # these limits.
                checkers=[
                    TotalLuts(LessThan(8700)),
                    Ffs(LessThan(130)),
                    BlockRams(LessThan(1)),
                    DspBlocks(LessThan(40)),
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
                    "g_pe_rows": _PE_ROWS,
                    "g_pe_cols": _PE_COLS,
                    "g_accum_width": _ACCUM_WIDTH,
                },
                # Baseline 2026-09 (Yosys v0.68 release): 175 LUTs, 59 FFs,
                # 72 block RAMs (8 RAMB36 + 64 RAMB18), 0 DSP. This is the
                # one entity whose size barely moved between Yosys versions,
                # because it is almost entirely hard BRAM rather than
                # optimizable logic. The memories are meant to
                # be BRAM, so `BlockRams` here is a floor-ish sanity bound
                # rather than a "keep it small" one -- if this ever drops to
                # 0 the memories have silently fallen back to distributed
                # LUT RAM and `TotalLuts` will blow up instead.
                checkers=[
                    TotalLuts(LessThan(300)),
                    Ffs(LessThan(80)),
                    BlockRams(LessThan(80)),
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
                # Baseline 2026-09 (Yosys v0.68 release): 23950 LUTs,
                # 199 FFs, 0 BRAM, 8 DSP. The D15 change (array payloads for the
                # window record) added ~193 LUTs from the prior measured 23757.
                #
                # KNOWN BLOWUP, deliberately fenced rather than hidden: the
                # proposal (section 7) budgets ~3 BRAM36 and modest logic for
                # the line buffers, but the combinational random-access read
                # (K_h rows at once) blocks BRAM inference, so Yosys emits
                # 4608 RAM64M distributed-RAM cells plus the LUTs to mux
                # them. On CI's Yosys that is ~37% of an XC7A100T's LUTs for
                # one small block (a local dev Yosys folds it to 8884, which
                # is why the CI number is the one that counts). Fixing it
                # means giving the row banks a registered read port --
                # tracked as an M7 optimization item, and the single biggest
                # area win available right now.
                #
                # The limits below fence the *current* number so it cannot
                # silently grow further; tighten them hard once the read port
                # is registered and BRAM inference kicks in.
                checkers=[
                    TotalLuts(LessThan(25500)),
                    Ffs(LessThan(260)),
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
        ]

    def setup_vunit(self, vunit_proj: VUnit, **kwargs) -> None:
        library = vunit_proj.library(self.library_name)

        self._setup_cnn_accel_bias_requant(library)
        self._setup_cnn_accel_pool(library)
        self._setup_cnn_accel_pe_array(library)
        self._setup_cnn_accel_pe_array_from_vectors(library)

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
