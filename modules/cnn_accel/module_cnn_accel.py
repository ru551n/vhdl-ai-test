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
        return [
            build(
                name="cnn_accel_bias_requant",
                generics={
                    "g_accum_width": _ACCUM_WIDTH,
                    # NOT `_PE_ROWS` (8), deliberately. This entity still
                    # carries its accumulator input on the fixed 128-bit
                    # `axi_stream_m2s_t`, so its own
                    # `g_accum_width*g_pe_rows <= axi_stream_data_sz` assert
                    # caps it at 4 lanes -- 8 lanes x int32 is 256 bits and
                    # fails to elaborate at all. The pending fix is to move
                    # this port to the `accum_m2s_t`/`accum_s2m_t` records
                    # already in `cnn_accel_pkg.vhd` (added for exactly this
                    # by M2), which is part of the M6 conv_core work. Until
                    # then this build pins the largest legal design point so
                    # the entity still gets real synthesis coverage; raise
                    # it to `_PE_ROWS` as soon as the retrofit lands.
                    "g_pe_rows": 4,
                    "g_bias_addr_width": 9,
                },
                # Baseline 2026-09: 2348 LUTs, 34 FFs, 0 BRAM, 16 DSP.
                # The DSPs are the 4 lanes' requant multiplies.
                checkers=[
                    TotalLuts(LessThan(2900)),
                    Ffs(LessThan(60)),
                    BlockRams(LessThan(1)),
                    DspBlocks(LessThan(20)),
                ],
            ),
            build(
                name="cnn_accel_pool",
                generics={
                    "g_max_kernel_size": _POOL_MAX_KERNEL_SIZE,
                    "g_accum_width": _POOL_ACCUM_WIDTH,
                },
                # Baseline 2026-09: 342 LUTs, 27 FFs, 0 BRAM, 0 DSP.
                # Pure combinational reduction network, must stay tiny.
                checkers=[
                    TotalLuts(LessThan(450)),
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
                # Baseline 2026-09: 181 LUTs, 59 FFs, 72 block RAMs
                # (8 RAMB36 + 64 RAMB18), 0 DSP. The memories are meant to
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
                # Baseline 2026-09: 8884 LUTs, 199 FFs, 0 BRAM, 8 DSP.
                #
                # KNOWN BLOWUP, deliberately fenced rather than hidden: the
                # proposal (section 7) budgets ~3 BRAM36 and modest logic for
                # the line buffers, but the combinational random-access read
                # (K_h rows at once) blocks BRAM inference, so Yosys emits
                # 4608 RAM64M distributed-RAM cells plus the LUTs to mux
                # them. That is ~14% of an XC7A100T's LUTs for one small
                # block. Fixing it means giving the row banks a registered
                # read port -- tracked as an M7 optimization item.
                #
                # The limits below fence the *current* number so it cannot
                # silently grow further; tighten them hard once the read port
                # is registered and BRAM inference kicks in.
                checkers=[
                    TotalLuts(LessThan(9500)),
                    Ffs(LessThan(260)),
                    DspBlocks(LessThan(12)),
                ],
            ),
        ]

    def setup_vunit(self, vunit_proj: VUnit, **kwargs) -> None:
        library = vunit_proj.library(self.library_name)

        self._setup_cnn_accel_bias_requant(library)
        self._setup_cnn_accel_pool(library)

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
