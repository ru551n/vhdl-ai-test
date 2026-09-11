"""The Python half of `test/tb_cnn_accel_top.vhd`.

`tb_cnn_accel_top` is deliberately dumb: it writes a DDR image, pokes
`PROGRAM_BASE_ADDR` + `CTRL.START`, waits for `DONE`/`ERROR`, then reads
back the CSR counters and a byte region -- all of it live, over VUnit's
Python FFI (`python_call`/`python_execute`, bridged through
`test/python_bridge/top_level_bridge.py`). No file is read or written
anywhere in this path. Every decision about *what* to run and *whether
the result is right* lives here, so adding a test adds a Python function
and never VHDL (arch doc section 11).

A `TbCase` bundles the four things one VUnit config needs:

* the `Model` (what to compute) and its `PlannedProgram` (where every
  tensor lives, and the predicted DDR traffic),
* the `ProgramImage` (descriptor chain + weights + preloaded inputs) --
  `compiled_regions`/`compiled_region_bytes` and `input_region`/
  `input_bytes` are the live-FFI views the bridge reads from it,
* the VUnit generics that tell the testbench the program entry point,
  the export window and the scratchpad geometry,
* `check_live`, the one verification entry point.

The verification `check_live` performs is deliberately in two layers:

1. **Data**: every graph output is read back out of the exported DDR
   bytes -- i.e. out of the bytes the DUT itself wrote to the memory
   model over AXI -- unpacked from the hardware's plane layout and
   compared element by element against `reference.run_reference`.
   Nothing is compared against a value the DUT reported about itself.
2. **Traffic**: the DUT's own CSR counters are compared against
   `Planner`'s prediction *and* against the testbench's passive AXI
   monitor (the `axi_*` counters). The residency invariants of arch doc
   section 10 are the whole point of rev 2, and a counter that only
   agrees with itself proves nothing -- `DDR_WR_BYTES` must agree with
   independently observed W-channel handshakes, or the "the intermediate
   never went to DDR" claim rests on the same logic it is testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cnn_accel_constants
import cnn_accel_model as golden
from accel_v2 import isa
from accel_v2.ddrmap import DdrMap
from accel_v2.memimage import MemoryImage
from accel_v2.model import Model, Tensor
from accel_v2.planner import (
    ComputeStep,
    ConstLoadStep,
    MoveStep,
    PlannedProgram,
    Planner,
    RowCopyStep,
    descriptor_count,
)
from accel_v2.program import ProgramImage, emit_program
from accel_v2.reference import ExecutionResult, run_reference

#: Bytes per `MemoryImage`/AXI word. Same constant the testbench derives
#: from the generated AXI data width.
WORD_BYTES = 8


class CheckFailure(Exception):
    """Raised (and caught) inside `TbCase.check_live` so that every
    failure is reported through one formatter."""


# ---------------------------------------------------------------------------
# Traffic-assertion policy
# ---------------------------------------------------------------------------


@dataclass
class TrafficPolicy:
    """Which traffic claims this case asserts.

    `write_bytes` is exact by default and is the decisive residency
    check: for a local chain it must equal exactly the closing `STORE`'s
    size (invariants R1/R2/R4).

    `read_bytes` is exact by default too, and has been since the
    prediction learned that the output-channel loop re-streams a
    convolution's ifmap once per tile (`planner.ifmap_passes`) and that
    every `ACT` refetches its 256-byte LUT. Those two were the whole of
    the gap: with them charged, all 56 non-error catalogue cases predict
    `DDR_RD_BYTES` to the byte, descriptors, weights, bias, scale tables,
    LUTs and operands included. A lower bound was the honest claim while
    the prediction was knowingly incomplete; it is not any more, and an
    exact read figure is what makes "the fused group moved exactly its
    boundary bytes" assertable at all. A case that genuinely cannot
    predict its reads sets `read_bytes_exact=False` **with a written
    reason**.

    `local_bytes_at_most` checks the DUT's `LOCAL_BYTES` CSR against the
    predicted `local_read_bytes`/`local_write_bytes`, in the KiB
    granularity the counter reports. One-sided on purpose: the counter
    increments from two read ports (and two write ports) in two separate
    `if` statements of one clocked process, so two ports handshaking in
    the same cycle contribute one word instead of two and the register
    *undercounts*. Measured over the catalogue that costs at most 1 KiB
    on the read side and nothing at all on the write side. What the check
    can still say without qualification is the direction that matters:
    the DUT must never move MORE local bytes than the program logically
    needs, which is what unmodelled scratchpad traffic would look like.
    """

    write_bytes_exact: bool = True
    read_bytes_exact: bool = True
    read_bytes_at_least: bool = True
    counts_exact: bool = True
    weight_bytes_exact: bool = True
    local_bytes_at_most: bool = True
    #: Cross-check the DUT's `DDR_*_BYTES` against the testbench's own
    #: passive AXI monitor. Only ever disabled with a written reason.
    monitor_cross_check: bool = True
    #: `[(lo, hi_exclusive), ...]`: every observed `AWADDR` must fall in
    #: one of these ranges. `None` derives it from the case's own
    #: expected write destinations.
    allowed_write_ranges: list[tuple[int, int]] | None = None


# ---------------------------------------------------------------------------
# TbCase
# ---------------------------------------------------------------------------


@dataclass
class TbCase:
    """One VUnit config's worth of test: a planned program, the DDR image
    that runs it, and the checks its result must satisfy."""

    name: str
    model: Model
    planned: PlannedProgram
    program: ProgramImage
    export_base: int
    export_bytes: int
    num_banks: int
    bank_words: int
    expect_error: bool = False
    expect_err_code: int | None = None
    #: Expected `STATUS.ERR_PC_LOW` (the low 16 bits of the faulting
    #: descriptor's byte address), checked verbatim against the counter
    #: when set. `None` (the default) skips the check -- most
    #: `expect_error` cases only care about the code, not exactly which
    #: descriptor raised it.
    expect_err_pc: int | None = None
    traffic: TrafficPolicy = field(default_factory=TrafficPolicy)
    #: Extra generic overrides merged last into `generics()`.
    generic_overrides: dict[str, object] = field(default_factory=dict)
    #: One additional check, run first by `check_live`, for a catalogue
    #: that needs to assert something about the *plan* itself (not
    #: runtime data) -- e.g. "no spill/reload step at all" (cases_tiling.py,
    #: cases_yolo.py, cases_concat_split.py's own `_with_extra_check`).
    #: Takes the case itself and raises `CheckFailure`/`GeometryError` on
    #: disagreement; `None` (the default) runs nothing extra.
    extra_check: Callable[["TbCase"], None] | None = None
    _expected: ExecutionResult | None = field(default=None, repr=False)

    # -- VUnit plumbing ---------------------------------------------------

    def generics(self) -> dict[str, object]:
        """The generics for `add_vunit_config`. `output_path` and
        `g_case_name` are filled in by the caller (VUnit itself, and
        `module_cnn_accel.py`'s registration loop, respectively) and must
        not appear here. Every other per-case value -- the program base
        address, the export/input DDR windows, whether the program is
        expected to error -- is fetched live by the testbench instead
        (`top_level_bridge.get_program_start_address`/`get_output_region`/
        `get_input_region`/`get_expect_error`), so it is not duplicated
        into a generic here either: only values that affect DUT/testbench
        ELABORATION (generics feeding a `generic map`, fixed before any
        `python_call` is even possible) belong in this dict."""
        # Sized from THIS case's own map, not from the class constant: a
        # tiled case plans against `DdrMap(scale=N)` (one descriptor per
        # plane per row copy runs the default 60 KiB PROGRAM region out),
        # and the testbench's DDR model plus its range validation must
        # cover the addresses that map actually hands out. Identical to
        # `DdrMap.LIMIT` for every scale-1 case, which is all of them
        # outside `cases_tiling.py`.
        ddr_bytes = self.planned.ddr_map.limit
        generics: dict[str, object] = {
            "g_ddr_bytes": ddr_bytes,
            "g_num_banks": self.num_banks,
            "g_bank_words": self.bank_words,
            "g_ddr_limit": ddr_bytes,
            # The catalogue's tensors are deliberately tiny (a passing case
            # finishes in a few thousand cycles), so the entity defaults of
            # 1M/2M cycles only ever cost wall-clock time: a genuinely stuck
            # DUT sits in the simulator for ~12 minutes before its watchdog
            # fires. The longest healthy case (local_chain_4op) measures
            # 6,953 cycles, so the 100k testbench bound below is ~14x that.
            # 'g_watchdog_cycles' is not a per-program bound at all: it is
            # cmd_proc's per-state watchdog, reloaded on every state change
            # (~13 reload sites in cmd_proc), so it only fires if a single
            # state hangs, not from a long-but-healthy program accumulating
            # many states. Together these bounds cannot mask a
            # slow-but-correct run, and they keep the "a stuck program must
            # ERROR, never hang" contract intact.
            "g_watchdog_cycles": 50_000,
            "g_timeout_cycles": 100_000,
        }
        generics.update(self.generic_overrides)
        return generics

    def input_region(self) -> tuple[int, int]:
        """`(base_addr, num_bytes)` of the DDR `INPUTS` region this case
        actually needs written: `DdrMap.INPUTS`'s fixed bounds, narrowed to the
        bytes really written there (the region's own size is generous
        headroom, not this case's usage). `num_bytes` is 0 for a case
        with no graph inputs to write (e.g. an `expect_error` case that
        starts a program with none)."""
        lo, hi = self.planned.ddr_map.region_bounds(DdrMap.INPUTS)
        words = self.program.image.words_in_range(lo, hi)
        if not words:
            return lo, 0
        return lo, (max(words) - lo) + WORD_BYTES

    def input_bytes(self) -> bytes:
        """The exact packed bytes of the `INPUTS` region -- the same
        bytes `pre_config` would otherwise have written into
        `mem_image.csv` there, produced by the same compiler pipeline
        (`accel_v2.program.emit_program`). Read live by
        `top_level_bridge.get_input_data` (test/python_bridge/
        top_level_bridge.py) instead of being written to a file."""
        lo, nbytes = self.input_region()
        return self.program.image.read_bytes(lo, nbytes)

    def compiled_regions(self) -> list[tuple[int, int]]:
        """`[(addr, num_bytes), ...]`, ascending by address: the
        contiguous byte runs of everything the compiler itself produces
        (the descriptor chain and the weight/bias/scale/LUT tables),
        with the `INPUTS` region excluded -- those bytes are written
        separately, via `input_bytes` above (renamed `get_input_data` in
        the bridge). Runs rather than one
        `[lowest, highest]` span: `DdrMap`'s regions sit at fixed,
        far-apart bases regardless of how much of each a given case
        actually uses, so spanning the whole range would mean writing
        mostly unused gap bytes for every case."""
        lo, hi = self.planned.ddr_map.region_bounds(DdrMap.INPUTS)
        image = self.program.image.without_range(lo, hi)
        return image.regions()

    def compiled_region_bytes(self, index: int) -> bytes:
        """The bytes of `compiled_regions()[index]` -- read live by
        `top_level_bridge.get_program_data` instead of being written
        to `mem_image.csv`."""
        addr, nbytes = self.compiled_regions()[index]
        return self.program.image.read_bytes(addr, nbytes)
        return True

    # -- derived values ---------------------------------------------------

    @property
    def tensor_mem_bytes(self) -> int:
        return self.num_banks * self.bank_words * WORD_BYTES

    @property
    def expected(self) -> ExecutionResult:
        """Bit-exact reference result. Computed lazily and cached: it must
        not run during `run.py --list`/`--compile`, only when a test
        actually needs checking.

        Runs against a *fresh* `MemoryImage`, never against
        `self.program.image`: `run_reference` mutates the image it is
        given exactly as the real DDR would, and letting it write into
        the DUT's preload would hand the DUT its own expected
        answers -- the most embarrassing possible backdoor."""
        if self._expected is None:
            self._expected = run_reference(self.planned, MemoryImage())
        return self._expected

    # -- checking ---------------------------------------------------------

    def check_live(self, counters: dict[str, int], export_base: int, export_bytes: list[int]) -> None:
        """Verify the run: called from `top_level_bridge.check_result`
        (test/python_bridge/top_level_bridge.py) via a `python_call`,
        right after `STATUS.DONE`/`STATUS.ERROR` fires inside the running
        simulation -- no `result.csv`/`counters.csv` file is read or
        written anywhere in this path. `export_bytes` is the raw exported
        region as plain Python ints (0..255, unsigned byte values -- the
        same convention `MemoryImage.write_bytes` expects), starting at
        `export_base`.

        Raises `CheckFailure`/`GeometryError` on the first disagreement,
        deliberately uncaught: `python_call` already turns an uncaught
        Python exception into a VUnit FAILURE with the full traceback
        (see `cnn_accel_python_ffi_pkg.vhd`'s callers), which is a
        better error report than a bool plus a hand-written `check_true`
        message could give VHDL -- so there is no `try`/`except` here,
        and no need for one at any call site either."""
        if self.extra_check is not None:
            self.extra_check(self)
        # HW_INFO/HW_INFO2/HW_INFO3 are read unconditionally by the
        # testbench regardless of how the program ends, so check them
        # unconditionally too, before branching on expect_error.
        self._check_hw_info(counters)
        self._check_status(counters)
        if self.expect_error:
            # A rejected program has no meaningful output tensors and no
            # meaningful traffic prediction: the whole point is that the
            # DUT stopped. The one traffic claim that still holds is the
            # arch doc's "never partially writes a validated-bad
            # command's destination" promise, checked below.
            self._check_error_wrote_nothing_unexpected(counters)
        else:
            exported = MemoryImage()
            exported.write_bytes(export_base, bytes(export_bytes))
            self._check_outputs_against(exported, "<live python_call, no file>")
            self._check_traffic(counters)

    def _check_hw_info(self, counters: dict[str, int]) -> None:
        """`HW_INFO`/`HW_INFO2`/`HW_INFO3` are read-only capability
        registers, driven straight from the elaborated generics
        (`module_cnn_accel.py`'s `hw_info`/`hw_info2`/`hw_info3`); this is
        the only place anything checks they actually read back what was
        elaborated, rather than merely getting dumped into `counters.csv`
        for a human to eyeball.

        Field widths/offsets are the generated register layout
        (`regs_src/cnn_accel_regs_pkg.vhd`): each register packs its
        fields LSB-first in declaration order, so bit positions are
        derived here from the field widths rather than hardcoded, to
        avoid silently reading the wrong bits if a field is ever resized.
        """
        generics = self.generics()

        def field(raw: int, lsb: int, width: int) -> int:
            return (raw >> lsb) & ((1 << width) - 1)

        expected_pe_rows = generics.get("g_pe_rows", cnn_accel_constants.PE_ROWS)
        expected_pe_cols = generics.get("g_pe_cols", cnn_accel_constants.PE_COLS)
        expected_tile_channels = generics.get("g_tile_channels", cnn_accel_constants.TILE_CHANNELS)
        expected_max_kernel_size = generics.get("g_max_kernel_size", cnn_accel_constants.MAX_KERNEL_SIZE)

        hw_info = counters["hw_info"]
        got_pe_rows = field(hw_info, 0, 8)
        got_pe_cols = field(hw_info, 8, 8)
        got_tile_channels = field(hw_info, 16, 8)
        got_max_kernel_size = field(hw_info, 24, 8)
        if (got_pe_rows, got_pe_cols, got_tile_channels, got_max_kernel_size) != (
            expected_pe_rows,
            expected_pe_cols,
            expected_tile_channels,
            expected_max_kernel_size,
        ):
            raise CheckFailure(
                f"HW_INFO=0x{hw_info:08x} decodes to "
                f"pe_rows={got_pe_rows}, pe_cols={got_pe_cols}, "
                f"tile_channels={got_tile_channels}, max_kernel_size={got_max_kernel_size}; "
                f"expected pe_rows={expected_pe_rows}, pe_cols={expected_pe_cols}, "
                f"tile_channels={expected_tile_channels}, max_kernel_size={expected_max_kernel_size}"
            )

        expected_isa_version = cnn_accel_constants.ISA_VERSION
        expected_tensor_mem_kib = self.tensor_mem_bytes // 1024

        hw_info2 = counters["hw_info2"]
        got_isa_version = field(hw_info2, 0, 16)
        got_tensor_mem_kib = field(hw_info2, 16, 16)
        if (got_isa_version, got_tensor_mem_kib) != (expected_isa_version, expected_tensor_mem_kib):
            raise CheckFailure(
                f"HW_INFO2=0x{hw_info2:08x} decodes to "
                f"isa_version=0x{got_isa_version:04x}, tensor_mem_kib={got_tensor_mem_kib}; "
                f"expected isa_version=0x{expected_isa_version:04x}, "
                f"tensor_mem_kib={expected_tensor_mem_kib}"
            )

        expected_max_pool_kernel_size = generics.get(
            "g_max_pool_kernel_size", cnn_accel_constants.MAX_POOL_KERNEL_SIZE
        )
        expected_max_row_tile_words = generics.get(
            "g_max_row_tile_words", cnn_accel_constants.MAX_ROW_TILE_WORDS
        )

        hw_info3 = counters["hw_info3"]
        got_max_pool_kernel_size = field(hw_info3, 0, 8)
        got_max_row_tile_words = field(hw_info3, 8, 16)
        if (got_max_pool_kernel_size, got_max_row_tile_words) != (
            expected_max_pool_kernel_size,
            expected_max_row_tile_words,
        ):
            raise CheckFailure(
                f"HW_INFO3=0x{hw_info3:08x} decodes to "
                f"max_pool_kernel_size={got_max_pool_kernel_size}, "
                f"max_row_tile_words={got_max_row_tile_words}; expected "
                f"max_pool_kernel_size={expected_max_pool_kernel_size}, "
                f"max_row_tile_words={expected_max_row_tile_words}"
            )

    def _check_status(self, counters: dict[str, int]) -> None:
        if self.expect_error:
            if counters["error"] != 1:
                raise CheckFailure(
                    f"expected STATUS.ERROR, but error={counters['error']} "
                    f"done={counters['done']} status=0x{counters['status']:08x}"
                )
            if self.expect_err_code is not None and counters["err_code"] != self.expect_err_code:
                raise CheckFailure(
                    f"expected ERR_CODE 0x{self.expect_err_code:x} "
                    f"({_err_name(self.expect_err_code)}), got "
                    f"0x{counters['err_code']:x} ({_err_name(counters['err_code'])}) "
                    f"at ERR_PC_LOW=0x{counters['err_pc_low']:04x}"
                )
            if self.expect_err_pc is not None:
                expected_pc_low = self.expect_err_pc & 0xFFFF
                if counters["err_pc_low"] != expected_pc_low:
                    raise CheckFailure(
                        f"expected ERR_PC_LOW=0x{expected_pc_low:04x} (from PC "
                        f"0x{self.expect_err_pc:08x}), got "
                        f"0x{counters['err_pc_low']:04x} -- ERR_CODE="
                        f"0x{counters['err_code']:x} ({_err_name(counters['err_code'])})"
                    )
            return

        if counters["error"] != 0:
            raise CheckFailure(
                f"unexpected DUT error: ERR_CODE=0x{counters['err_code']:x} "
                f"({_err_name(counters['err_code'])}) at "
                f"ERR_PC_LOW=0x{counters['err_pc_low']:04x}, "
                f"status=0x{counters['status']:08x}\n"
                f"{self._program_listing()}"
            )
        if counters["done"] != 1:
            raise CheckFailure(f"program did not reach DONE (status=0x{counters['status']:08x})")
        if counters["busy"] != 0:
            raise CheckFailure(f"STATUS.BUSY still set after DONE (status=0x{counters['status']:08x})")

        # One descriptor retired per planned step, plus the closing HALT
        # -- except a `RowCopyStep` (one per activation plane) and a
        # `ConstLoadStep` (one per packed weight sub-image), which is
        # what `planner.descriptor_count` is for. Identical to
        # `len(steps) + 1` for every untiled case.
        expected_cmds = sum(descriptor_count(step) for step in self.planned.steps) + 1
        if counters["cmd_count"] != expected_cmds:
            raise CheckFailure(
                f"CMD_COUNT={counters['cmd_count']}, expected {expected_cmds} "
                f"({len(self.planned.steps)} steps, {expected_cmds - 1} descriptors "
                f"+ HALT)\n{self._program_listing()}"
            )

    def _check_outputs_against(self, exported: "MemoryImage", source_description: str) -> None:
        """Compare every graph output tensor in `exported` (built by
        `check_live` from the bytes a `python_call` handed over) against
        `self.expected`."""
        if self.export_bytes == 0:
            raise CheckFailure(
                "case exports no bytes, so no output can be verified -- "
                "either mark a graph output or set expect_error"
            )

        for tensor in self.model.outputs:
            addr = self.planned.tensor_ddr_addr[tensor.name]
            if addr < self.export_base or addr + tensor.size_bytes > self.export_base + self.export_bytes:
                raise CheckFailure(
                    f"output '{tensor.name}' at 0x{addr:08x}+{tensor.size_bytes} lies outside "
                    f"the exported window [0x{self.export_base:08x}, "
                    f"0x{self.export_base + self.export_bytes:08x}) -- harness bug, not a DUT bug"
                )
            raw = exported.read_bytes(addr, tensor.size_bytes)
            actual = golden.unpack_activation_planes(
                [b - 256 if b >= 128 else b for b in raw],
                tensor.width,
                tensor.height,
                tensor.channels,
            )
            expected = self.expected.tensor_data[tensor.name]
            self._compare_tensor(tensor, expected, actual, addr, source_description)

    def _compare_tensor(
        self,
        tensor: Tensor,
        expected: list[int],
        actual: list[int],
        addr: int,
        source_description: str,
    ) -> None:
        if len(expected) != len(actual):
            raise CheckFailure(
                f"tensor '{tensor.name}': reference has {len(expected)} elements, "
                f"read back {len(actual)}"
            )
        for index, (want, got) in enumerate(zip(expected, actual)):
            if want == got:
                continue
            # Logical layout is HWC, matching golden.unpack_activation_planes.
            h = index // (tensor.width * tensor.channels)
            rem = index % (tensor.width * tensor.channels)
            w = rem // tensor.channels
            c = rem % tensor.channels
            n_wrong = sum(1 for a, b in zip(expected, actual) if a != b)
            raise CheckFailure(
                f"tensor '{tensor.name}' mismatch\n"
                f"  shape (HxWxC)   : {tensor.height}x{tensor.width}x{tensor.channels}"
                f" ({len(expected)} elements, {tensor.size_bytes} packed bytes)\n"
                f"  first mismatch  : h={h} w={w} c={c} (flat HWC index {index})\n"
                f"  expected        : {want}\n"
                f"  actual          : {got}\n"
                f"  total mismatches: {n_wrong} of {len(expected)}\n"
                f"  produced by     : {self._producer_description(tensor)}\n"
                f"  read back from  : DDR 0x{addr:08x} (+{tensor.size_bytes} bytes), "
                f"{source_description}\n"
                f"{self._program_listing()}"
            )

    def _check_traffic(self, counters: dict[str, int]) -> None:
        predicted = self.planned.traffic
        policy = self.traffic

        if policy.monitor_cross_check:
            # The DUT's self-reported byte counts versus the testbench's
            # passive W/R-handshake monitor. A disagreement means either
            # the counters lie or traffic happened that the CSR does not
            # admit to -- both fatal to every residency claim below.
            if counters["ddr_wr_bytes"] != counters["axi_wr_bytes"]:
                raise CheckFailure(
                    f"DUT DDR_WR_BYTES={counters['ddr_wr_bytes']} disagrees with the "
                    f"testbench's independently observed AXI write bytes "
                    f"{counters['axi_wr_bytes']} ({counters['axi_wr_beats']} W beats in "
                    f"{counters['axi_aw_count']} transactions)"
                )
            if counters["ddr_rd_bytes"] != counters["axi_rd_bytes"]:
                raise CheckFailure(
                    f"DUT DDR_RD_BYTES={counters['ddr_rd_bytes']} disagrees with the "
                    f"testbench's independently observed AXI read bytes "
                    f"{counters['axi_rd_bytes']} ({counters['axi_rd_beats']} R beats in "
                    f"{counters['axi_ar_count']} transactions)"
                )

        if policy.write_bytes_exact and counters["ddr_wr_bytes"] != predicted.write_bytes:
            raise CheckFailure(
                f"DDR write traffic {counters['ddr_wr_bytes']} bytes, expected exactly "
                f"{predicted.write_bytes}.\n"
                f"  This is the residency check (arch doc section 10): only explicit "
                f"STORE/spill instructions may write DDR.\n"
                f"  expected stores: {self._store_description()}\n"
                f"{self._program_listing()}"
            )

        if policy.read_bytes_exact:
            if counters["ddr_rd_bytes"] != predicted.read_bytes:
                raise CheckFailure(
                    f"DDR read traffic {counters['ddr_rd_bytes']} bytes, expected exactly "
                    f"{predicted.read_bytes}\n{self._program_listing()}"
                )
        elif policy.read_bytes_at_least and counters["ddr_rd_bytes"] < predicted.read_bytes:
            raise CheckFailure(
                f"DDR read traffic {counters['ddr_rd_bytes']} bytes is *less* than the "
                f"{predicted.read_bytes} bytes the program logically needs -- the DUT cannot "
                f"have fetched every descriptor, weight and input\n{self._program_listing()}"
            )

        if policy.counts_exact:
            for key, want, what in (
                ("tensor_load_count", predicted.tensor_load_count, "LOAD"),
                ("tensor_store_count", predicted.tensor_store_count, "STORE"),
            ):
                if counters[key] != want:
                    raise CheckFailure(
                        f"{key.upper()}={counters[key]}, expected {want} -- the program "
                        f"contains exactly {want} explicit {what} instruction(s), so any "
                        f"other value means hidden data movement\n{self._program_listing()}"
                    )

        if policy.weight_bytes_exact and counters["weight_load_bytes"] != predicted.weight_bytes:
            raise CheckFailure(
                f"WEIGHT_LOAD_BYTES={counters['weight_load_bytes']}, expected "
                f"{predicted.weight_bytes}\n{self._program_listing()}"
            )

        if policy.local_bytes_at_most:
            self._check_local_bytes(counters)

        ranges = policy.allowed_write_ranges
        if ranges is None:
            ranges = self._expected_write_ranges()
        if ranges and counters["axi_aw_count"] > 0:
            for key in ("axi_wr_lo_addr", "axi_wr_hi_addr"):
                addr = counters[key]
                if not any(lo <= addr < hi for lo, hi in ranges):
                    pretty = ", ".join(f"[0x{lo:08x}, 0x{hi:08x})" for lo, hi in ranges)
                    raise CheckFailure(
                        f"{key}=0x{addr:08x} lies outside every expected write "
                        f"destination {pretty} -- the DUT wrote DDR somewhere the program "
                        f"never told it to\n{self._program_listing()}"
                    )

    def _check_local_bytes(self, counters: dict[str, int]) -> None:
        """`LOCAL_BYTES` (section 8) packs `rd_kib` in bits [15:0] and
        `wr_kib` in [31:16], each the whole-KiB part of a byte counter --
        so the comparison is made in KiB, on the prediction's own floor.
        See `TrafficPolicy.local_bytes_at_most` for why this is one-sided."""
        predicted = self.planned.traffic
        raw = counters["local_bytes"]
        for measured, want, what in (
            (raw & 0xFFFF, predicted.local_read_bytes, "read"),
            ((raw >> 16) & 0xFFFF, predicted.local_write_bytes, "written"),
        ):
            if measured > want // 1024:
                raise CheckFailure(
                    f"LOCAL_BYTES says {measured} KiB {what} in the scratchpad, but the "
                    f"program only needs {want} bytes ({want // 1024} KiB) -- the DUT is "
                    f"moving local data the plan does not account for\n{self._program_listing()}"
                )

    def _check_error_wrote_nothing_unexpected(self, counters: dict[str, int]) -> None:
        ranges = self.traffic.allowed_write_ranges
        if ranges is None:
            return
        if not ranges and counters["axi_aw_count"] != 0:
            raise CheckFailure(
                f"a rejected program issued {counters['axi_aw_count']} AXI write "
                f"transaction(s) ({counters['axi_wr_bytes']} bytes, first at "
                f"0x{counters['axi_wr_lo_addr']:08x}); the arch doc requires that a "
                f"validated-bad command never partially writes its destination"
            )

    # -- diagnostics ------------------------------------------------------

    def _expected_write_ranges(self) -> list[tuple[int, int]]:
        """Every DDR byte range the program is allowed to write: the
        destination of each explicit `STORE`/spill step."""
        ranges: list[tuple[int, int]] = []
        for step in self.planned.steps:
            if isinstance(step, MoveStep) and step.dst_space == isa.SPACE_DDR:
                ranges.append((step.dst_addr, step.dst_addr + step.nbytes))
            elif isinstance(step, RowCopyStep) and step.dst_space == isa.SPACE_DDR:
                # One range per plane: a strip store writes `plane_count`
                # separate windows of the destination, and the bytes
                # between them belong to other strips.
                ranges.extend((dst, dst + n) for _src, dst, n in step.transfers)
            elif isinstance(step, ComputeStep) and step.output_space == isa.SPACE_DDR:
                ranges.append((step.output_addr, step.output_addr + step.op.output.size_bytes))
        return ranges

    def _store_description(self) -> str:
        parts = [
            f"0x{lo:08x}+{hi - lo}" for lo, hi in self._expected_write_ranges()
        ]
        return ", ".join(parts) if parts else "(none)"

    def _producer_description(self, tensor: Tensor) -> str:
        for index, step in enumerate(self.planned.steps):
            if isinstance(step, ComputeStep) and step.op.output.name == tensor.name:
                return (
                    f"command {index} {type(step.op).__name__} -> "
                    f"{_space_name(step.output_space)} 0x{step.output_addr:08x}"
                )
            if isinstance(step, MoveStep) and step.tensor.name == tensor.name:
                return (
                    f"command {index} {step.kind} "
                    f"{_space_name(step.src_space)} 0x{step.src_addr:08x} -> "
                    f"{_space_name(step.dst_space)} 0x{step.dst_addr:08x}"
                )
        if tensor.alias_parts:
            parts = ", ".join(
                f"{part.name}@plane{part.alias_plane_offset}" for part in tensor.alias_parts
            )
            return f"(CONCAT buffer -- written in slices by {parts})"
        if tensor.alias_parent is not None:
            return (
                f"(view of '{tensor.alias_parent.name}' at plane "
                f"{tensor.alias_plane_offset}, {tensor.alias_role})"
            )
        return "(graph input -- never produced by a command)"

    def _program_listing(self) -> str:
        """Disassembly-ish dump of the planned program, printed with every
        failure so a log line is enough to see what ran."""
        total = sum(descriptor_count(step) for step in self.planned.steps)
        lines = [
            f"  planned program ({len(self.planned.steps)} steps, {total} descriptors "
            "+ HALT):"
        ]
        cursor = 0
        for index, step in enumerate(self.planned.steps):
            addr = self.program.program_addr + cursor * isa.INSTR_WORD_BYTES
            cursor += descriptor_count(step)
            if isinstance(step, RowCopyStep):
                op = step.op
                lines.append(
                    f"    [{index:2d}] pc=0x{addr:08x} {'rowcopy':<10s} "
                    f"{op.output.name:<12s} <- {op.inputs[0].name} rows {op.src_rows.r0}.."
                    f"{op.src_rows.r1} @{_space_name(step.src_space)} -> rows "
                    f"{op.dst_rows.r0}..{op.dst_rows.r1} @{_space_name(step.dst_space)} "
                    f"({len(step.transfers)} planes, {step.nbytes} bytes)"
                )
            elif isinstance(step, ConstLoadStep):
                images = ", ".join(
                    f"{kind}@LOCAL 0x{local:08x} ({nbytes} bytes)"
                    for kind, local, nbytes in step.images
                )
                lines.append(
                    f"    [{index:2d}] pc=0x{addr:08x} {'constload':<10s} "
                    f"{step.conv.name:<12s} -> {images}"
                )
            elif isinstance(step, MoveStep):
                lines.append(
                    f"    [{index:2d}] pc=0x{addr:08x} {step.kind:<10s} "
                    f"{step.tensor.name:<12s} "
                    f"{_space_name(step.src_space)} 0x{step.src_addr:08x} -> "
                    f"{_space_name(step.dst_space)} 0x{step.dst_addr:08x} "
                    f"({step.nbytes} bytes)"
                )
            else:
                inputs = ", ".join(
                    f"{t.name}@{_space_name(s)} 0x{a:08x}"
                    for t, s, a in zip(step.op.inputs, step.input_spaces, step.input_addrs)
                )
                lines.append(
                    f"    [{index:2d}] pc=0x{addr:08x} {type(step.op).__name__:<10s} "
                    f"{step.op.output.name:<12s} <- {inputs} -> "
                    f"{_space_name(step.output_space)} 0x{step.output_addr:08x}"
                )
        lines.append(
            f"    [{len(self.planned.steps):2d}] pc="
            f"0x{self.program.program_addr + total * isa.INSTR_WORD_BYTES:08x} HALT"
        )
        lines.append(
            f"  predicted traffic: rd={self.planned.traffic.read_bytes} "
            f"wr={self.planned.traffic.write_bytes} "
            f"weights={self.planned.traffic.weight_bytes} "
            f"loads={self.planned.traffic.tensor_load_count} "
            f"stores={self.planned.traffic.tensor_store_count}"
        )
        return "\n".join(lines)


_ERR_NAMES = {
    0x0: "no error",
    0x1: "ERR_UNSUPPORTED_OP",
    0x2: "ERR_BAD_SPACE",
    0x3: "ERR_MISALIGNED",
    0x4: "ERR_LOCAL_RANGE",
    0x5: "ERR_DDR_RANGE",
    0x6: "ERR_BAD_RESERVED",
    0x7: "ERR_BAD_GEOMETRY",
    0x8: "ERR_AXI",
    0x9: "ERR_TIMEOUT",
}


def _err_name(code: int) -> str:
    return _ERR_NAMES.get(code, f"unknown code 0x{code:x}")


_SPACE_NAMES = {
    isa.SPACE_DDR: "DDR",
    isa.SPACE_LOCAL_TENSOR: "LOCAL",
    isa.SPACE_LOCAL_WEIGHT: "LWGT",
}


def _space_name(space: int) -> str:
    return _SPACE_NAMES.get(space, f"space{space}")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_case(
    name: str,
    build: "object",
    *,
    seed: int,
    num_banks: int = 2,
    bank_words: int = 1024,
    expect_error: bool = False,
    expect_err_code: int | None = None,
    expect_err_pc: int | None = None,
    traffic: TrafficPolicy | None = None,
    generic_overrides: dict[str, object] | None = None,
    ddr_scale: int = 1,
) -> TbCase:
    """Build one `TbCase`. `build(model)` adds the case's tensors and ops
    to a freshly seeded `Model` (and marks its graph outputs); everything
    downstream -- planning, DDR layout, descriptor emission -- follows
    from that.

    `seed` is explicit and mandatory: every randomized value in the case
    (inputs, weights, biases, per-channel scale tables) comes from it, so
    a failing case is reproducible from its name alone.

    `ddr_scale` grows every `DdrMap` region (and the testbench's modelled
    DDR with it, see `generics`). It exists for tiled cases: a strip
    program emits one descriptor per activation plane per row copy, which
    runs the default 60 KiB `PROGRAM` region out long before anything
    else. Leave it at 1 for an untiled case -- the DDR model costs real
    simulator memory.
    """
    model = Model(seed=seed, name=name)
    build(model)
    return case_from_model(
        name,
        model,
        num_banks=num_banks,
        bank_words=bank_words,
        expect_error=expect_error,
        expect_err_code=expect_err_code,
        expect_err_pc=expect_err_pc,
        traffic=traffic,
        generic_overrides=generic_overrides,
        ddr_scale=ddr_scale,
    )


def case_from_model(
    name: str,
    model: Model,
    *,
    num_banks: int = 2,
    bank_words: int = 1024,
    expect_error: bool = False,
    expect_err_code: int | None = None,
    expect_err_pc: int | None = None,
    traffic: TrafficPolicy | None = None,
    generic_overrides: dict[str, object] | None = None,
    ddr_scale: int = 1,
) -> TbCase:
    """`build_case` for a `Model` that already exists.

    `build_case` owns the "fresh seeded model, then a builder function"
    contract every hand-written case uses. A *tiled* case has no such
    builder: `tiler.tile` produces the model to run as a whole-graph
    rewrite of another one, and `tiling_select.select` has to see that
    other one first. Everything downstream -- planning, DDR layout,
    descriptor emission, the export window -- is identical, so it lives
    here and `build_case` calls it."""
    # `cnn_accel_tensor_mem` requires a power-of-two `g_bank_words` (it
    # decodes bank/offset as bit slices of the word address) and asserts
    # it at elaboration. Caught here instead, because otherwise a case
    # that gets it wrong compiles, plans, emits and only dies inside GHDL
    # minutes later with an assertion that names the generic but not the
    # case -- which is exactly how it was found.
    if bank_words & (bank_words - 1) != 0:
        raise ValueError(
            f"case '{name}': bank_words must be a power of two "
            f"(cnn_accel_tensor_mem asserts it), got {bank_words}"
        )

    tensor_mem_bytes = num_banks * bank_words * WORD_BYTES
    # `bank_words` is passed on, not just multiplied in: the scratchpad is
    # `num_banks` INDEPENDENT banks, and `cnn_accel_tensor_mem` clamps any
    # transfer that would run past the end of the bank its address decodes
    # to, so the planner has to know where those boundaries are or it will
    # happily place a buffer across one and have the hardware silently
    # truncate every access to it.
    planned = Planner(
        tensor_mem_bytes=tensor_mem_bytes,
        bank_bytes=bank_words * WORD_BYTES,
        ddr_map=DdrMap(scale=ddr_scale),
    ).plan(model)
    program = emit_program(planned)

    export_base, export_bytes = _export_window(model, planned)

    return TbCase(
        name=name,
        model=model,
        planned=planned,
        program=program,
        export_base=export_base,
        export_bytes=export_bytes,
        num_banks=num_banks,
        bank_words=bank_words,
        expect_error=expect_error,
        expect_err_code=expect_err_code,
        expect_err_pc=expect_err_pc,
        traffic=traffic if traffic is not None else TrafficPolicy(),
        generic_overrides=dict(generic_overrides or {}),
    )


def _export_window(model: Model, planned: PlannedProgram) -> tuple[int, int]:
    """Smallest word-aligned byte window covering every graph output.
    `(0, 0)` when the model has none (error-injection cases)."""
    if not model.outputs:
        return 0, 0
    lo = min(planned.tensor_ddr_addr[t.name] for t in model.outputs)
    hi = max(planned.tensor_ddr_addr[t.name] + t.size_bytes for t in model.outputs)
    lo -= lo % WORD_BYTES
    hi += (-hi) % WORD_BYTES
    return lo, hi - lo


__all__ = [
    "CheckFailure",
    "TbCase",
    "TrafficPolicy",
    "build_case",
    "case_from_model",
]
