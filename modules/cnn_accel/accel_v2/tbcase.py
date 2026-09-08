"""The Python half of `test/tb_cnn_accel_top.vhd`.

`tb_cnn_accel_top` is deliberately dumb: it loads a DDR image from CSV,
pokes `PROGRAM_BASE_ADDR` + `CTRL.START`, waits for `DONE`/`ERROR`,
dumps the CSR counters and exports a byte region back to CSV. Every
decision about *what* to run and *whether the result is right* lives
here, so adding a test adds a Python function and never VHDL (arch doc
section 11).

A `TbCase` bundles the four things one VUnit config needs:

* the `Model` (what to compute) and its `PlannedProgram` (where every
  tensor lives, and the predicted DDR traffic),
* the `ProgramImage` (descriptor chain + weights + seeded inputs) that
  becomes `mem_image.csv`,
* the VUnit generics that tell the testbench the program entry point,
  the export window and the scratchpad geometry,
* `pre_config` / `post_check` hooks.

The verification `post_check` performs is deliberately in two layers:

1. **Data**: every graph output is read back out of `result.csv` -- i.e.
   out of the bytes the DUT itself wrote to the memory model over AXI --
   unpacked from the hardware's plane layout and compared element by
   element against `reference.run_reference`. Nothing is compared against
   a value the DUT reported about itself.
2. **Traffic**: the DUT's own CSR counters are compared against
   `Planner`'s prediction *and* against the testbench's passive AXI
   monitor (`axi_*` in `counters.csv`). The residency invariants of arch
   doc section 10 are the whole point of rev 2, and a counter that only
   agrees with itself proves nothing -- `DDR_WR_BYTES` must agree with
   independently observed W-channel handshakes, or the "the intermediate
   never went to DDR" claim rests on the same logic it is testing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cnn_accel_model as golden
from accel_v2 import isa
from accel_v2.ddrmap import DdrMap
from accel_v2.memimage import MemoryImage
from accel_v2.model import Model, Tensor
from accel_v2.planner import ComputeStep, MoveStep, PlannedProgram, Planner
from accel_v2.program import ProgramImage, emit_program
from accel_v2.reference import ExecutionResult, run_reference

#: File names the testbench and this module agree on, all directly
#: inside VUnit's per-config `output_path`.
MEM_IMAGE_CSV = "mem_image.csv"
RESULT_CSV = "result.csv"
COUNTERS_CSV = "counters.csv"

#: Bytes per `MemoryImage`/AXI word. Same constant the testbench derives
#: from the generated AXI data width.
WORD_BYTES = 8


class CheckFailure(Exception):
    """Raised (and caught) inside `TbCase.post_check` so that every
    failure is reported through one formatter."""


# ---------------------------------------------------------------------------
# counters.csv
# ---------------------------------------------------------------------------

#: Every key `tb_cnn_accel_top` writes to `counters.csv`. Listed here so
#: a testbench/harness disagreement fails loudly with a diff of names
#: rather than a `KeyError` deep inside a check.
COUNTER_KEYS = (
    # CSR, decoded
    "status",
    "busy",
    "done",
    "error",
    "err_code",
    "err_pc_low",
    "hw_info",
    "hw_info2",
    "cmd_count",
    "cycle_count",
    "compute_cycles",
    "stall_cycles",
    "ddr_rd_bytes",
    "ddr_wr_bytes",
    "tensor_load_count",
    "tensor_store_count",
    "weight_load_bytes",
    "local_bytes",
    # Testbench's own passive AXI monitor -- independent of anything the
    # DUT says about itself.
    "axi_ar_count",
    "axi_aw_count",
    "axi_rd_beats",
    "axi_wr_beats",
    "axi_rd_bytes",
    "axi_wr_bytes",
    "axi_wr_lo_addr",
    "axi_wr_hi_addr",
)


def read_counters(path: str) -> dict[str, int]:
    """Parse a `counters.csv` written by `tb_cnn_accel_top`."""
    with open(path) as handle:
        lines = [line.strip() for line in handle]

    values: dict[str, int] = {}
    seen_header = False
    for lineno, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        if not seen_header:
            if line != "name,value":
                raise CheckFailure(f"{path}:{lineno}: expected header 'name,value', got {line!r}")
            seen_header = True
            continue
        name, _, raw = line.partition(",")
        if not _:
            raise CheckFailure(f"{path}:{lineno}: malformed record {line!r}")
        values[name] = int(raw, 10)

    if not seen_header:
        raise CheckFailure(f"{path}: no 'name,value' header line -- did the testbench die early?")

    missing = [key for key in COUNTER_KEYS if key not in values]
    if missing:
        raise CheckFailure(f"{path}: testbench did not write counter(s) {missing}")
    return values


# ---------------------------------------------------------------------------
# Traffic-assertion policy
# ---------------------------------------------------------------------------


@dataclass
class TrafficPolicy:
    """Which traffic claims this case asserts.

    `write_bytes` is exact by default and is the decisive residency
    check: for a local chain it must equal exactly the closing `STORE`'s
    size (invariants R1/R2/R4).

    `read_bytes` is a *lower* bound by default. The prediction counts the
    logical bytes each operand needs, while the hardware fetches whole
    AXI beats and may re-fetch a weight image per output-channel tile, so
    an exact equality here would encode burst behaviour that is not part
    of the residency contract. Tests that do care about an exact read
    figure set `read_bytes_exact=True` themselves.
    """

    write_bytes_exact: bool = True
    read_bytes_exact: bool = False
    read_bytes_at_least: bool = True
    counts_exact: bool = True
    weight_bytes_exact: bool = True
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
    traffic: TrafficPolicy = field(default_factory=TrafficPolicy)
    #: Extra generic overrides merged last into `generics()`.
    generic_overrides: dict[str, object] = field(default_factory=dict)
    _expected: ExecutionResult | None = field(default=None, repr=False)

    # -- VUnit plumbing ---------------------------------------------------

    def generics(self) -> dict[str, object]:
        """The generics for `add_vunit_config`. `output_path` is filled in
        by VUnit itself and must not appear here."""
        generics: dict[str, object] = {
            "g_ddr_bytes": DdrMap.LIMIT,
            "g_program_base": self.program.program_addr,
            "g_export_base": self.export_base,
            "g_export_bytes": self.export_bytes,
            "g_expect_error": self.expect_error,
            "g_num_banks": self.num_banks,
            "g_bank_words": self.bank_words,
            "g_ddr_limit": DdrMap.LIMIT,
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

    def pre_config(self, output_path: str) -> bool:
        """Write `mem_image.csv` into VUnit's per-config `output_path`.
        Nothing is checked into the repository and the testbench reads
        nothing else."""
        os.makedirs(output_path, exist_ok=True)
        self.program.image.write_csv(
            os.path.join(output_path, MEM_IMAGE_CSV),
            comment_lines=(
                f"case: {self.name}",
                f"model: {self.model.name} seed={self.model.seed}",
                f"program_base: 0x{self.program.program_addr:08x} "
                f"({len(self.program.descs)} descriptors incl. HALT)",
                f"tensor_mem: {self.num_banks} banks x {self.bank_words} words "
                f"= {self.tensor_mem_bytes} bytes",
            ),
        )
        return True

    def post_check(self, output_path: str) -> bool:
        """Verify the run. Returns False (after printing a diagnosable
        report) rather than raising, which is what VUnit wants from a
        `post_check` hook."""
        try:
            self._post_check(output_path)
        except CheckFailure as exc:
            print(f"\npost_check FAILED for case '{self.name}':\n{exc}\n")
            return False
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
        the DUT's preload would seed the DUT with its own expected
        answers -- the most embarrassing possible backdoor."""
        if self._expected is None:
            self._expected = run_reference(self.planned, MemoryImage())
        return self._expected

    # -- checking ---------------------------------------------------------

    def _post_check(self, output_path: str) -> None:
        counters = read_counters(os.path.join(output_path, COUNTERS_CSV))
        self._check_status(counters)
        if self.expect_error:
            # A rejected program has no meaningful output tensors and no
            # meaningful traffic prediction: the whole point is that the
            # DUT stopped. The one traffic claim that still holds is the
            # arch doc's "never partially writes a validated-bad
            # command's destination" promise, checked below.
            self._check_error_wrote_nothing_unexpected(counters)
            return
        self._check_outputs(output_path)
        self._check_traffic(counters)

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

        # One descriptor retired per planned step, plus the closing HALT.
        expected_cmds = len(self.planned.steps) + 1
        if counters["cmd_count"] != expected_cmds:
            raise CheckFailure(
                f"CMD_COUNT={counters['cmd_count']}, expected {expected_cmds} "
                f"({len(self.planned.steps)} steps + HALT)\n{self._program_listing()}"
            )

    def _check_outputs(self, output_path: str) -> None:
        result_path = os.path.join(output_path, RESULT_CSV)
        exported = MemoryImage.read_csv(result_path)

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
            self._compare_tensor(tensor, expected, actual, addr, result_path)

    def _compare_tensor(
        self,
        tensor: Tensor,
        expected: list[int],
        actual: list[int],
        addr: int,
        result_path: str,
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
                f"exported to {result_path}\n"
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
        lines = [f"  planned program ({len(self.planned.steps)} steps + HALT):"]
        for index, step in enumerate(self.planned.steps):
            addr = self.program.program_addr + index * isa.INSTR_WORD_BYTES
            if isinstance(step, MoveStep):
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
            f"0x{self.program.program_addr + len(self.planned.steps) * isa.INSTR_WORD_BYTES:08x} HALT"
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
    traffic: TrafficPolicy | None = None,
    generic_overrides: dict[str, object] | None = None,
) -> TbCase:
    """Build one `TbCase`. `build(model)` adds the case's tensors and ops
    to a freshly seeded `Model` (and marks its graph outputs); everything
    downstream -- planning, DDR layout, descriptor emission -- follows
    from that.

    `seed` is explicit and mandatory: every randomized value in the case
    (inputs, weights, biases, per-channel scale tables) comes from it, so
    a failing case is reproducible from its name alone.
    """
    model = Model(seed=seed, name=name)
    build(model)

    tensor_mem_bytes = num_banks * bank_words * WORD_BYTES
    planned = Planner(tensor_mem_bytes=tensor_mem_bytes).plan(model)
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
    "COUNTERS_CSV",
    "COUNTER_KEYS",
    "CheckFailure",
    "MEM_IMAGE_CSV",
    "RESULT_CSV",
    "TbCase",
    "TrafficPolicy",
    "build_case",
    "read_counters",
]
