"""`tb_cnn_accel_top` cases for the ISA v2.1 error model
(`cnn_accel_v2_pkg.vhd`'s `c_err_*` codes, doc/cnn_accel_top_v2_arch.md
section 9). A third catalogue alongside `cases.py`/`cases_pool_pad.py`,
registered by `module_cnn_accel.py` in the same loop and following the
same one-function-per-case contract.

Before this file, `tbcase.py` had `expect_error`/`expect_err_code` and
`_check_error_wrote_nothing_unexpected` but **no case ever set them**, so
the DUT's entire error path -- nine distinct `c_err_*` codes -- had zero
coverage. Each case here starts from the smallest well-formed program
(`case_single_conv`'s shape) and mutates exactly one field of its first
descriptor to provoke exactly one error condition, then patches the
mutated bytes straight into the already-emitted `ProgramImage` (bypassing
`Model`/`Planner`/`emit_program`, which only ever produce well-formed
programs by construction). This keeps every case's blast radius to "one
malformed descriptor" and its failure mode to "the DUT must reject this
specific thing, at the very first instruction, before doing any work" --
which is also why every case runs with a tight, directed watchdog
(`g_watchdog_cycles`/`g_timeout_cycles` overrides) instead of the
100k-cycle default: a case that *doesn't* fail fast is itself a bug.

Reading `cnn_accel_cmd_proc.vhd`'s `st_validate`/`st_range` FSM states is
what makes each mutation unambiguous: validation runs in a fixed order
(unsupported_op -> bad_reserved -> bad_space -> misaligned -> bad_geometry
-> [later] local_range/ddr_range -> more bad_geometry), so changing only
one field downstream of every earlier check guarantees that earlier
check still passes and this case's target code is the one that fires --
never a different code winning on a technicality.

Every case asserts three things end to end, not just "some error
happened":

1. `expect_err_code` -- the RIGHT code, not merely STATUS.ERROR=1.
2. `expect_err_pc` -- `STATUS.ERR_PC_LOW` points at the mutated
   descriptor (always the program's first instruction here, i.e.
   `program.program_addr`).
3. `traffic=TrafficPolicy(allowed_write_ranges=[])` -- the program wrote
   nothing at all (`_check_error_wrote_nothing_unexpected`): a
   validated-bad command must be dropped before touching DDR.

Not every `c_err_*` code is reachable this way -- see the two functions
at the bottom of this file for the ones this catalogue deliberately does
NOT attempt, and why.
"""

from __future__ import annotations

import dataclasses

from accel_v2 import isa
from accel_v2.ddrmap import DdrMap
from accel_v2.model import Model
from accel_v2.tbcase import TbCase, TrafficPolicy, build_case

# Same tiny shape as cases.py's case_single_conv: these cases never run
# their DUT program to completion (they are rejected at the first
# descriptor), so the shape only has to be *well-formed*, not exercised.
_H = 8
_W = 8
_C = 8

# These cases must fail fast: a malformed first descriptor is rejected
# in the validate/range FSM stages, long before any engine or watchdog
# would matter. Small, directed bounds (instead of the 100k/50k
# integration-case defaults) mean a case that regresses into *not*
# failing fast burns seconds, not minutes, of simulation time.
_FAST_WATCHDOG_CYCLES = 2_000
_FAST_TIMEOUT_CYCLES = 4_000


def _build_single_conv(model: Model) -> None:
    x = model.input(_H, _W, _C, name="x")
    y = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="y")
    model.output(y)


def _build_single_dts(model: Model) -> None:
    """The `DEPTH_TO_SPACE` equivalent of `_build_single_conv`: one
    well-formed pixel-shuffle, DDR in -> DDR out, as the program's first
    and only real instruction -- so `_error_case` can mutate exactly one
    of ITS descriptor fields. A conv-fed graph would put the conv at
    `descs[0]` and mutate the wrong instruction."""
    x = model.input(_H, _W, 4 * _C, name="x")
    y = model.depth_to_space(x, name="y")
    model.output(y)


def _error_case(
    name: str,
    seed: int,
    err_code: int,
    overrides: dict[str, object] | "object" = None,
    build=_build_single_conv,
) -> TbCase:
    """One malformed-first-descriptor case: build the smallest well-formed
    program (`_build_single_conv`), then splice `overrides` into its
    first (and, but for the closing HALT, only) descriptor and rewrite
    just that descriptor's bytes into the already-emitted `ProgramImage`.

    `overrides` is either a plain `dict` or a `desc -> dict` callable (for
    the rare case that needs to compute its override relative to the
    known-good field, e.g. "one byte past an aligned address"; a callable
    avoids building the case twice just to read that value).

    `dataclasses.replace` keeps every field this case does not name at
    its known-good value from the working conv2d descriptor, which is
    exactly what isolates "this one field is wrong" from "this program
    is wrong in several ways at once" -- the latter would make it
    ambiguous which check actually fired.
    """
    case = build_case(
        name,
        build,
        seed=seed,
        expect_error=True,
        expect_err_code=err_code,
        traffic=TrafficPolicy(allowed_write_ranges=[]),
        generic_overrides={
            "g_watchdog_cycles": _FAST_WATCHDOG_CYCLES,
            "g_timeout_cycles": _FAST_TIMEOUT_CYCLES,
        },
    )
    good = case.program.descs[0]
    field_overrides = overrides(good) if callable(overrides) else dict(overrides or {})
    mutated = dataclasses.replace(good, **field_overrides)
    case.program.image.write_bytes(case.program.program_addr, isa.encode_desc(mutated))
    case.program.descs[0] = mutated
    # The mutated descriptor is the program's first instruction, so a
    # correctly-behaving DUT latches ERR_PC_LOW at program_addr.
    case.expect_err_pc = case.program.program_addr
    return case


# ---------------------------------------------------------------------------
# Reachable codes.
# ---------------------------------------------------------------------------


def case_err_unsupported_op() -> TbCase:
    """`ERR_UNSUPPORTED_OP` (0x1): an opcode value with no allocation at
    all (section 5.2's table has gaps between FC=5 and LOAD=16, and above
    ACT=22). `cnn_accel_cmd_proc.classify` falls through to `cls_bad` for
    anything it does not name, which is the first check `st_validate`
    makes -- so this fires no matter what the rest of the descriptor
    says."""
    return _error_case("err_unsupported_op", seed=101, err_code=isa.ERR_UNSUPPORTED_OP, overrides={"opcode": 0x0F})


def case_err_unsupported_op_dwconv2d() -> TbCase:
    """`ERR_UNSUPPORTED_OP` via `DWCONV2D` (opcode 0x02) specifically:
    section 5.2 calls this one out as "allocated, **rejected**", a
    distinct case from a plain gap in the opcode space -- `classify`
    falls through to `cls_bad` for it exactly the same way, but a
    regression that accidentally started accepting DWCONV2D (e.g. by
    aliasing it to CONV2D) would not be caught by
    `case_err_unsupported_op` alone."""
    return _error_case(
        "err_unsupported_op_dwconv2d",
        seed=102,
        err_code=isa.ERR_UNSUPPORTED_OP,
        overrides={"opcode": isa.OPCODE_DWCONV2D},
    )


def case_err_bad_space() -> TbCase:
    """`ERR_BAD_SPACE` (0x2): `space_src0` set to space tag `3`
    (`SPACE_RESERVED`), always illegal regardless of opcode (section 3).
    Checked after `unsupported_op`/`bad_reserved` and before alignment/
    geometry, so leaving every other field at its known-good conv2d value
    isolates this one."""
    return _error_case(
        "err_bad_space", seed=103, err_code=isa.ERR_BAD_SPACE, overrides={"space_src0": isa.SPACE_RESERVED}
    )


def case_err_misaligned() -> TbCase:
    """`ERR_MISALIGNED` (0x3): `in_addr` shifted by one byte off its
    (correct, 8-byte-aligned) value, so every operand space/geometry
    field is still valid and only the alignment check fires."""
    return _error_case(
        "err_misaligned",
        seed=104,
        err_code=isa.ERR_MISALIGNED,
        overrides=lambda good: {"in_addr": good.in_addr + 1},
    )


def case_err_local_range() -> TbCase:
    """`ERR_LOCAL_RANGE` (0x4): destination redirected to `LOCAL_TENSOR`
    (itself a legal space tag, so `bad_space` does not fire) at an
    address exactly equal to the scratchpad's size --
    `default num_banks=2, bank_words=1024` gives `tensor_mem_bytes =
    2*1024*8 = 16384` bytes, and any nonzero write length starting there
    already runs past the end."""
    tensor_mem_bytes = 2 * 1024 * 8
    return _error_case(
        "err_local_range",
        seed=105,
        err_code=isa.ERR_LOCAL_RANGE,
        overrides={"space_dst": isa.SPACE_LOCAL_TENSOR, "out_addr": tensor_mem_bytes},
    )


def case_err_ddr_range() -> TbCase:
    """`ERR_DDR_RANGE` (0x5): `in_addr` set to exactly `g_ddr_limit`
    (`DdrMap.LIMIT`, the same value `TbCase.generics()` passes as
    `g_ddr_limit`) -- 8-byte aligned, a legal DDR address tag, but
    already past the bound before the source's own byte count is even
    added."""
    return _error_case("err_ddr_range", seed=106, err_code=isa.ERR_DDR_RANGE, overrides={"in_addr": DdrMap.LIMIT})


def case_err_bad_reserved() -> TbCase:
    """`ERR_BAD_RESERVED` (0x6): `reserved_w0` (W0 byte 3) set nonzero.
    `DescV2.reserved_w0`/`reserved_w10` exist in the ISA model for
    exactly this purpose (see their docstring in `accel_v2/isa.py`) and
    were, before this case, never actually used by anything."""
    return _error_case("err_bad_reserved", seed=107, err_code=isa.ERR_BAD_RESERVED, overrides={"reserved_w0": 1})


def case_err_bad_geometry() -> TbCase:
    """`ERR_BAD_GEOMETRY` (0x7): `kernel_h` zeroed. `st_validate`'s
    geometry check for `cls_conv` rejects a zero kernel dimension before
    any address range is even considered, so this isolates cleanly from
    `local_range`/`ddr_range` even though both live in the same
    descriptor class."""
    return _error_case("err_bad_geometry", seed=108, err_code=isa.ERR_BAD_GEOMETRY, overrides={"kernel_h": 0})


def case_err_dts_bad_factor() -> TbCase:
    """`ERR_BAD_GEOMETRY` (0x7) from `DEPTH_TO_SPACE`'s own contract:
    `dts_factor = 3`, which the ISA field can encode but v1 hardware does
    not implement (`cnn_accel_cmd_proc.chk_dts_geom_q`).

    This is the FIRST case to reach a geometry rejection through the
    `cls_elem` path -- `case_err_bad_geometry` above goes through
    `cls_conv` -- and the point of it is that an unimplemented factor is
    refused outright rather than silently executed as factor 2, which
    would write a plausible-looking wrong tensor."""
    return _error_case(
        "err_dts_bad_factor",
        seed=109,
        err_code=isa.ERR_BAD_GEOMETRY,
        overrides={"dts_factor": 3},
        build=_build_single_dts,
    )


def case_err_dts_bad_channels() -> TbCase:
    """`ERR_BAD_GEOMETRY` (0x7): `out_channels` mutated to 9, breaking
    `in_channels == factor**2 * out_channels` (32 != 36).

    `out_channels` is mutated rather than `in_channels` deliberately: it
    is the one field of this descriptor that no length or address
    derives from, so the ONLY check that can fire is the geometry one --
    mutating `in_channels` would also change `in_total_bytes` and leave
    it ambiguous whether the DUT rejected the shape or the range."""
    return _error_case(
        "err_dts_bad_channels",
        seed=110,
        err_code=isa.ERR_BAD_GEOMETRY,
        overrides={"out_channels": 9},
        build=_build_single_dts,
    )


# ---------------------------------------------------------------------------
# Not attempted, deliberately, and why.
# ---------------------------------------------------------------------------

# `ERR_AXI` (0x8): "AXI RRESP/BRESP not OKAY" (section 9). The
# testbench's only path to memory is VUnit's `bfm.axi_slave`
# (`c_axi_read_slave`/`c_axi_write_slave` in tb_cnn_accel_top.vhd), which
# has no supported way to make a *specific* transaction come back
# SLVERR/DECERR while the rest of the program still runs normally -- an
# out-of-permission access on that BFM fails the simulation with a VUnit
# check error instead of driving a bus-level error response, which is a
# testbench bug, not the DUT behaviour this code is supposed to cover.
# Reaching this for real needs a fault-injecting AXI slave BFM (or a
# custom one) standing in for the memory model, which is a real feature,
# not a directed-case addition -- left for future work rather than
# contrived here.
#
# `ERR_TIMEOUT` (0x9): "engine failed to complete within
# g_watchdog_cycles" -- by construction, the only way to hit this is a
# genuinely stuck engine, which is not something any ISA-level program
# can provoke (it would require a hardware bug in the engine being
# tested, or a `g_watchdog_cycles` generic set unrealistically low
# against a real workload -- fragile either way, and not the kind of
# thing a fast, deterministic case should depend on to pass).


ALL_CASES = (
    case_err_unsupported_op,
    case_err_unsupported_op_dwconv2d,
    case_err_bad_space,
    case_err_misaligned,
    case_err_local_range,
    case_err_ddr_range,
    case_err_bad_reserved,
    case_err_bad_geometry,
    case_err_dts_bad_factor,
    case_err_dts_bad_channels,
)


def all_cases() -> list[TbCase]:
    return [case() for case in ALL_CASES]


__all__ = ["all_cases"]
