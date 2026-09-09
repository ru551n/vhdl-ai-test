"""The tiled `tb_cnn_accel_top` catalogue, without the simulator.

Two jobs, neither of which needs GHDL:

* **Build every case and check what it claims.** A `TbCase` is a planned
  program plus a set of assertions; the assertions only run inside
  `post_check`, i.e. only after a simulation. These tests build the
  catalogue and check the properties that make each case worth
  simulating -- it is fused, it does not fall back, its strips really are
  more than one -- so that a case which quietly stopped testing its own
  subject fails here, in a second, rather than passing in the regression.

* **Mutation-test the geometry checks themselves.** `tiling_checks`'
  assertions are the only thing standing between a wrong address and a
  green run: the DUT and `reference.py` both execute the planner's
  addresses, so a placement bug agrees with itself. A check that cannot
  fail is worth nothing, so each one is shown failing on a plan that has
  been deliberately broken in exactly the way it exists to catch.
"""

from __future__ import annotations

import functools

import pytest

from accel_v2 import cases_tiling, isa
from accel_v2.planner import ComputeStep, ConstLoadStep, RowCopyStep
from accel_v2.tiling_checks import (
    GeometryError,
    check_no_unpinned_ddr_placements,
    check_row_copy_partition,
    check_units_confined,
)


@functools.lru_cache(maxsize=None)
def _case(name: str):
    builder = next(b for b in cases_tiling.CASE_BUILDERS if b.__name__.endswith(name))
    return builder()


#: Cases that are deliberately NOT tiled, and what they are for.
_UNTILED = {
    "case_ot_restream_from_ddr": "isolates the output-channel re-stream",
    "case_unfused_c2f_64ch": "the 'before' of the headline comparison",
}


def test_every_case_builds_and_plans_without_falling_back() -> None:
    """No *tiled* case in this catalogue may leave a buffer in DDR by
    accident. A pinned group boundary is deliberate and is not in
    `ddr_placements`; anything that is means the strip height the rule
    chose does not actually fit, and every traffic figure the case
    asserts is then about a different program.

    The two untiled cases are exempt from the fallback check by
    construction -- `unfused_c2f_64ch` exists *because* its graph does
    not fit -- but their confinement is checked like everything else."""
    for builder in cases_tiling.CASE_BUILDERS:
        case = builder()
        if builder.__name__ not in _UNTILED:
            check_no_unpinned_ddr_placements(case.planned)
        check_units_confined(case.planned)
        assert case.export_bytes > 0, case.name


def test_the_tiled_cases_are_actually_tiled() -> None:
    """Every case built through the tiler must contain a row copy, and
    the ones whose subject is a fused *group* must contain more than one
    strip's worth of them. Guards against a shape drifting until the rule
    picks `S = 1` and the case silently becomes an untiled one."""
    tiled_cases = [b for b in cases_tiling.CASE_BUILDERS if b.__name__ not in _UNTILED]
    for builder in tiled_cases:
        case = builder()
        copies = [s for s in case.planned.steps if isinstance(s, RowCopyStep)]
        assert copies, f"{case.name} contains no row copy, so nothing was tiled"
        loads = [s for s in copies if s.src_space == isa.SPACE_DDR]
        assert len(loads) >= 2, (
            f"{case.name} loads a strip only once, so it was cut into a single strip"
        )


def test_the_resident_weight_case_serves_its_weights_from_the_scratchpad() -> None:
    case = _case("resident_weights_strip_pair")
    loads = [s for s in case.planned.steps if isinstance(s, ConstLoadStep)]
    assert len(loads) == 1
    convs = [
        s
        for s in case.planned.steps
        if isinstance(s, ComputeStep) and s.weight_space == isa.SPACE_LOCAL_TENSOR
    ]
    assert len(convs) == 2, "both strips must fetch their weights locally"


def test_the_bank_straddling_case_really_straddles() -> None:
    """The case is vacuous unless some buffer spans a bank. Asserted here
    too, not only in its own `post_check`, because a change to the
    allocator that stopped producing the straddle would otherwise make
    the case pass by no longer testing anything."""
    case = _case("bank_straddling_strip")
    bank = case.planned.bank_bytes
    straddling = [
        name
        for name, addr, size in case.planned.local_placements
        if addr // bank != (addr + size - 1) // bank
    ]
    assert straddling, case.planned.local_placements


# ---------------------------------------------------------------------------
# Mutation: each geometry check, shown failing.
# ---------------------------------------------------------------------------


def _stored_tensor(case):
    """The pinned tensor a case's strip stores assemble."""
    return next(t for t in case.model.outputs)


def test_mutation_a_shifted_strip_store_breaks_the_partition_check() -> None:
    """R11's check, mutated: move one strip store down by a single plane
    row.

    That is the smallest error a halo bug can make, and it is invisible
    to everything else. The DUT would write the shifted rows and read
    them back; `reference.py`, executing the same addresses, would write
    and read exactly the same shifted rows; the two would agree, and the
    output would differ from the untiled answer only where the gap and
    the overlap happened to fall. `check_row_copy_partition` sees it as
    what it is -- one cell written twice and one never written."""
    case = _case("fused_c2f_64ch")
    tensor = _stored_tensor(case)
    check_row_copy_partition(case.planned, tensor)  # sound before the mutation

    store = next(
        s
        for s in case.planned.steps
        if isinstance(s, RowCopyStep) and s.dst_space == isa.SPACE_DDR
    )
    original = list(store.transfers)
    row_bytes = tensor.row_bytes
    store.transfers = [(src, dst + row_bytes, n) for src, dst, n in original]
    try:
        with pytest.raises(GeometryError) as excinfo:
            check_row_copy_partition(case.planned, tensor)
        assert "written by more than one strip store" in str(
            excinfo.value
        ) or "no strip store" in str(excinfo.value)
    finally:
        store.transfers = original
    check_row_copy_partition(case.planned, tensor)


def test_mutation_a_split_plane_row_breaks_the_partition_check() -> None:
    """The other way a row copy can be wrong: a destination address that
    is not a whole number of plane rows into the buffer. A row copy moves
    whole plane-rows; anything else leaves the tail of a row as whatever
    the previous tenant wrote."""
    case = _case("detect_branch_80ch")
    tensor = _stored_tensor(case)
    store = next(
        s
        for s in case.planned.steps
        if isinstance(s, RowCopyStep) and s.dst_space == isa.SPACE_DDR
    )
    original = list(store.transfers)
    src, dst, n = original[0]
    store.transfers = [(src, dst + 8, n)] + original[1:]
    try:
        with pytest.raises(GeometryError, match="whole plane-rows"):
            check_row_copy_partition(case.planned, tensor)
    finally:
        store.transfers = original


def test_mutation_a_straddling_plane_breaks_the_confinement_check() -> None:
    """R5's check, mutated: nudge one plane-confined buffer so that a
    plane crosses a bank boundary.

    `cnn_accel_tensor_mem` clamps such a request and the tail of the
    plane reads back as stale RAM -- and `reference.py` models the
    scratchpad as one flat `bytearray`, so it cannot see banks at all and
    would never disagree. This check re-derives the property from
    `local_placements` alone, which is the only place it can be seen."""
    case = _case("bank_straddling_strip")
    check_units_confined(case.planned)

    name, addr, size = next(
        (n, a, s)
        for n, a, s in case.planned.local_placements
        if n in case.planned.local_confine_units
    )
    unit = case.planned.local_confine_units[name]
    bank = case.planned.bank_bytes
    # Put the buffer so that its first plane sits astride a boundary.
    broken = bank - unit // 2
    index = case.planned.local_placements.index((name, addr, size))
    case.planned.local_placements[index] = (name, broken, size)
    try:
        with pytest.raises(GeometryError, match="crossing a .* bank boundary"):
            check_units_confined(case.planned)
    finally:
        case.planned.local_placements[index] = (name, addr, size)
    check_units_confined(case.planned)


def test_mutation_a_ddr_fallback_breaks_the_no_fallback_check() -> None:
    case = _case("fused_wide_128ch")
    check_no_unpinned_ddr_placements(case.planned)
    case.planned.ddr_placements.append(("invented", 0, 1))
    try:
        with pytest.raises(GeometryError, match="fell back to DDR"):
            check_no_unpinned_ddr_placements(case.planned)
    finally:
        case.planned.ddr_placements.pop()
