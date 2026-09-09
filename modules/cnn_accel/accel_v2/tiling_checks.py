"""Independent geometry assertions over a planned tiled program.

The standing blind spot of this project (memory note `cnn-accel-
verification-blind-spot`) applies to tiling with full force: the DUT and
`reference.py` both execute the addresses `planner.py` chose, so a wrong
row range or a wrong plane offset is *self-consistently* wrong on both
sides and a value-only comparison agrees with it. The only thing that
catches such a bug is a statement about the geometry made without
consulting the planner's reasoning.

This module is the reusable half of that: checks that read a
`PlannedProgram`'s emitted steps and verify a property the tiling must
have, stated in terms of `(plane, row)` cells and bank boundaries rather
than in terms of what the tiler intended. The *row-range recurrence*
itself is deliberately NOT here -- each test re-derives the expected
strip rows from the closed form in the design document, because a shared
implementation would just be the tiler agreeing with itself.

Used by `tests/test_row_copy.py` and `tests/test_tiler.py` now, and by
the fused `TbCase._check`s that step 7 adds (arch doc acceptance 4: a
case is not "fused" unless its check proves the row ranges from the
formula, not from the plan).
"""

from __future__ import annotations

from accel_v2 import isa
from accel_v2.model import PLANE_CHANNELS, Tensor, alias_byte_offset, alias_root
from accel_v2.planner import PlannedProgram, RowCopyStep


class GeometryError(AssertionError):
    """A tiled program's addresses do not describe the tensors it claims
    to be moving. Always a compiler bug, never a legal program."""


def row_copy_cells(planned: PlannedProgram, tensor: Tensor) -> list[tuple[int, int]]:
    """Every `(plane, row)` cell of `tensor` written by a `RowCopyStep`
    of `planned`, recovered **from the emitted addresses**, not from the
    ops' `dst_rows`.

    That distinction is the entire value of this function. `dst_rows` is
    what the tiler meant; the address is what the hardware will do. They
    are computed at different times by different code, and the whole
    class of bug this guards against is the two disagreeing."""
    root = alias_root(tensor)
    base = planned.tensor_ddr_addr.get(root.name)
    if base is None:
        raise GeometryError(
            f"'{tensor.name}' (buffer '{root.name}') has no DDR address in this plan, so "
            "there is nothing to check the strip stores against -- a group boundary must "
            "be pinned"
        )
    origin = base + alias_byte_offset(tensor)
    plane_bytes = tensor.plane_bytes
    row_bytes = tensor.row_bytes

    cells: list[tuple[int, int]] = []
    for step in planned.steps:
        if not isinstance(step, RowCopyStep) or step.dst_space != isa.SPACE_DDR:
            continue
        for _src, dst, nbytes in step.transfers:
            offset = dst - origin
            if not 0 <= offset < tensor.size_bytes:
                continue
            plane, within = divmod(offset, plane_bytes)
            if within % row_bytes:
                raise GeometryError(
                    f"a row copy writes '{tensor.name}' at byte {offset} into its buffer, "
                    f"which is {within % row_bytes} bytes into a {row_bytes}-byte plane "
                    "row. A row copy must move whole plane-rows -- this one would split a "
                    "row, and on the DUT the tail of it would be whatever the previous "
                    "tenant left there."
                )
            if nbytes % row_bytes:
                raise GeometryError(
                    f"a row copy moves {nbytes} bytes into '{tensor.name}', which is not a "
                    f"whole number of its {row_bytes}-byte plane rows"
                )
            first_row = within // row_bytes
            cells.extend((plane, first_row + i) for i in range(nbytes // row_bytes))
    return cells


def check_row_copy_partition(planned: PlannedProgram, tensor: Tensor) -> None:
    """R11. The strip stores into `tensor` must cover every `(plane, row)`
    cell of it **exactly once**.

    Two failure modes, both silent: a gap leaves stale bytes that the
    next group happily reads (and the reference reads the same stale
    bytes, so the values agree and are both wrong), and an overlap means
    two strips wrote the same rows -- harmless when they agree, a
    race-shaped bug when the halo arithmetic made them disagree. A
    partition is the exact property, so it is the one asserted."""
    cells = row_copy_cells(planned, tensor)
    expected = {
        (plane, row)
        for plane in range(tensor.plane_count)
        for row in range(tensor.height)
    }
    seen = set()
    duplicates = set()
    for cell in cells:
        (duplicates if cell in seen else seen).add(cell)

    if duplicates:
        raise GeometryError(
            f"'{tensor.name}' has {len(duplicates)} (plane, row) cell(s) written by more "
            f"than one strip store, e.g. {sorted(duplicates)[:5]} -- two strips claim the "
            "same rows, so at least one halo boundary is wrong"
        )
    missing = expected - seen
    if missing:
        raise GeometryError(
            f"'{tensor.name}' has {len(missing)} (plane, row) cell(s) that no strip store "
            f"ever writes, e.g. {sorted(missing)[:5]} -- the strips do not cover the "
            "tensor, and the gap will read back as stale memory on the DUT and as stale "
            "memory in reference.py, which is why the values would still agree"
        )
    stray = seen - expected
    if stray:  # pragma: no cover - would mean row_copy_cells' own bounds check missed one
        raise GeometryError(f"'{tensor.name}': strip stores land outside the tensor: {sorted(stray)[:5]}")


def check_units_confined(planned: PlannedProgram) -> None:
    """R5. Re-derive, from `local_placements` alone, that every request
    the hardware will make against every local buffer lies inside one
    bank.

    `cnn_accel_tensor_mem` clamps a straddling request instead of
    refusing it, so the tail of such a transfer is stale RAM and nothing
    in the toolchain notices: `reference.py` models the scratchpad as one
    flat `bytearray` and cannot see banks at all. This is therefore the
    only check of the relaxed confinement rule that exists on the pytest
    side."""
    units = {
        alias_root(t).name: (
            t.confine_unit_bytes if t.confine_unit_bytes is not None else t.size_bytes
        )
        for t in planned.model.tensors
        if t.confine_unit_bytes is not None
    }
    # A resident weight image is not a graph tensor at all -- it is a
    # planner-owned buffer of packed constants -- so its request size
    # comes from the plan's own side table rather than from a `Tensor`.
    units.update(planned.local_confine_units)
    bank = planned.bank_bytes
    for name, addr, size in planned.local_placements:
        unit = units.get(name, size)
        offset = 0
        while offset < size:
            lo = addr + offset
            hi = addr + min(offset + unit, size)
            if lo // bank != (hi - 1) // bank:
                raise GeometryError(
                    f"buffer '{name}' at {addr} (+{size} bytes, requests of {unit}) has a "
                    f"request [{lo}, {hi}) crossing a {bank}-byte bank boundary. The "
                    "hardware would clamp it and silently truncate the transfer."
                )
            offset += unit


def check_no_unpinned_ddr_placements(planned: PlannedProgram) -> None:
    """A fused group must be fallback-free: the only buffers in DDR are
    the ones the tiling deliberately pinned there (section 7 level 1).
    Anything in `ddr_placements` is level 2 -- a buffer that did not fit
    -- and means the chosen strip height was not actually viable."""
    if planned.ddr_placements:
        raise GeometryError(
            "the plan fell back to DDR for buffer(s) "
            f"{[name for name, _, _ in planned.ddr_placements]}, which are not pinned "
            "group boundaries. A fused strip sub-program is required to be fallback-free; "
            "this strip height does not fit."
        )


__all__ = [
    "GeometryError",
    "row_copy_cells",
    "check_row_copy_partition",
    "check_units_confined",
    "check_no_unpinned_ddr_placements",
]
