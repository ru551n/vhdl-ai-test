"""Tests for `RowCopyOp` -- the one new operation spatial tiling needs
(design section 2.2, implementation plan step 4).

It is deliberately not a new opcode: the planner resolves it into one
per-plane sub-transfer each, and `program.py` emits an ordinary
`LOAD`/`STORE`/`COPY` for each of those, picked from the two resolved
operand spaces. So the things worth testing are (a) the arithmetic of the
per-plane addresses, which is where a channel-plane-major layout hides
off-by-one-plane errors, (b) that a strip round trip through a pinned
full-resolution tensor reproduces the untiled bytes exactly, and (c) that
the descriptor count and the traffic charged both follow the plane count
rather than the tensor count.
"""

from __future__ import annotations

import pytest

from accel_v2 import isa
from accel_v2.memimage import MemoryImage
from accel_v2.model import Activation, Model, RowRange
from accel_v2.planner import Planner, RowCopyStep
from accel_v2.program import emit_program
from accel_v2.reference import run_reference
from accel_v2.tiling_checks import GeometryError, check_row_copy_partition

_MEM = dict(tensor_mem_bytes=64 * 1024, bank_bytes=8 * 1024)


def _row_copy_steps(planned) -> list[RowCopyStep]:
    return [s for s in planned.steps if isinstance(s, RowCopyStep)]


# ---------------------------------------------------------------------------
# Per-plane addressing.
# ---------------------------------------------------------------------------


def test_a_row_copy_emits_one_sub_transfer_per_plane_at_the_right_offsets() -> None:
    """The addresses, checked against the layout formula rather than
    against the planner: plane `p`, first row `r` of a tensor of height
    `H` and width `W` is at `p*H*W*T + r*W*T` bytes into its buffer, and
    the two ends have *different* `H`."""
    m = Model(seed=1)
    src = m.input(10, 4, 24, name="src")  # 3 planes, 10 rows
    m.output(m.copy(m.copy_rows(src, RowRange(3, 7), name="win"), name="out"))
    planned = Planner(**_MEM).plan(m)

    (step,) = _row_copy_steps(planned)
    win = next(t for t in m.tensors if t.name == "win")
    assert win.height == 4 and win.channels == 24
    assert len(step.transfers) == 3, "one sub-transfer per activation plane"

    src_base = planned.tensor_ddr_addr["src"]
    dst_base = next(addr for name, addr, _ in planned.local_placements if name == "win")
    t = 8
    for plane, (src_addr, dst_addr, nbytes) in enumerate(step.transfers):
        assert src_addr == src_base + plane * (10 * 4 * t) + 3 * (4 * t)
        assert dst_addr == dst_base + plane * (4 * 4 * t) + 0
        assert nbytes == 4 * 4 * t


def test_the_store_form_writes_at_a_row_offset_of_the_destination() -> None:
    """`copy_rows(..., into=full, at=[o0, o1))`: the destination's plane
    stride is the FULL tensor's height, not the strip's -- the mistake
    that would put strip 1 of plane 1 on top of strip 0 of plane 2."""
    m = Model(seed=2)
    x = m.input(4, 4, 16, name="x")
    full = m.input(12, 4, 16, name="full")  # stands in for a pinned buffer
    full.pin_ddr = True
    m.copy_rows(x, RowRange(0, 4), into=full, at=RowRange(4, 8), name="store")
    planned = Planner(**_MEM).plan(m)

    (step,) = _row_copy_steps(planned)
    full_base = planned.tensor_ddr_addr["full"]
    t = 8
    for plane, (_src, dst_addr, nbytes) in enumerate(step.transfers):
        assert dst_addr == full_base + plane * (12 * 4 * t) + 4 * (4 * t)
        assert nbytes == 4 * 4 * t


def test_a_single_plane_row_window_is_one_contiguous_transfer() -> None:
    m = Model(seed=3)
    src = m.input(8, 4, 8, name="src")
    m.output(m.copy(m.copy_rows(src, RowRange(2, 5), name="win"), name="out"))
    planned = Planner(**_MEM).plan(m)
    (step,) = _row_copy_steps(planned)
    assert len(step.transfers) == 1
    assert step.transfers[0][2] == 3 * 4 * 8


# ---------------------------------------------------------------------------
# Opcode selection and descriptor count.
# ---------------------------------------------------------------------------


def test_descriptor_count_equals_plane_count_and_the_chain_links_through() -> None:
    """`program.py` emits one descriptor per plane, and each chains to
    the next plane of the same copy -- the program is one flat chain, it
    has no notion of a step."""
    m = Model(seed=4)
    src = m.input(8, 4, 32, name="src")  # 4 planes
    m.output(m.copy(m.copy_rows(src, RowRange(1, 5), name="win"), name="out"))
    planned = Planner(**_MEM).plan(m)
    image = emit_program(planned)

    loads = [d for d in image.descs if d.opcode == isa.OPCODE_LOAD]
    assert len(loads) == 4

    addrs = []
    addr = image.program_addr
    seen = set()
    while addr not in seen:
        seen.add(addr)
        desc = isa.decode_desc(image.image.read_bytes(addr, isa.INSTR_WORD_BYTES))
        addrs.append(addr)
        if desc.opcode == isa.OPCODE_HALT:
            break
        addr = desc.next_instr_addr
    assert len(addrs) == len(image.descs), "the chain must visit every emitted descriptor"


@pytest.mark.parametrize(
    "pin_src, pin_dst, expected",
    [
        (True, False, isa.OPCODE_LOAD),  # pinned group input -> strip buffer
        (False, True, isa.OPCODE_STORE),  # strip buffer -> pinned group output
        (False, False, isa.OPCODE_COPY),  # join window, local to local
    ],
)
def test_the_opcode_follows_only_the_two_resolved_spaces(pin_src, pin_dst, expected) -> None:
    m = Model(seed=5)
    x = m.input(8, 4, 8, name="x")
    src = m.conv2d(x, 8, kernel=(1, 1), activation=Activation.NONE, name="src")
    src.pin_ddr = pin_src
    dst = m.copy_rows(src, RowRange(2, 6), name="win")
    dst.pin_ddr = pin_dst
    m.output(m.copy(dst, name="out"))
    planned = Planner(**_MEM).plan(m)
    image = emit_program(planned)
    (step,) = _row_copy_steps(planned)
    opcodes = {
        d.opcode
        for d in image.descs
        if d.xfer_bytes == step.transfers[0][2] and d.in_addr == step.transfers[0][0]
    }
    assert expected in opcodes


def test_traffic_prediction_matches_what_the_reference_actually_moves() -> None:
    """The row copy's bytes are the *window*, not either tensor -- and
    the planner's prediction and the reference's measurement must agree
    on that field by field, as they do for every other step."""
    m = Model(seed=6)
    src = m.input(12, 4, 24, name="src")
    full = m.input(12, 4, 24, name="full")
    full.pin_ddr = True
    win = m.copy_rows(src, RowRange(2, 8), name="win")
    m.copy_rows(win, RowRange(0, 6), into=full, at=RowRange(3, 9), name="store")
    planned = Planner(**_MEM).plan(m)
    actual = run_reference(planned, MemoryImage()).traffic
    assert vars(actual) == vars(planned.traffic)

    window_bytes = 6 * 4 * 8 * 3
    assert planned.traffic.pinned_write_bytes == window_bytes
    assert planned.traffic.tensor_load_count == 3, "3 planes loaded from the pinned source"
    assert planned.traffic.tensor_store_count == 3, "3 planes stored into the pinned output"


def test_a_row_copy_never_hoists_its_full_resolution_source_into_the_scratchpad() -> None:
    """The reason tiling wins at all: reading a window out of a group
    boundary must not drag the whole boundary on chip first, even when
    the buffer has several remaining consumers (which is exactly what a
    group input read by `S` strips looks like)."""
    m = Model(seed=7)
    full = m.input(64, 8, 8, name="full")
    full.pin_ddr = True
    for s in range(4):
        m.output(m.copy(m.copy_rows(full, RowRange(16 * s, 16 * s + 16), name=f"w{s}"), name=f"o{s}"))
    planned = Planner(**_MEM).plan(m)
    assert "full" not in dict((n, a) for n, a, _ in planned.local_placements)
    for step in _row_copy_steps(planned):
        assert step.src_space == isa.SPACE_DDR
    # Each strip read only its own 16 rows: four windows, not four copies
    # of the whole 64-row tensor.
    assert planned.traffic.pinned_read_bytes == 4 * 16 * 8 * 8


# ---------------------------------------------------------------------------
# The round trip: does it actually reproduce the untiled bytes?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [8, 16, 24, 12, 3])
@pytest.mark.parametrize("strips", [1, 2, 3, 5])
def test_a_strip_round_trip_through_a_pinned_tensor_is_the_identity(channels, strips) -> None:
    """Cut a tensor into `strips` row strips, copy each out into its own
    compact buffer and each back into a pinned full-resolution tensor,
    and the result must be the original -- for channel counts that are
    and are not multiples of the 8-channel plane."""
    height = 15
    bounds = [round(i * height / strips) for i in range(strips + 1)]

    m = Model(seed=8)
    src = m.input(height, 4, channels, name="src")
    src.pin_ddr = True
    full = m.input(height, 4, channels, name="full")
    full.pin_ddr = True
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        strip = m.copy_rows(src, RowRange(lo, hi), name=f"strip{i}")
        m.copy_rows(strip, RowRange(0, hi - lo), into=full, at=RowRange(lo, hi), name=f"store{i}")

    planned = Planner(**_MEM).plan(m)
    result = run_reference(planned, MemoryImage())

    assert result.tensor_data["full"] == m.tensors[0].data
    check_row_copy_partition(planned, full)


def test_the_partition_check_catches_a_gap_between_two_strips() -> None:
    """Mutation test for R11's own helper: drop one row from a strip and
    the partition check must notice, even though every value that IS
    written is still correct."""
    m = Model(seed=9)
    src = m.input(8, 4, 16, name="src")
    src.pin_ddr = True
    full = m.input(8, 4, 16, name="full")
    full.pin_ddr = True
    m.copy_rows(m.copy_rows(src, RowRange(0, 4), name="a"), RowRange(0, 4), into=full, at=RowRange(0, 4))
    # Rows 4 and 5 are covered; 6 and 7 are not.
    m.copy_rows(m.copy_rows(src, RowRange(4, 6), name="b"), RowRange(0, 2), into=full, at=RowRange(4, 6))
    planned = Planner(**_MEM).plan(m)
    with pytest.raises(GeometryError, match="no strip store ever writes"):
        check_row_copy_partition(planned, full)


def test_the_partition_check_catches_two_strips_writing_the_same_rows() -> None:
    m = Model(seed=10)
    src = m.input(8, 4, 16, name="src")
    src.pin_ddr = True
    full = m.input(8, 4, 16, name="full")
    full.pin_ddr = True
    m.copy_rows(m.copy_rows(src, RowRange(0, 5), name="a"), RowRange(0, 5), into=full, at=RowRange(0, 5))
    m.copy_rows(m.copy_rows(src, RowRange(4, 8), name="b"), RowRange(0, 4), into=full, at=RowRange(4, 8))
    planned = Planner(**_MEM).plan(m)
    with pytest.raises(GeometryError, match="more than one strip store"):
        check_row_copy_partition(planned, full)


# ---------------------------------------------------------------------------
# Builder-level rejections.
# ---------------------------------------------------------------------------


def test_copy_rows_refuses_a_range_outside_the_source() -> None:
    m = Model(seed=11)
    src = m.input(8, 4, 8, name="src")
    with pytest.raises(ValueError, match="outside the tensor"):
        m.copy_rows(src, RowRange(6, 12))


def test_copy_rows_refuses_a_destination_of_a_different_row_count() -> None:
    m = Model(seed=12)
    src = m.input(8, 4, 8, name="src")
    dst = m.input(8, 4, 8, name="dst")
    with pytest.raises(ValueError, match="does not resample"):
        m.copy_rows(src, RowRange(0, 4), into=dst, at=RowRange(0, 5))


def test_copy_rows_refuses_a_width_or_plane_count_mismatch() -> None:
    m = Model(seed=13)
    src = m.input(8, 4, 8, name="src")
    dst = m.input(8, 8, 8, name="dst")
    with pytest.raises(ValueError, match="width and plane count"):
        m.copy_rows(src, RowRange(0, 4), into=dst, at=RowRange(0, 4))


# ---------------------------------------------------------------------------
# R8: the weight_reuse guard.
# ---------------------------------------------------------------------------


def test_weight_reuse_is_refused_when_the_conv_has_more_than_one_pass() -> None:
    """R8. `cmd_proc` skips the weight refill for *every* output-channel
    pass, so `WEIGHT_REUSE` with `out_channels > PE_ROWS` computes passes
    2.. on whichever weight tile happened to be resident -- silent
    numeric corruption, visible only on the DUT."""
    m = Model(seed=14)
    x = m.input(8, 4, 8, name="x")
    c1 = m.conv2d(x, 16, kernel=(1, 1), name="c1")
    m.output(m.conv2d(x, 16, kernel=(1, 1), weight_reuse_from=c1, name="c2"))
    planned = Planner(**_MEM).plan(m)
    with pytest.raises(ValueError, match="WEIGHT_REUSE"):
        emit_program(planned)


def test_weight_reuse_is_still_allowed_for_a_single_pass_conv() -> None:
    m = Model(seed=15)
    x = m.input(8, 4, 8, name="x")
    c1 = m.conv2d(x, 8, kernel=(1, 1), name="c1")
    m.output(m.conv2d(x, 8, kernel=(1, 1), weight_reuse_from=c1, name="c2"))
    emit_program(Planner(**_MEM).plan(m))
