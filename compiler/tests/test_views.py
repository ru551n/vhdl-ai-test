"""`tosa.concat` / `tosa.slice` -> buffer views, end to end.

Neither op emits an instruction. Activations live in DDR as decision-S6
channel planes (`[C/T][H][W][T]`), so a channel range of a tensor is a
contiguous range of whole planes, and:

* a channel `slice` is a read-only window into the buffer its input
  already lives in, and
* a channel `concat` is *producer-directed placement*: each operand's own
  producer is redirected to write its plane range of one result buffer.

So the tests here are not about values arriving somewhere -- they are
about values arriving at the RIGHT ADDRESS, which is the one thing a
value-only check of each producer in isolation cannot see (each part is
individually correct wherever it lands). Hence
`test_concat_costs_no_instructions_and_no_extra_bytes`, the address
assertions, and the mutation test that moves one part by a single plane
and shows the end-to-end comparison catches it.

The two limits that come from the plane granularity are pinned too:
only the channel axis, and only the LAST operand of a concat may have a
partial plane (a non-final one's padding lanes are where the next
operand's first channels must go).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from cnnc.driver import compile_tosa
from cnnc.errors import CapabilityError
from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir import interp
from cnnc.gir.ir import ConcatAttrs, SliceAttrs
from cnnc.lower.to_hir import to_hir
from cnnc.passes import PassContext, default_pipeline, run_pipeline

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# fixture name -> input shape.
FIXTURES = {
    # Two convolutions over one input, concatenated: the plainest
    # producer-directed placement there is.
    "concat_two_convs": (1, 8, 8, 8),
    # YOLOv8n's C2f shape: a conv output sliced in half, the second half
    # convolved again, all three pieces concatenated back together.
    "slice_concat_c2f": (1, 8, 8, 8),
    # Two equal, one-plane operands: the only concat shape whose parts can
    # be swapped without any address becoming illegal.
    "concat_equal_halves": (1, 8, 8, 8),
}

_PLANE_CHANNELS = 8


def _compile(name: str, target, tmp_path=None):
    out = None if tmp_path is None else tmp_path / "out"
    return compile_tosa(FIXTURES_DIR / f"{name}.mlir", target, out_dir=out)


def _seed_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-128, 128, size=shape).astype(np.int8)


def _text(name: str) -> str:
    return (FIXTURES_DIR / f"{name}.mlir").read_text()


def _fuse(text: str, target):
    graph = import_tosa(parse_module(text))
    return run_pipeline(graph, default_pipeline(target), PassContext(target=target), first_index=2)


def _buffer_with_channels(planned, channels: int):
    """The single activation buffer with this channel count. Activation
    buffers are the rank-4 ones; the program blob and the weight/bias
    consts are not."""
    matches = [
        b for b in planned.buffers.values()
        if len(b.shape) == 4 and b.shape[3] == channels and b.layout == "PLANES"
    ]
    assert len(matches) == 1, f"expected one {channels}-channel activation buffer, got {matches}"
    return matches[0]


def _plane_bytes(shape: tuple[int, ...]) -> int:
    _n, h, w, _c = shape
    return _PLANE_CHANNELS * h * w


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_reads_slice_start_from_its_const_shape_operand():
    """`start` is an operand, not an attribute, and the result type gives
    only `size` away -- so the constant really has to be read."""
    graph = import_tosa(parse_module(_text("slice_concat_c2f")))
    slices = [op for op in graph.ops if op.kind == "slice"]
    assert [op.attrs for op in slices] == [
        SliceAttrs(start=(0, 0, 0, 0), size=(1, 8, 8, 8)),
        SliceAttrs(start=(0, 0, 0, 8), size=(1, 8, 8, 8)),
    ]


def test_import_keeps_the_concat_axis():
    graph = import_tosa(parse_module(_text("concat_two_convs")))
    op = next(op for op in graph.ops if op.kind == "concat")
    assert op.attrs == ConcatAttrs(axis=3)
    assert len(op.inputs) == 2


def test_import_refuses_a_repeated_concat_operand():
    from cnnc.errors import UnsupportedAttribute

    text = (
        '"builtin.module"() ({\n'
        '  "func.func"() <{function_type = (tensor<1x4x4x8xi8>) -> tensor<1x4x4x16xi8>, '
        'sym_name = "main"}> ({\n'
        "  ^bb0(%arg0: tensor<1x4x4x8xi8>):\n"
        '    %0 = "tosa.concat"(%arg0, %arg0) <{axis = 3 : i32}> : '
        "(tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> tensor<1x4x4x16xi8>\n"
        '    "func.return"(%0) : (tensor<1x4x4x16xi8>) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )
    with pytest.raises(UnsupportedAttribute) as exc:
        import_tosa(parse_module(text))
    assert "two different channel ranges" in str(exc.value)


# ---------------------------------------------------------------------------
# Lowering: the aliasing itself
# ---------------------------------------------------------------------------


def test_concat_costs_no_instructions_and_no_extra_bytes(target, tmp_path):
    """The whole claim, stated as a test: a channel concat adds no
    descriptor to the program and no bytes to the memory image -- its
    operands' producers were simply pointed at different plane offsets of
    one buffer."""
    result = _compile("concat_two_convs", target, tmp_path)
    planned = result.hir_planned

    # One descriptor per convolution, plus HALT. Nothing for the concat.
    assert [op.kind for op in planned.ops] == ["conv_layer", "conv_layer"]
    assert len(result.program.descriptors) == 3

    concat = _buffer_with_channels(planned, 24)
    parts = [b for b in planned.buffers.values() if b.alias_kind == "part"]
    assert len(parts) == 2
    assert {p.alias_parent for p in parts} == {concat.id}
    # Every part lives inside the result, and they tile it exactly.
    assert sum(p.size_bytes for p in parts) == concat.size_bytes
    for part in parts:
        assert concat.addr <= part.addr
        assert part.addr + part.size_bytes <= concat.addr + concat.size_bytes
        assert part.addr == concat.addr + part.alias_plane_offset * _plane_bytes(part.shape)


def test_the_producers_write_straight_into_the_concat_result(target, tmp_path):
    """The descriptors' `out_addr` are the plane offsets, not some
    scratch buffer that a later copy would gather from."""
    result = _compile("concat_two_convs", target, tmp_path)
    planned = result.hir_planned
    concat = _buffer_with_channels(planned, 24)

    out_addrs = sorted(desc.out_addr for desc in result.program.descriptors[:2])
    assert out_addrs == [concat.addr, concat.addr + _plane_bytes((1, 8, 8, 8))]


def test_a_slice_is_a_window_into_its_parent(target, tmp_path):
    result = _compile("slice_concat_c2f", target, tmp_path)
    planned = result.hir_planned
    views = [b for b in planned.buffers.values() if b.alias_kind == "view"]
    assert len(views) == 2
    for view in views:
        parent = planned.buffer(view.alias_parent)
        assert view.addr == parent.addr + view.alias_plane_offset * _plane_bytes(view.shape)
        assert view.size_bytes <= parent.size_bytes
    # Nothing writes a view: the parent's producer already did.
    for op in planned.ops:
        assert not set(op.writes) & {v.id for v in views}


def test_c2f_moves_the_whole_parent_into_the_concat_result(target, tmp_path):
    """When a concat's operands are slices of one tensor, that TENSOR is
    what gets placed (once), not the slices -- a view's address is not its
    own to choose. Here the stem conv's output is placed at the concat
    result's base and its two halves fall out of that."""
    result = _compile("slice_concat_c2f", target, tmp_path)
    planned = result.hir_planned
    concat = _buffer_with_channels(planned, 24)
    parts = sorted(
        (b for b in planned.buffers.values() if b.alias_kind == "part"), key=lambda b: b.addr
    )
    assert len(parts) == 2  # the 16-channel stem, then the 8-channel branch
    assert [p.shape[3] for p in parts] == [16, 8]
    assert parts[0].addr == concat.addr
    assert parts[1].addr == concat.addr + 2 * _plane_bytes(parts[1].shape)


def test_reading_a_view_depends_on_whoever_wrote_its_parent(target, tmp_path):
    """Nothing writes a view's buffer id, so an op consuming one looks
    dependency-free unless the scheduler resolves through the alias. It
    must: the C2f branch convolution reads a slice of the stem's output
    and has to run after the stem.

    (This is not hypothetical -- the dependency was invisible until
    `HirModule.storage_dependencies` existed, and the program only came
    out in the right order because GIR order happened to agree.)"""
    result = _compile("slice_concat_c2f", target, tmp_path)
    stem, branch = result.hir_planned.ops
    assert branch.deps == (stem.id,)
    assert stem.seq < branch.seq


def test_the_program_is_exactly_the_convolutions(target, tmp_path):
    result = _compile("slice_concat_c2f", target, tmp_path)
    assert [op.kind for op in result.hir_planned.ops] == ["conv_layer", "conv_layer"]


# ---------------------------------------------------------------------------
# The limits the plane granularity imposes
# ---------------------------------------------------------------------------


def _table_concat_module(part_channels: list[int]) -> str:
    """A module that runs one `tosa.table` per entry of `part_channels`
    and concatenates the results on channels.

    `tosa.table` rather than a convolution because it is channel-
    preserving and has no output-channel divisibility rule -- which is
    what lets these tests exercise a 12-channel operand, the case the
    partial-plane rule is about. Each operand still has a real producer,
    which is what the concat lowering needs."""
    total = sum(part_channels)
    lines = [
        '"builtin.module"() ({',
        "  \"func.func\"() <{function_type = ("
        + ", ".join(f"tensor<1x4x4x{c}xi8>" for c in part_channels)
        + f") -> tensor<1x4x4x{total}xi8>, sym_name = \"main\"}}> ({{",
        "  ^bb0("
        + ", ".join(f"%arg{i}: tensor<1x4x4x{c}xi8>" for i, c in enumerate(part_channels))
        + "):",
        '    %lut = "tosa.const"() <{values = dense<1> : tensor<256xi8>}> : () -> tensor<256xi8>',
    ]
    names = []
    for i, channels in enumerate(part_channels):
        lines.append(
            f'    %t{i} = "tosa.table"(%arg{i}, %lut) : '
            f"(tensor<1x4x4x{channels}xi8>, tensor<256xi8>) -> tensor<1x4x4x{channels}xi8>"
        )
        names.append(f"%t{i}")
    in_types = ", ".join(f"tensor<1x4x4x{c}xi8>" for c in part_channels)
    lines += [
        f'    %out = "tosa.concat"({", ".join(names)}) <{{axis = 3 : i32}}> : '
        f"({in_types}) -> tensor<1x4x4x{total}xi8>",
        f'    "func.return"(%out) : (tensor<1x4x4x{total}xi8>) -> ()',
        "  }) : () -> ()",
        "}) : () -> ()",
        "",
    ]
    return "\n".join(lines)


def _height_concat_module() -> str:
    """A well-formed concat on the HEIGHT axis: two `1x4x4x8` tensors into
    one `1x8x4x8`. Valid TOSA, and not a contiguous plane range of
    anything -- interleaving it would touch every plane."""
    return "\n".join(
        [
            '"builtin.module"() ({',
            '  "func.func"() <{function_type = (tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> '
            'tensor<1x8x4x8xi8>, sym_name = "main"}> ({',
            "  ^bb0(%arg0: tensor<1x4x4x8xi8>, %arg1: tensor<1x4x4x8xi8>):",
            '    %lut = "tosa.const"() <{values = dense<1> : tensor<256xi8>}> : () -> tensor<256xi8>',
            '    %t0 = "tosa.table"(%arg0, %lut) : (tensor<1x4x4x8xi8>, tensor<256xi8>) -> '
            "tensor<1x4x4x8xi8>",
            '    %t1 = "tosa.table"(%arg1, %lut) : (tensor<1x4x4x8xi8>, tensor<256xi8>) -> '
            "tensor<1x4x4x8xi8>",
            '    %out = "tosa.concat"(%t0, %t1) <{axis = 1 : i32}> : '
            "(tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> tensor<1x8x4x8xi8>",
            '    "func.return"(%out) : (tensor<1x8x4x8xi8>) -> ()',
            "  }) : () -> ()",
            "}) : () -> ()",
            "",
        ]
    )


def test_only_the_last_operand_may_have_a_partial_plane(target):
    """A 12-channel non-final operand occupies 2 planes, 4 lanes of which
    are padding -- and those lanes are exactly where the next operand's
    first channels go. Its producer would overwrite them."""
    fused = _fuse(_table_concat_module([12, 8]), target)
    with pytest.raises(CapabilityError) as exc:
        to_hir(fused, target)
    message = str(exc.value)
    assert "padding lanes" in message
    assert "Only the final operand may have a partial plane" in message


def test_a_partial_plane_is_fine_on_the_last_operand(target):
    """The mirror image: the same channel counts the other way round has
    nothing after it to overwrite, and lowers."""
    fused = _fuse(_table_concat_module([8, 12]), target)
    module = to_hir(fused, target)
    parts = [b for b in module.buffers.values() if b.alias_kind == "part"]
    assert sorted(b.shape[3] for b in parts) == [8, 12]


def test_only_the_channel_axis_can_be_concatenated(target):
    fused = _fuse(_height_concat_module(), target)
    with pytest.raises(CapabilityError) as exc:
        to_hir(fused, target)
    assert "channel planes" in str(exc.value)


def test_a_graph_input_cannot_be_a_concat_operand(target):
    """Nothing produces a graph input, so there is no producer to redirect
    -- it would take a real COPY, which this compiler does not emit."""
    text = (
        '"builtin.module"() ({\n'
        '  "func.func"() <{function_type = (tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> '
        'tensor<1x4x4x16xi8>, sym_name = "main"}> ({\n'
        "  ^bb0(%arg0: tensor<1x4x4x8xi8>, %arg1: tensor<1x4x4x8xi8>):\n"
        '    %0 = "tosa.concat"(%arg0, %arg1) <{axis = 3 : i32}> : '
        "(tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> tensor<1x4x4x16xi8>\n"
        '    "func.return"(%0) : (tensor<1x4x4x16xi8>) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )
    with pytest.raises(CapabilityError) as exc:
        to_hir(_fuse(text, target), target)
    assert "COPY" in str(exc.value)


def test_a_slice_must_start_on_a_plane_boundary(target):
    """A view can only begin where a plane begins; a channel-4 start is
    four lanes into a plane and is not a contiguous range at all."""
    text = _text("slice_concat_c2f").replace(
        "dense<[0, 0, 0, 8]> : tensor<4xindex>", "dense<[0, 0, 0, 4]> : tensor<4xindex>"
    )
    with pytest.raises(CapabilityError) as exc:
        to_hir(_fuse(text, target), target)
    assert "plane boundary" in str(exc.value)


def test_a_concat_of_views_that_do_not_cover_their_parent_is_refused(target):
    """Only the FIRST half of the stem is concatenated, so moving the stem
    as one piece would drag the other half along to an address the concat
    never asked for."""
    text = _text("slice_concat_c2f").replace(
        '"tosa.concat"(%13, %16, %27)', '"tosa.concat"(%13, %27)'
    ).replace(
        "(tensor<1x8x8x8xi8>, tensor<1x8x8x8xi8>, tensor<1x8x8x8xi8>) -> tensor<1x8x8x24xi8>",
        "(tensor<1x8x8x8xi8>, tensor<1x8x8x8xi8>) -> tensor<1x8x8x16xi8>",
    ).replace("tensor<1x8x8x24xi8>", "tensor<1x8x8x16xi8>")
    with pytest.raises(CapabilityError) as exc:
        to_hir(_fuse(text, target), target)
    assert "do not cover" in str(exc.value)


# ---------------------------------------------------------------------------
# End to end: TOSA reference == emitted program on the golden model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_view_program_matches_tosa_reference(name, seed, target, tmp_path):
    from cnnc.backend.cnn_accel_v1 import run_program

    result = _compile(name, target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES[name], seed)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    assert set(expected) == set(actual)
    for tid, want in expected.items():
        np.testing.assert_array_equal(actual[tid], want, err_msg=f"{name} seed {seed} tensor %{tid}")


# ---------------------------------------------------------------------------
# Mutation: move a part, prove the equality test notices
# ---------------------------------------------------------------------------


def test_mutation_a_part_placed_one_plane_off_is_refused_before_emission(
    monkeypatch, target, tmp_path
):
    """Shifting one part by a plane pushes it past the end of the result,
    and the HIR verifier's containment rule stops it before a program is
    ever emitted. Pinned as a refusal, not as a wrong answer: failing here
    is strictly better than failing at the output."""
    from cnnc.hir.verify import MemoryPlanError
    from cnnc.lower import to_hir as to_hir_mod

    real = to_hir_mod._alias_buffer

    def broken(buf, parent_id, plane_offset, kind):
        return real(buf, parent_id, plane_offset + 1 if plane_offset else 0, kind)

    monkeypatch.setattr(to_hir_mod, "_alias_buffer", broken)
    with pytest.raises(MemoryPlanError) as exc:
        _compile("concat_two_convs", target, tmp_path)
    assert "not contained in parent" in str(exc.value)


def test_mutation_a_slice_view_pointed_at_the_wrong_plane_is_refused(monkeypatch, target, tmp_path):
    """Pointing the second half's view at plane 0 makes the C2f concat's
    operands stop covering their parent, and the lowering says so."""
    from cnnc.lower import to_hir as to_hir_mod

    real = to_hir_mod._lower_slice

    def broken(*args, **kwargs):
        view = real(*args, **kwargs)
        if view.alias_plane_offset == 1:
            return dataclasses.replace(view, alias_plane_offset=0)
        return view

    monkeypatch.setattr(to_hir_mod, "_lower_slice", broken)
    with pytest.raises(CapabilityError) as exc:
        _compile("slice_concat_c2f", target, tmp_path)
    assert "do not cover" in str(exc.value)


def test_mutation_swapping_two_equal_parts_breaks_the_equality_test(monkeypatch, target, tmp_path):
    """The mis-placement nothing structural can catch: two one-plane
    operands exchanged. Every address stays contained, disjoint and
    exactly tiling -- the verifier has nothing to object to, and both
    convolutions still compute correct values. They just land in each
    other's plane, and only reading the concat result back finds it."""
    from cnnc.backend.cnn_accel_v1 import run_program
    from cnnc.lower import to_hir as to_hir_mod

    real = to_hir_mod._alias_buffer
    seen: list[int] = []

    def broken(buf, parent_id, plane_offset, kind):
        if kind == "part":
            seen.append(plane_offset)
            plane_offset = 1 - plane_offset  # 0 <-> 1
        return real(buf, parent_id, plane_offset, kind)

    monkeypatch.setattr(to_hir_mod, "_alias_buffer", broken)
    result = _compile("concat_equal_halves", target, tmp_path)
    assert sorted(seen) == [0, 1], "fixture no longer has two one-plane parts to swap"

    graph = result.imported_graph
    x = _seed_input(FIXTURES["concat_equal_halves"], 1)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(actual[graph.outputs[0]], expected[graph.outputs[0]])

    # And it really is a swap, not corruption: the halves are each other's.
    swapped = np.concatenate(
        [expected[graph.outputs[0]][..., 8:], expected[graph.outputs[0]][..., :8]], axis=3
    )
    np.testing.assert_array_equal(actual[graph.outputs[0]], swapped)
