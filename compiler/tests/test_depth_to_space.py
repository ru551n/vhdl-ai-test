"""The pixel-shuffle idiom -> `DEPTH_TO_SPACE` (ISA v2.2), end to end.

TOSA has no depth_to_space op, exactly as it has no upsample op. What an
exporter emits for `nn.PixelShuffle(r)` is a three-op chain

    reshape -> transpose -> reshape

over a rank-6 intermediate, and `frontend.tosa_import` recognises it the
way `test_upsample.py`'s chain is recognised: by *evaluating* the index
mapping, not by matching the shape sequence.

The part with no analogue in `upsample` is the CHANNEL GROUPING, and most
of this file is about it. `nn.PixelShuffle` groups channel-major
(`cin = c*r**2 + dy*r + dx`); `OPCODE_DEPTH_TO_SPACE` groups plane-major
(`cin = (dy*r + dx)*out_c + c`), because that is what lets the engine move
whole channel tiles. Both chains are matched and the difference recorded
on the GIR op; `passes.depth_to_space_channels` then converts the first
into the second by permuting the PRODUCING CONVOLUTION's weight rows at
compile time -- free, and semantics-preserving, which is asserted here
directly (`interp` before == `interp` after) rather than inferred.

The end-to-end tests deliberately compare the emitted program's result
against the interpretation of the graph AS IMPORTED -- i.e. still
channel-major, still `nn.PixelShuffle` -- so the permutation is inside
what is being checked, not assumed correct. `test_espcn_tail_matches_a_
plain_python_pixel_shuffle` closes the loop against a definition of
PixelShuffle written out here from first principles, owing nothing to the
compiler or to the golden model.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from cnnc.backend.cnn_accel_v1 import decode_program, run_program
from cnnc.driver import compile_tosa
from cnnc.errors import CapabilityError, UnsupportedOp
from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir import interp
from cnnc.gir.ir import CHANNEL_MAJOR, PLANE_MAJOR, DepthToSpaceAttrs
from cnnc.passes import PadDepthToSpaceChannelsPass, PassContext, PermuteDepthToSpaceChannelsPass
from cnnc.passes.depth_to_space_channels import plane_major_source_rows
from cnnc.passes.depth_to_space_pad import padded_source_rows
from cnnc.target.contract import Target

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# fixture name -> input shape.
FIXTURES = {
    # A bare plane-major shuffle straight off the graph input: no
    # convolution, so no permutation -- the chain has to lower alone.
    "depth_to_space2x": (1, 3, 5, 32),
    # ESPCN's tail: conv -> ReLU -> CHANNEL-major shuffle. Exercises the
    # weight-row permutation as part of a real two-instruction program.
    "espcn_tail": (1, 4, 6, 4),
    # The same, with the output channel count a REAL luma-only ESPCN has:
    # 1, which is not a whole activation channel tile. Exercises the
    # channel PADDING pass on top of the permutation.
    "espcn_luma": (1, 4, 6, 4),
}


def _compile(name: str, target, tmp_path=None):
    out = None if tmp_path is None else tmp_path / "out"
    return compile_tosa(FIXTURES_DIR / f"{name}.mlir", target, out_dir=out)


def _text(name: str) -> str:
    return (FIXTURES_DIR / f"{name}.mlir").read_text()


def _seed_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-128, 128, size=shape).astype(np.int8)


def _dts_ops(result):
    return [op for op in result.hir_planned.ops if op.kind == "depth_to_space"]


def _pixel_shuffle_nhwc(x: np.ndarray, r: int) -> np.ndarray:
    """`torch.nn.PixelShuffle(r)` on an NHWC int array, written out from
    its own definition and owing nothing to this compiler: go to NCHW,
    split C into `[out_c, r, r]`, interleave, come back.

        out[n, c, yi*r + dy, xi*r + dx] = in[n, c*r*r + dy*r + dx, yi, xi]
    """
    n, h, w, cin = x.shape
    out_c = cin // (r * r)
    nchw = np.transpose(x.astype(np.int64), (0, 3, 1, 2))
    shuffled = (
        nchw.reshape(n, out_c, r, r, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(n, out_c, h * r, w * r)
    )
    return np.transpose(shuffled, (0, 2, 3, 1))


# ---------------------------------------------------------------------------
# Hand-built chains (no fixture file), for the shapes a fixture cannot cover
# ---------------------------------------------------------------------------


def _module(body: str, in_dims: str, out_dims: str) -> str:
    return (
        '"builtin.module"() ({\n'
        f'  "func.func"() <{{function_type = (tensor<{in_dims}xi8>) -> tensor<{out_dims}xi8>, '
        'sym_name = "main"}> ({\n'
        f"  ^bb0(%arg0: tensor<{in_dims}xi8>):\n"
        f"{body}"
        f'    "func.return"(%out) : (tensor<{out_dims}xi8>) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )


def _shape_const(name: str, values: list[int]) -> str:
    dims = ", ".join(str(v) for v in values)
    return (
        f'    {name} = "tosa.const_shape"() <{{values = dense<[{dims}]> : '
        f"tensor<{len(values)}xindex>}}> : () -> !tosa.shape<{len(values)}>\n"
    )


def _dims(shape) -> str:
    return "x".join(str(d) for d in shape)


def _chain_body(in_shape, mid, perms, out_shape, *, perms_as_operand: bool = False) -> str:
    permuted = tuple(mid[p] for p in perms)
    body = _shape_const("%s0", list(mid)) + _shape_const("%s1", list(out_shape))
    body += (
        f'    %0 = "tosa.reshape"(%arg0, %s0) : (tensor<{_dims(in_shape)}xi8>, '
        f"!tosa.shape<{len(mid)}>) -> tensor<{_dims(mid)}xi8>\n"
    )
    if perms_as_operand:
        # Pre-TOSA-1.0 spelling: `perms` is a constant OPERAND, not an
        # attribute. Real exported graphs in the wild are still written
        # this way, so the matcher reads both.
        vals = ", ".join(str(p) for p in perms)
        body += (
            f'    %p = "tosa.const"() <{{values = dense<[{vals}]> : tensor<{len(perms)}xi32>}}> '
            f": () -> tensor<{len(perms)}xi32>\n"
        )
        body += (
            f'    %1 = "tosa.transpose"(%0, %p) : (tensor<{_dims(mid)}xi8>, '
            f"tensor<{len(perms)}xi32>) -> tensor<{_dims(permuted)}xi8>\n"
        )
    else:
        vals = ", ".join(str(p) for p in perms)
        body += (
            f'    %1 = "tosa.transpose"(%0) <{{perms = array<i32: {vals}>}}> : '
            f"(tensor<{_dims(mid)}xi8>) -> tensor<{_dims(permuted)}xi8>\n"
        )
    body += (
        f'    %out = "tosa.reshape"(%1, %s1) : (tensor<{_dims(permuted)}xi8>, '
        f"!tosa.shape<{len(out_shape)}>) -> tensor<{_dims(out_shape)}xi8>\n"
    )
    return body


def _shuffle_module(in_shape, factor, *, channel_major: bool, **kwargs) -> str:
    n, h, w, cin = in_shape
    out_c = cin // (factor * factor)
    mid = (n, h, w, out_c, factor, factor) if channel_major else (n, h, w, factor, factor, out_c)
    perms = (0, 1, 4, 2, 5, 3) if channel_major else (0, 1, 3, 2, 4, 5)
    out_shape = (n, h * factor, w * factor, out_c)
    body = _chain_body(in_shape, mid, perms, out_shape, **kwargs)
    return _module(body, _dims(in_shape), _dims(out_shape))


# ---------------------------------------------------------------------------
# Recognising the chain
# ---------------------------------------------------------------------------


def test_import_collapses_the_three_op_chain_to_one_depth_to_space():
    graph = import_tosa(parse_module(_text("depth_to_space2x")))
    ops = [op for op in graph.ops if op.kind != "const"]
    assert [op.kind for op in ops] == ["depth_to_space"]
    assert ops[0].attrs == DepthToSpaceAttrs(factor=2, channel_order=PLANE_MAJOR)
    assert ops[0].inputs == ("arg0",)
    assert graph.tensor(ops[0].outputs[0]).shape == (1, 6, 10, 8)


def test_import_records_the_channel_major_grouping_as_such():
    """The channel-major chain is the SAME shape sequence as the
    plane-major one with a different permutation, so nothing but
    evaluating the mapping can tell them apart -- and the difference must
    survive into the GIR, because it decides whether a weight permutation
    is owed."""
    graph = import_tosa(parse_module(_shuffle_module((1, 3, 5, 32), 2, channel_major=True)))
    (op,) = [o for o in graph.ops if o.kind != "const"]
    assert op.attrs == DepthToSpaceAttrs(factor=2, channel_order=CHANNEL_MAJOR)


def test_the_channel_major_chain_really_is_nn_pixelshuffle():
    """Ground truth for the whole feature: what the frontend calls
    `channel_major` computes exactly what `torch.nn.PixelShuffle` does."""
    graph = import_tosa(parse_module(_shuffle_module((1, 3, 5, 32), 2, channel_major=True)))
    x = _seed_input((1, 3, 5, 32), 7)
    got = interp.run(graph, {"arg0": x})[graph.outputs[0]]
    np.testing.assert_array_equal(got, _pixel_shuffle_nhwc(x, 2))


def test_the_plane_major_chain_is_the_golden_models_depth_to_space():
    """And the other grouping is bit-exactly `cnn_accel_model.depth_to_space`
    -- the reference the RTL implements."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules" / "cnn_accel"))
    import cnn_accel_model  # noqa: E402

    shape = (1, 3, 5, 32)
    graph = import_tosa(parse_module(_shuffle_module(shape, 2, channel_major=False)))
    x = _seed_input(shape, 11)
    got = interp.run(graph, {"arg0": x})[graph.outputs[0]]

    desc = cnn_accel_model.LayerDesc(
        opcode=cnn_accel_model.OPCODE_DEPTH_TO_SPACE,
        in_width=shape[2], in_height=shape[1], in_channels=shape[3],
        out_channels=8, dts_factor=2,
    )
    want = np.array(
        cnn_accel_model.depth_to_space([int(v) for v in x.reshape(-1)], desc, factor=2)
    ).reshape(1, 6, 10, 8)
    np.testing.assert_array_equal(got, want)


def test_perms_given_as_a_constant_operand_are_read_too():
    """Pre-TOSA-1.0 exporters put `perms` in a second operand rather than
    an attribute. Same chain, same instruction."""
    module = _shuffle_module((1, 3, 5, 32), 2, channel_major=False, perms_as_operand=True)
    graph = import_tosa(parse_module(module))
    (op,) = [o for o in graph.ops if o.kind != "const"]
    assert op.attrs == DepthToSpaceAttrs(factor=2, channel_order=PLANE_MAJOR)


def test_a_factor_3_chain_is_matched_by_the_frontend():
    """Unlike UPSAMPLE, `dts_factor` IS an ISA field, so the frontend
    matches any factor and it is the LOWERING that decides what the
    hardware can do (see `test_a_factor_the_hardware_does_not_have_is_refused`).
    Keeping the two separate is what would let a 3x device work without
    touching the frontend at all."""
    graph = import_tosa(parse_module(_shuffle_module((1, 2, 3, 18), 3, channel_major=False)))
    (op,) = [o for o in graph.ops if o.kind != "const"]
    assert op.attrs == DepthToSpaceAttrs(factor=3, channel_order=PLANE_MAJOR)
    assert graph.tensor(op.outputs[0]).shape == (1, 6, 9, 2)


def test_refuses_a_reshape_transpose_chain_that_is_not_a_pixel_shuffle():
    """A chain with the right op sequence, the right rank-6 intermediate
    and a WRONG permutation: it has neither grouping's index mapping, and
    the evaluation is what says so."""
    # (0, 1, 5, 2, 4, 3) on a [1,3,5,8,2,2] intermediate still folds to
    # the right 1x6x10x8 output shape -- it just moves the wrong elements
    # there. Shape-matching would accept it; evaluation does not.
    body = _chain_body((1, 3, 5, 32), (1, 3, 5, 8, 2, 2), (0, 1, 5, 2, 4, 3), (1, 6, 10, 8))
    with pytest.raises(UnsupportedOp) as exc:
        import_tosa(parse_module(_module(body, "1x3x5x32", "1x6x10x8")))
    assert "not a depth-to-space" in str(exc.value)


def test_a_standalone_transpose_is_still_a_host_side_head_op():
    """The asymmetry that makes the idiom diagnostic safe to add:
    `tosa.transpose` is an ordinary detection-head op, and one that is NOT
    inside a reshape/transpose chain must keep being told to split the
    graph. (`tosa.tile` can afford the opposite rule -- it has no other
    reason to appear.)"""
    body = (
        '    %out = "tosa.transpose"(%arg0) <{perms = array<i32: 0, 2, 1, 3>}> : '
        "(tensor<1x4x6x32xi8>) -> tensor<1x6x4x32xi8>\n"
    )
    with pytest.raises(UnsupportedOp) as exc:
        import_tosa(parse_module(_module(body, "1x4x6x32", "1x6x4x32")))
    assert "host-side detection head" in str(exc.value)
    assert "depth-to-space" not in str(exc.value)


def test_the_upsample_idiom_still_reports_itself():
    """The two matchers share `tosa.reshape`; adding the second one must
    not have stolen the first one's diagnostic."""
    body = (
        _shape_const("%s0", [1, 2, 2, 1])
        + '    %out = "tosa.tile"(%arg0, %s0) : (tensor<1x2x3x8xi8>, !tosa.shape<4>) -> tensor<1x4x6x8xi8>\n'
    )
    with pytest.raises(UnsupportedOp) as exc:
        import_tosa(parse_module(_module(body, "1x2x3x8", "1x4x6x8")))
    assert "nearest-2x replication" in str(exc.value)


# ---------------------------------------------------------------------------
# The channel-permutation pass
# ---------------------------------------------------------------------------


def test_plane_major_source_rows_is_the_inverse_relabelling():
    """`src[j] = i`: plane-major input channel `j` must receive what the
    channel-major graph put in `i`. Checked against both index formulas
    written out independently."""
    r, out_c = 2, 8
    src = plane_major_source_rows(r, out_c)
    assert sorted(src) == list(range(r * r * out_c))  # a permutation
    for dy in range(r):
        for dx in range(r):
            for c in range(out_c):
                j = (dy * r + dx) * out_c + c
                i = c * r * r + dy * r + dx
                assert src[j] == i


def test_the_pass_is_semantics_preserving(target):
    """The property that makes the rewrite legal at all: same function
    before and after, byte for byte, on the interpreter."""
    graph = import_tosa(parse_module(_text("espcn_tail")))
    (dts,) = [o for o in graph.ops if o.kind == "depth_to_space"]
    assert dts.attrs.channel_order == CHANNEL_MAJOR

    ctx = PassContext(target=target)
    rewritten = PermuteDepthToSpaceChannelsPass().run(graph, ctx)
    (new_dts,) = [o for o in rewritten.ops if o.kind == "depth_to_space"]
    assert new_dts.attrs.channel_order == PLANE_MAJOR
    assert ctx.notes == []

    x = _seed_input(FIXTURES["espcn_tail"], 5)
    before = interp.run(graph, {"arg0": x})[graph.outputs[0]]
    after = interp.run(rewritten, {"arg0": x})[rewritten.outputs[0]]
    np.testing.assert_array_equal(before, after)


def test_the_pass_permutes_the_weight_and_bias_rows(target):
    """It really is a row relabelling of the producing convolution's
    constants -- nothing else in the graph changes."""
    graph = import_tosa(parse_module(_text("espcn_tail")))
    conv = next(o for o in graph.ops if o.kind == "conv2d")
    _x, w_id, b_id = conv.inputs
    w_before, b_before = graph.tensor(w_id), graph.tensor(b_id)

    rewritten = PermuteDepthToSpaceChannelsPass().run(graph, PassContext(target=target))
    w_after, b_after = rewritten.tensor(w_id), rewritten.tensor(b_id)

    src = plane_major_source_rows(2, 8)
    row = w_before.shape[1] * w_before.shape[2] * w_before.shape[3]
    assert w_after.shape == w_before.shape and b_after.shape == b_before.shape
    for j, i in enumerate(src):
        assert w_after.values[j * row : (j + 1) * row] == w_before.values[i * row : (i + 1) * row]
        assert b_after.values[j] == b_before.values[i]
    # Same op list, same tensors otherwise.
    assert [o.kind for o in rewritten.ops] == [o.kind for o in graph.ops]


def test_the_pass_declines_when_there_is_no_producing_convolution(target):
    """A channel-major shuffle straight off a graph input has no weight
    rows to relabel. The pass must leave it alone and say so, not invent a
    runtime permutation."""
    graph = import_tosa(parse_module(_shuffle_module((1, 3, 5, 32), 2, channel_major=True)))
    ctx = PassContext(target=target)
    rewritten = PermuteDepthToSpaceChannelsPass().run(graph, ctx)
    (op,) = [o for o in rewritten.ops if o.kind == "depth_to_space"]
    assert op.attrs.channel_order == CHANNEL_MAJOR
    assert any("could not be permuted" in n for n in ctx.notes)


def test_a_channel_major_op_that_reached_lowering_is_refused(target):
    """And then the lowering is what refuses it -- by name, with the
    reason -- rather than emitting the plane-major opcode against
    channel-major data."""
    from cnnc.lower.to_hir import to_hir

    graph = import_tosa(parse_module(_shuffle_module((1, 3, 5, 32), 2, channel_major=True)))
    with pytest.raises(CapabilityError) as exc:
        to_hir(graph, target)
    assert "channel_order" in str(exc.value)


# ---------------------------------------------------------------------------
# Lower + emit
# ---------------------------------------------------------------------------


def test_depth_to_space_lowers_onto_the_elementwise_unit(target, tmp_path):
    result = _compile("depth_to_space2x", target, tmp_path)
    (op,) = _dts_ops(result)
    assert op.unit == "elementwise_engine"
    # Input geometry PLUS both of the things the input geometry does not
    # imply: the output channel count, and the factor.
    assert dict(op.params) == {
        "in_width": 5, "in_height": 3, "in_channels": 32, "out_channels": 8, "dts_factor": 2,
    }
    assert len(op.reads) == 1


def test_descriptor_carries_the_geometry_the_factor_and_out_channels(target, tmp_path):
    result = _compile("depth_to_space2x", target, tmp_path)
    planned = result.hir_planned
    program_addr = planned.buffer(planned.program).addr
    desc = decode_program(result.program.program_bytes, target, program_addr=program_addr)[0]
    (op,) = _dts_ops(result)

    assert desc.opcode == target.isa.opcodes["DEPTH_TO_SPACE"]
    assert desc.in_addr == planned.buffer(op.reads[0]).addr
    assert desc.out_addr == planned.buffer(op.writes[0]).addr
    assert (desc.in_width, desc.in_height, desc.in_channels) == (5, 3, 32)
    assert (desc.out_channels, desc.dts_factor) == (8, 2)
    assert desc.flags == 0  # no epilogue of any kind
    assert (desc.weight_addr, desc.bias_addr, desc.scale_addr, desc.xfer_bytes) == (0, 0, 0, 0)
    assert (desc.requant_scale, desc.requant_shift) == (0, 0)


def test_the_output_buffer_trades_channels_for_area(target, tmp_path):
    """`factor**2` times the pixels, `factor**2` times fewer channels: the
    same number of bytes, unlike UPSAMPLE's 4x growth."""
    result = _compile("depth_to_space2x", target, tmp_path)
    (op,) = _dts_ops(result)
    in_buf = result.hir_planned.buffer(op.reads[0])
    out_buf = result.hir_planned.buffer(op.writes[0])
    assert out_buf.shape == (1, 6, 10, 8)
    assert out_buf.size_bytes == in_buf.size_bytes


def test_the_factor_is_read_from_the_golden_model(target):
    """v1 hardware implements exactly one `dts_factor`, and the compiler
    must take that number from the hardware rather than keep a copy."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules" / "cnn_accel"))
    import cnn_accel_model  # noqa: E402

    unit = target.unit("elementwise_engine")
    assert unit.depth_to_space_factor == cnn_accel_model.DEPTH_TO_SPACE_FACTOR


def test_a_factor_the_hardware_does_not_have_is_refused(target):
    """A 3x shuffle is a perfectly good graph that this device cannot run:
    `cnn_accel_cmd_proc` rejects any `dts_factor` but its own with
    ERR_BAD_GEOMETRY, so the compiler must not emit one."""
    from cnnc.lower.to_hir import to_hir

    graph = import_tosa(parse_module(_shuffle_module((1, 2, 3, 18), 3, channel_major=False)))
    with pytest.raises(CapabilityError) as exc:
        to_hir(graph, target)
    assert "factor 3" in str(exc.value)


def test_out_channels_below_one_channel_tile_with_no_producer_is_refused(target):
    """`out_channels = 1` makes the four input planes four LANES of one
    channel tile, which the engine has no hardware to separate. The fix is
    to pad the producing convolution's output channels up to a whole tile
    (`passes.depth_to_space_pad`) -- but THIS shuffle reads the graph's own
    INPUT, so there is no producer to widen, and widening the graph input
    is not the compiler's to do. So it is still refused, by name.

    (This test used to assert that `out_channels = 1` was refused
    UNCONDITIONALLY. That was wrong: with a producing convolution it
    compiles and is bit-exact -- see
    `test_a_luma_only_espcn_out_channels_1_now_compiles` below, which is
    the replacement for the case this test no longer covers.)"""
    from cnnc.lower.to_hir import to_hir

    graph = import_tosa(parse_module(_shuffle_module((1, 3, 5, 4), 2, channel_major=False)))
    with pytest.raises(CapabilityError) as exc:
        to_hir(graph, target)
    message = str(exc.value)
    assert "out_channels 1" in message
    assert "multiple of the activation plane width 8" in message
    assert "depth_to_space_pad" in message


def test_the_pad_pass_declines_when_there_is_no_producing_convolution(target):
    """And the pass itself is what declines, leaving the op alone and
    saying why, rather than inventing a graph-input reshape."""
    graph = import_tosa(parse_module(_shuffle_module((1, 3, 5, 4), 2, channel_major=False)))
    ctx = PassContext(target=target)
    rewritten = PadDepthToSpaceChannelsPass().run(graph, ctx)
    (op,) = [o for o in rewritten.ops if o.kind == "depth_to_space"]
    assert rewritten.tensor(op.outputs[0]).shape[3] == 1  # untouched
    assert any("could not be padded" in n for n in ctx.notes)


def test_depth_to_space_needs_a_unit_that_implements_it(target):
    from cnnc.lower.to_hir import to_hir

    graph = import_tosa(parse_module(_text("depth_to_space2x")))
    data = target.to_dict()
    for unit in data["units"]:
        unit["ops"] = [o for o in unit["ops"] if o != "depth_to_space"]
        unit["depth_to_space_factor"] = None
    with pytest.raises(CapabilityError) as exc:
        to_hir(graph, Target.from_dict(data))
    assert "depth_to_space" in str(exc.value)


# ---------------------------------------------------------------------------
# End to end: TOSA reference == emitted program on the golden model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_program_matches_the_tosa_reference(name, seed, target, tmp_path):
    """`interp` runs the graph AS IMPORTED -- for `espcn_tail` that is
    still the channel-major `nn.PixelShuffle` -- so the weight permutation
    is part of what this equality is testing."""
    result = _compile(name, target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES[name], seed)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    assert set(expected) == set(actual)
    for tid, want in expected.items():
        np.testing.assert_array_equal(actual[tid], want, err_msg=f"{name} seed {seed} tensor %{tid}")


def test_espcn_tail_matches_a_plain_python_pixel_shuffle(target, tmp_path):
    """Independently of the interpreter AND of the golden model's own
    depth_to_space: run the compiled program, then compute what
    `nn.PixelShuffle` would give from the convolution's own output, and
    require the two to agree. This is the test that would catch a
    permutation applied in the wrong direction."""
    result = _compile("espcn_tail", target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES["espcn_tail"], 9)

    # The conv+ReLU output, straight from the interpreter, before any
    # shuffle -- in the graph's own (channel-major) channel order.
    dts = next(o for o in graph.ops if o.kind == "depth_to_space")
    conv_out = interp.evaluate_all(graph, {graph.inputs[0]: x})[dts.inputs[0]]
    want = _pixel_shuffle_nhwc(np.asarray(conv_out), 2)

    got = run_program(result.program, {graph.inputs[0]: x})[graph.outputs[0]]
    np.testing.assert_array_equal(got, want)


def test_the_espcn_tail_is_two_instructions(target, tmp_path):
    """CONV2D then DEPTH_TO_SPACE: the permutation costs no instruction,
    which is the entire claim being made for it."""
    result = _compile("espcn_tail", target, tmp_path)
    planned = result.hir_planned
    program_addr = planned.buffer(planned.program).addr
    descs = decode_program(result.program.program_bytes, target, program_addr=program_addr)
    opcodes = [d.opcode for d in descs]
    assert opcodes == [
        target.isa.opcodes["CONV2D"],
        target.isa.opcodes["DEPTH_TO_SPACE"],
        target.isa.opcodes["HALT"],
    ]


# ---------------------------------------------------------------------------
# Mutation: break the lowering, prove the equality tests notice
# ---------------------------------------------------------------------------


def test_mutation_swapping_the_descriptor_geometry_breaks_the_equality_test(
    monkeypatch, target, tmp_path
):
    """`in_width`/`in_height` are not interchangeable; the fixture is 3x5
    so transposing them walks the ifmap in the wrong order."""
    from cnnc.backend.cnn_accel_v1 import emit

    real = emit._build_depth_to_space_descriptor

    def broken(*args, **kwargs):
        desc = real(*args, **kwargs)
        return dataclasses.replace(desc, in_width=desc.in_height, in_height=desc.in_width)

    monkeypatch.setattr(emit, "_build_depth_to_space_descriptor", broken)
    result = _compile("depth_to_space2x", target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES["depth_to_space2x"], 1)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(actual[graph.outputs[0]], expected[graph.outputs[0]])


def test_mutation_skipping_the_channel_permutation_breaks_the_equality_test(
    monkeypatch, target, tmp_path
):
    """The permutation is load-bearing, not cosmetic: emit the
    plane-major opcode against channel-major weights and the ESPCN tail
    stops matching `nn.PixelShuffle`. (Without this, a pass that silently
    did nothing would still pass every other test in this file.)"""
    from cnnc.passes import depth_to_space_channels

    monkeypatch.setattr(
        depth_to_space_channels,
        "plane_major_source_rows",
        lambda factor, out_channels: tuple(range(factor * factor * out_channels)),
    )
    result = _compile("espcn_tail", target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES["espcn_tail"], 1)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(actual[graph.outputs[0]], expected[graph.outputs[0]])


# ---------------------------------------------------------------------------
# The channel-PADDING pass: a real ESPCN's `out_channels` (1 for luma, 3
# for RGB) is not a whole activation channel tile, and the hardware moves
# nothing smaller. Padding the producing convolution up to a whole tile is
# what makes such a graph compile -- with no new hardware, because the
# padding lanes are the same ones `unpack_activation_planes` already drops
# for every other tensor whose channel count is not a multiple of T.
# ---------------------------------------------------------------------------


def test_padded_source_rows_plane_major_interleaves_and_zero_fills():
    """Plane-major grows every plane from `out_channels` to
    `padded` wide, so the real rows scatter and the gaps are dummies."""
    src = padded_source_rows(2, 3, 8, PLANE_MAJOR)
    assert len(src) == 4 * 8
    for plane in range(4):
        for c in range(8):
            want = plane * 3 + c if c < 3 else None
            assert src[plane * 8 + c] == want
    # Every real row used exactly once, nothing invented.
    assert sorted(i for i in src if i is not None) == list(range(4 * 3))


def test_padded_source_rows_channel_major_is_a_pure_append():
    """Channel-major numbers channels outermost, so the added channels sit
    entirely after the existing ones and nothing moves."""
    src = padded_source_rows(2, 1, 8, CHANNEL_MAJOR)
    assert len(src) == 4 * 8
    assert src[:4] == (0, 1, 2, 3)
    assert set(src[4:]) == {None}


def test_the_pad_pass_widens_the_conv_rows_with_zeros(target):
    """The added output channels are genuinely DUMMY: zero weight row,
    zero bias. The real rows are carried over unchanged."""
    graph = import_tosa(parse_module(_text("espcn_luma")))
    conv = next(o for o in graph.ops if o.kind == "conv2d")
    _x, w_id, b_id = conv.inputs
    w_before, b_before = graph.tensor(w_id), graph.tensor(b_id)
    assert w_before.shape[0] == 4 and b_before.shape == (4,)  # factor**2 * 1

    # With `out_channels == 1` the two channel groupings are the SAME
    # mapping (`c` only ever takes the value 0), so the frontend reports
    # the plane-major one and the padding interleaves rather than appends:
    # the four real rows land at 0, 8, 16, 24, one per plane.
    (dts,) = [o for o in graph.ops if o.kind == "depth_to_space"]
    assert dts.attrs.channel_order == PLANE_MAJOR
    src = padded_source_rows(2, 1, 8, PLANE_MAJOR)
    assert [j for j, i in enumerate(src) if i is not None] == [0, 8, 16, 24]

    rewritten = PadDepthToSpaceChannelsPass().run(graph, PassContext(target=target))
    w_after, b_after = rewritten.tensor(w_id), rewritten.tensor(b_id)

    assert w_after.shape == (32,) + tuple(w_before.shape[1:])
    assert b_after.shape == (32,)
    row = w_before.shape[1] * w_before.shape[2] * w_before.shape[3]
    for j, i in enumerate(src):
        after = w_after.values[j * row : (j + 1) * row]
        if i is None:
            # A dummy channel: zero weights AND zero bias, so it computes
            # nothing and contributes nothing.
            assert set(after) == {0}
            assert b_after.values[j] == 0
        else:
            assert after == w_before.values[i * row : (i + 1) * row]
            assert b_after.values[j] == b_before.values[i]


def test_the_pad_pass_widens_the_shuffle_shapes_and_keeps_the_true_one(target):
    """GIR must not disagree with the instruction about how many channels
    get written -- so the shuffle's shapes really are widened -- while the
    tensor's true shape survives as `logical_shape`."""
    graph = import_tosa(parse_module(_text("espcn_luma")))
    (dts,) = [o for o in graph.ops if o.kind == "depth_to_space"]
    assert graph.tensor(dts.inputs[0]).shape == (1, 4, 6, 4)
    assert graph.tensor(dts.outputs[0]).shape == (1, 8, 12, 1)

    rewritten = PadDepthToSpaceChannelsPass().run(graph, PassContext(target=target))
    (dts,) = [o for o in rewritten.ops if o.kind == "depth_to_space"]
    x, y = rewritten.tensor(dts.inputs[0]), rewritten.tensor(dts.outputs[0])
    assert x.shape == (1, 4, 6, 32)  # factor**2 * 8
    assert y.shape == (1, 8, 12, 8)
    assert y.logical_shape == (1, 8, 12, 1)
    assert x.logical_shape is None  # an intermediate: nobody outside reads it


def test_padding_costs_no_bytes(target):
    """The whole argument for doing this at all: `C = 1` and `C = 8`
    occupy the SAME whole channel plane, so widening the tensor moves no
    byte and costs no DDR."""
    from cnnc.lower.layout import activation_bytes

    plane_channels = target.memory.activation_plane_channels
    assert activation_bytes((1, 8, 12, 1), plane_channels) == activation_bytes(
        (1, 8, 12, 8), plane_channels
    )


def test_a_luma_only_espcn_out_channels_1_now_compiles(target, tmp_path):
    """The case that used to be a `CapabilityError`. It compiles, and the
    instruction says the padded count while the manifest keeps the true
    one."""
    result = _compile("espcn_luma", target, tmp_path)
    planned = result.hir_planned
    (op,) = _dts_ops(result)

    # The descriptor field is the PADDED count -- a whole channel tile,
    # which is the only thing the engine can address.
    assert op.params["out_channels"] == 8
    assert op.params["in_channels"] == 32  # factor**2 * 8
    desc = decode_program(
        result.program.program_bytes, target, program_addr=planned.buffer(planned.program).addr
    )[1]
    assert desc.opcode == target.isa.opcodes["DEPTH_TO_SPACE"]
    assert (desc.out_channels, desc.in_channels, desc.dts_factor) == (8, 32, 2)

    # The buffer is written 8 channels wide and is still a 1-channel
    # tensor; both facts are recorded, and they imply the same size.
    out_buf = planned.buffer(op.writes[0])
    assert out_buf.shape == (1, 8, 12, 8)
    assert out_buf.logical_shape == (1, 8, 12, 1)
    (entry,) = [b for b in result.program.manifest["buffers"] if b["role"] == "output"]
    assert entry["shape"] == [1, 8, 12, 8]
    assert entry["logical_shape"] == [1, 8, 12, 1]
    assert entry["size_bytes"] == 8 * 12 * 8


def test_the_luma_espcn_is_still_two_instructions(target, tmp_path):
    """Padding buys the compile with dummy CHANNELS, not with an extra
    instruction or a host-side fixup pass."""
    result = _compile("espcn_luma", target, tmp_path)
    planned = result.hir_planned
    descs = decode_program(
        result.program.program_bytes, target, program_addr=planned.buffer(planned.program).addr
    )
    assert [d.opcode for d in descs] == [
        target.isa.opcodes["CONV2D"],
        target.isa.opcodes["DEPTH_TO_SPACE"],
        target.isa.opcodes["HALT"],
    ]


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_the_luma_espcn_matches_a_plain_python_pixel_shuffle(target, tmp_path, seed):
    """The real bar: the ACTUAL bytes the device writes for the one real
    channel are right, padding included but ignored. Checked against
    `nn.PixelShuffle` written out from its own definition, on the
    convolution's own (unpadded, channel-major) output."""
    result = _compile("espcn_luma", target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES["espcn_luma"], seed)

    dts = next(o for o in graph.ops if o.kind == "depth_to_space")
    conv_out = interp.evaluate_all(graph, {graph.inputs[0]: x})[dts.inputs[0]]
    want = _pixel_shuffle_nhwc(np.asarray(conv_out), 2)
    assert want.shape == (1, 8, 12, 1)

    got = run_program(result.program, {graph.inputs[0]: x})[graph.outputs[0]]
    assert got.shape == (1, 8, 12, 1)  # the TRUE shape, not the padded one
    np.testing.assert_array_equal(got, want)


def test_the_luma_espcn_matches_the_golden_models_depth_to_space(target, tmp_path):
    """And against `cnn_accel_model.depth_to_space` itself, fed the padded
    plane-major tensor the instruction actually reads: the golden model's
    8-channel result, cropped to its true 1 channel, is what comes back."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules" / "cnn_accel"))
    import cnn_accel_model  # noqa: E402

    result = _compile("espcn_luma", target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES["espcn_luma"], 11)

    # The padded, plane-major tensor the DEPTH_TO_SPACE instruction reads,
    # taken from the compiled graph's own interpretation (so the padding
    # and the permutation are both inside what is being checked).
    (dts,) = [o for o in result.fused_graph.ops if o.kind == "depth_to_space"]
    hw_in = np.asarray(
        interp.evaluate_all(result.fused_graph, {graph.inputs[0]: x})[dts.inputs[0]]
    )
    assert hw_in.shape == (1, 4, 6, 32)

    desc = cnn_accel_model.LayerDesc(
        opcode=target.isa.opcodes["DEPTH_TO_SPACE"],
        in_width=6, in_height=4, in_channels=32, out_channels=8,
    )
    flat = cnn_accel_model.depth_to_space([int(v) for v in hw_in.reshape(-1)], desc, factor=2)
    model_out = np.asarray(flat, dtype=np.int8).reshape(1, 8, 12, 8)

    got = run_program(result.program, {graph.inputs[0]: x})[graph.outputs[0]]
    np.testing.assert_array_equal(got, model_out[:, :, :, :1])


def test_mutation_padding_the_wrong_rows_breaks_the_equality_test(monkeypatch, target, tmp_path):
    """The padding's ROW PLACEMENT is load-bearing, not just its row
    COUNT. Append the real rows instead of interleaving them -- a graph
    that still has exactly the right number of channels, the right buffer
    size and the right descriptor -- and the ESPCN tail stops matching
    `nn.PixelShuffle`. Without this, a pass that widened to the right
    shape but put the real data in the wrong lanes would pass every other
    test in this file."""
    from cnnc.passes import depth_to_space_pad

    def appended(factor, out_channels, padded_out_channels, channel_order):
        real = factor * factor * out_channels
        total = factor * factor * padded_out_channels
        return tuple(list(range(real)) + [None] * (total - real))

    monkeypatch.setattr(depth_to_space_pad, "padded_source_rows", appended)
    result = _compile("espcn_luma", target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES["espcn_luma"], 1)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(actual[graph.outputs[0]], expected[graph.outputs[0]])
