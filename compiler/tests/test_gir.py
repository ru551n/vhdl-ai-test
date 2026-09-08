"""M2 tests: `cnnc.gir` (Graph IR dataclasses, verifier, printer/to_json)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from cnnc.errors import VerifyError
from cnnc.frontend.tosa_import import load_tosa_file
from cnnc.gir.ir import ClampAttrs, ConvAttrs, Graph, Op, RescaleParams, Tensor, conv2d_output_shape
from cnnc.gir.printer import to_json
from cnnc.gir.verify import verify

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"


def _fixture_graph() -> Graph:
    return load_tosa_file(FIXTURE_PATH)


def _simple_conv_graph(**overrides) -> Graph:
    """A minimal, valid 1x1 conv (no rescale/clamp) for programmatic
    verifier negative tests: 1x2x2x1 input, 1 out channel."""
    conv_attrs = ConvAttrs(pad=(0, 0, 0, 0), stride=(1, 1), dilation=(1, 1), in_zp=0, w_zp=0, acc_dtype="i32")
    tensors = {
        "arg0": Tensor(id="arg0", shape=(1, 2, 2, 1), dtype="i8"),
        "w": Tensor(id="w", shape=(1, 1, 1, 1), dtype="i8", values=(1,)),
        "b": Tensor(id="b", shape=(1,), dtype="i32", values=(0,)),
        "4": Tensor(id="4", shape=(1, 2, 2, 1), dtype="i32"),
    }
    ops = (
        Op(id="%w", kind="const", inputs=(), outputs=("w",), attrs=None),
        Op(id="%b", kind="const", inputs=(), outputs=("b",), attrs=None),
        Op(id="%4", kind="conv2d", inputs=("arg0", "w", "b"), outputs=("4",), attrs=conv_attrs),
    )
    kwargs = dict(name="main", tensors=tensors, ops=ops, inputs=("arg0",), outputs=("4",))
    kwargs.update(overrides)
    return Graph(**kwargs)


# --------------------------------------------------------------------------
# Verifier: positive
# --------------------------------------------------------------------------


def test_verify_passes_on_fixture_graph():
    verify(_fixture_graph())  # no exception


def test_verify_passes_on_simple_conv_graph():
    verify(_simple_conv_graph())  # no exception


# --------------------------------------------------------------------------
# Verifier: negatives, constructed programmatically
# --------------------------------------------------------------------------


def test_verify_rejects_use_of_undefined_tensor():
    g = _simple_conv_graph()
    bad_op = dataclasses.replace(g.ops[-1], inputs=("arg0", "w", "does_not_exist"))
    g2 = g.replace(ops=g.ops[:-1] + (bad_op,))
    with pytest.raises(VerifyError, match="undefined"):
        verify(g2)


def test_verify_rejects_duplicate_producer():
    g = _simple_conv_graph()
    dup = dataclasses.replace(g.ops[0], id="%w2", outputs=("w",))  # re-produces tensor "w"
    g2 = g.replace(ops=g.ops + (dup,))
    with pytest.raises(VerifyError, match="more than once"):
        verify(g2)


def test_verify_rejects_wrong_bias_length():
    g = _simple_conv_graph()
    tensors = dict(g.tensors)
    tensors["b"] = Tensor(id="b", shape=(2,), dtype="i32", values=(0, 0))
    g2 = g.replace(tensors=tensors)
    with pytest.raises(VerifyError, match="bias length"):
        verify(g2)


def test_verify_rejects_clamp_min_gt_max():
    conv_attrs = ConvAttrs(pad=(0, 0, 0, 0), stride=(1, 1), dilation=(1, 1), in_zp=0, w_zp=0, acc_dtype="i32")
    rescale_attrs = RescaleParams(
        multiplier=(1,), shift=(30,), per_channel=False, in_zp=0, out_zp=0,
        rounding="SINGLE_ROUND", scale32=True, input_unsigned=False, output_unsigned=False,
    )
    tensors = {
        "arg0": Tensor(id="arg0", shape=(1, 2, 2, 1), dtype="i8"),
        "w": Tensor(id="w", shape=(1, 1, 1, 1), dtype="i8", values=(1,)),
        "b": Tensor(id="b", shape=(1,), dtype="i32", values=(0,)),
        "4": Tensor(id="4", shape=(1, 2, 2, 1), dtype="i32"),
        "5": Tensor(id="5", shape=(1, 2, 2, 1), dtype="i8"),
        "6": Tensor(id="6", shape=(1, 2, 2, 1), dtype="i8"),
    }
    ops = (
        Op(id="%w", kind="const", inputs=(), outputs=("w",), attrs=None),
        Op(id="%b", kind="const", inputs=(), outputs=("b",), attrs=None),
        Op(id="%4", kind="conv2d", inputs=("arg0", "w", "b"), outputs=("4",), attrs=conv_attrs),
        Op(id="%5", kind="rescale", inputs=("4",), outputs=("5",), attrs=rescale_attrs),
        Op(id="%6", kind="clamp", inputs=("5",), outputs=("6",), attrs=ClampAttrs(min=100, max=0)),
    )
    g = Graph(name="main", tensors=tensors, ops=ops, inputs=("arg0",), outputs=("6",))
    with pytest.raises(VerifyError, match=r"min .* > max"):
        verify(g)


def test_verify_rejects_bias_dtype_not_i32():
    g = _simple_conv_graph()
    tensors = dict(g.tensors)
    tensors["b"] = Tensor(id="b", shape=(1,), dtype="i8", values=(0,))
    g2 = g.replace(tensors=tensors)
    with pytest.raises(VerifyError, match="bias dtype"):
        verify(g2)


def test_verify_rejects_weight_dtype_not_i8():
    g = _simple_conv_graph()
    tensors = dict(g.tensors)
    tensors["w"] = Tensor(id="w", shape=(1, 1, 1, 1), dtype="i32", values=(1,))
    g2 = g.replace(tensors=tensors)
    with pytest.raises(VerifyError, match="weight dtype"):
        verify(g2)


def test_verify_rejects_graph_input_batch_not_one():
    g = _simple_conv_graph()
    tensors = dict(g.tensors)
    tensors["arg0"] = Tensor(id="arg0", shape=(2, 2, 2, 1), dtype="i8")
    g2 = g.replace(tensors=tensors)
    with pytest.raises(VerifyError, match="batch size"):
        verify(g2)


def test_verify_rejects_const_value_length_mismatch():
    g = _simple_conv_graph()
    tensors = dict(g.tensors)
    tensors["w"] = Tensor(id="w", shape=(1, 1, 1, 1), dtype="i8", values=(1, 2))
    g2 = g.replace(tensors=tensors)
    with pytest.raises(VerifyError, match="value count"):
        verify(g2)


def test_verify_rejects_out_of_range_const_value():
    g = _simple_conv_graph()
    tensors = dict(g.tensors)
    tensors["w"] = Tensor(id="w", shape=(1, 1, 1, 1), dtype="i8", values=(200,))
    g2 = g.replace(tensors=tensors)
    with pytest.raises(VerifyError, match="out of range"):
        verify(g2)


def _conv_graph(*, in_h: int, in_w: int, k: int, stride: tuple[int, int], pad: tuple[int, int, int, int]) -> Graph:
    """A conv2d-only graph (8x8-ish, single in/out channel) with fully
    configurable spatial shape/kernel/stride/pad, for the stride-
    divisibility verifier tests below. `conv2d_output_shape` is used
    directly so the declared output tensor always matches, letting the
    stride-divisibility check (or lack thereof) be the only thing that can
    fail."""
    conv_attrs = ConvAttrs(pad=pad, stride=stride, dilation=(1, 1), in_zp=0, w_zp=0, acc_dtype="i32")
    x_shape = (1, in_h, in_w, 1)
    w_shape = (1, k, k, 1)
    out_shape = conv2d_output_shape(x_shape, w_shape, conv_attrs)
    tensors = {
        "arg0": Tensor(id="arg0", shape=x_shape, dtype="i8"),
        "w": Tensor(id="w", shape=w_shape, dtype="i8", values=tuple([1] * (k * k))),
        "b": Tensor(id="b", shape=(1,), dtype="i32", values=(0,)),
        "4": Tensor(id="4", shape=out_shape, dtype="i32"),
    }
    ops = (
        Op(id="%w", kind="const", inputs=(), outputs=("w",), attrs=None),
        Op(id="%b", kind="const", inputs=(), outputs=("b",), attrs=None),
        Op(id="%4", kind="conv2d", inputs=("arg0", "w", "b"), outputs=("4",), attrs=conv_attrs),
    )
    return Graph(name="main", tensors=tensors, ops=ops, inputs=("arg0",), outputs=("4",))


def test_verify_rejects_stride_not_dividing_padded_extent():
    # 8x8, k=3, stride=2, pad=(0,0,0,0): (8-1+0-2) % 2 == 1 != 0.
    g = _conv_graph(in_h=8, in_w=8, k=3, stride=(2, 2), pad=(0, 0, 0, 0))
    with pytest.raises(VerifyError, match="stride_h"):
        verify(g)


def test_verify_passes_stride_dividing_padded_extent():
    # 8x8, k=3, stride=2, pad=(1,0,1,0): (8-1+1-2) % 2 == 0 for both dims.
    g = _conv_graph(in_h=8, in_w=8, k=3, stride=(2, 2), pad=(1, 0, 1, 0))
    verify(g)  # no exception


# --------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------


def test_graph_producer_and_users():
    g = _fixture_graph()
    conv = g.producer("4")
    assert conv is not None
    assert conv.kind == "conv2d"
    assert g.producer("arg0") is None  # graph input, no producer
    users = g.users("4")
    assert len(users) == 1
    assert users[0].kind == "rescale"
    assert g.users("10") == ()  # graph output, no users


def test_graph_replace_returns_new_graph():
    g = _fixture_graph()
    g2 = g.replace(name="renamed")
    assert g2.name == "renamed"
    assert g.name == "main"
    assert g2 is not g


# --------------------------------------------------------------------------
# Immutability
# --------------------------------------------------------------------------


def test_tensor_is_frozen():
    t = Tensor(id="x", shape=(1,), dtype="i8")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.dtype = "i32"


def test_graph_is_frozen():
    g = _fixture_graph()
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.name = "renamed"


def test_op_is_frozen():
    op = Op(id="%0", kind="const", inputs=(), outputs=("0",), attrs=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.kind = "conv2d"


# --------------------------------------------------------------------------
# to_json determinism
# --------------------------------------------------------------------------


def test_to_json_is_deterministic():
    g = _fixture_graph()
    j1 = json.dumps(to_json(g), sort_keys=True)
    j2 = json.dumps(to_json(g), sort_keys=True)
    assert j1 == j2


def test_to_json_omits_const_values_but_hashes_them():
    g = _fixture_graph()
    dump = to_json(g)
    weight_tensor = dump["tensors"]["0"]
    assert "values" not in weight_tensor
    assert "values_sha256" in weight_tensor
    assert weight_tensor["numel"] == 288
