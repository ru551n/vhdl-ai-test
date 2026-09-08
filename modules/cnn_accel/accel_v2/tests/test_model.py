"""Tests for `accel_v2.model`: the graph builder's determinism, shapes
and the `WEIGHT_REUSE` geometry guard (checked eagerly here rather than
left for `program.py` to discover late)."""

from __future__ import annotations

import pytest

import cnn_accel_model as golden

from accel_v2.model import Activation, Conv2dOp, Model


def _build_chain(seed: int) -> Model:
    m = Model(seed=seed)
    x = m.input(8, 8, 4, scale=1.0)
    c1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    p1 = m.pool_max(c1, kernel=(2, 2), stride=(2, 2))
    m.output(p1)
    return m


def test_same_seed_is_byte_identical() -> None:
    m1 = _build_chain(seed=123)
    m2 = _build_chain(seed=123)
    assert m1.inputs[0].data == m2.inputs[0].data
    c1_a = m1.ops[0]
    c1_b = m2.ops[0]
    assert isinstance(c1_a, Conv2dOp) and isinstance(c1_b, Conv2dOp)
    assert c1_a.weight == c1_b.weight
    assert c1_a.bias == c1_b.bias


def test_different_seed_differs() -> None:
    m1 = _build_chain(seed=1)
    m2 = _build_chain(seed=2)
    assert m1.inputs[0].data != m2.inputs[0].data


def test_conv2d_output_shape_matches_padding_stride() -> None:
    m = Model(seed=1)
    x = m.input(8, 8, 4)
    y = m.conv2d(x, out_channels=6, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1))
    # out = floor((in + pad_top + pad_bottom - k) / stride) + 1
    assert (y.height, y.width, y.channels) == (4, 4, 6)


def test_conv2d_rejects_non_positive_output_dims() -> None:
    m = Model(seed=1)
    x = m.input(2, 2, 4)
    with pytest.raises(ValueError, match="non-positive"):
        m.conv2d(x, out_channels=4, kernel=(5, 5), padding=(0, 0, 0, 0))


def test_tensor_size_bytes_matches_golden_activation_bytes() -> None:
    m = Model(seed=1)
    x = m.input(8, 8, 4)
    assert x.size_bytes == golden.activation_bytes(x.width, x.height, x.channels)


def test_weight_reuse_shares_weights_and_sets_flag() -> None:
    m = Model(seed=1)
    x = m.input(8, 8, 4)
    c1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1))
    c2 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), weight_reuse_from=c1)
    op1, op2 = c1.producer, c2.producer
    assert isinstance(op1, Conv2dOp) and isinstance(op2, Conv2dOp)
    assert op2.weight_reuse is True
    assert op2.reused_weight_op is op1
    assert op2.weight is op1.weight
    assert op2.flags() & (1 << golden.FLAG_WEIGHT_REUSE)


def test_weight_reuse_rejects_geometry_mismatch() -> None:
    m = Model(seed=1)
    x = m.input(8, 8, 4)
    c1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="geometry mismatch"):
        m.conv2d(x, out_channels=4, kernel=(5, 5), padding=(2, 2, 2, 2), weight_reuse_from=c1)


def test_conv2d_flags_relu_vs_clamp_are_mutually_exclusive() -> None:
    m = Model(seed=1)
    x = m.input(4, 4, 4)
    relu = m.conv2d(x, out_channels=4, activation=Activation.RELU, clamp=None)
    clamped = m.conv2d(x, out_channels=4, activation=Activation.RELU, clamp=(-10, 10))
    relu_op, clamp_op = relu.producer, clamped.producer
    assert isinstance(relu_op, Conv2dOp) and isinstance(clamp_op, Conv2dOp)
    assert relu_op.flags() & (1 << golden.FLAG_RELU_EN)
    assert not (relu_op.flags() & (1 << golden.FLAG_CLAMP_EN))
    # clamp_en set means relu_en is ignored by the epilogue (section 5.1),
    # but the builder still leaves RELU_EN clear when CLAMP_EN is set.
    assert clamp_op.flags() & (1 << golden.FLAG_CLAMP_EN)
    assert not (clamp_op.flags() & (1 << golden.FLAG_RELU_EN))


def test_act_default_lut_is_relu() -> None:
    m = Model(seed=1)
    x = m.input(2, 2, 4)
    y = m.act(x)
    op = y.producer
    assert op.lut[0] == 0  # byte 0 -> value 0 -> relu(0) == 0
    assert op.lut[1] == 1  # byte 1 -> value 1 -> relu(1) == 1
    assert op.lut[255] == 0  # byte 255 -> value -1 -> relu(-1) == 0


def test_act_rejects_wrong_size_lut() -> None:
    m = Model(seed=1)
    x = m.input(2, 2, 4)
    with pytest.raises(ValueError, match="256 entries"):
        m.act(x, lut=[0] * 10)


def test_upsample2x_doubles_spatial_dims_preserves_channels() -> None:
    m = Model(seed=1)
    x = m.input(4, 4, 4)
    y = m.upsample2x(x)
    assert (y.height, y.width, y.channels) == (8, 8, 4)
