"""M3 tests: `cnnc.gir.interp` (TOSA-semantics reference execution,
doc/tosa_compiler_plan.md §6, §10 item 1, §13 M3 acceptance criteria)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from cnnc.frontend.tosa_import import load_tosa_file
from cnnc.gir import interp
from cnnc.gir.interp import AccumulatorOverflow, apply_scale_32, evaluate_all, rescale_array, run
from cnnc.gir.ir import ClampAttrs, ConvAttrs, FusedConvAttrs, Graph, Op, RescaleParams, Tensor

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"


def _rescale_params(mult: int, shift: int, rounding: str, *, in_zp: int = 0, out_zp: int = 0) -> RescaleParams:
    return RescaleParams(
        multiplier=(mult,),
        shift=(shift,),
        per_channel=False,
        in_zp=in_zp,
        out_zp=out_zp,
        rounding=rounding,
        scale32=True,
        input_unsigned=False,
        output_unsigned=False,
    )


# --------------------------------------------------------------------------
# rescale tie table (§13 M3 acceptance: matches IREE's observed rounding)
# --------------------------------------------------------------------------

_TIE_VALUES = (1, -1, 5, -5, 3, -3)
_TIE_EXPECTED = (1, 0, 3, -2, 2, -1)


def test_rescale_tie_table_single_round():
    got = tuple(apply_scale_32(v, 2**30, 31, "SINGLE_ROUND") for v in _TIE_VALUES)
    assert got == _TIE_EXPECTED


def test_rescale_tie_table_identity_legalisation():
    # shift=31,mult=2**30 and shift=1,mult=1 both encode the multiplier
    # 0.5; a shift<15 legalisation step that scales both identically must
    # produce bit-identical results.
    a = tuple(apply_scale_32(v, 2**30, 31, "SINGLE_ROUND") for v in _TIE_VALUES)
    b = tuple(apply_scale_32(v, 1, 1, "SINGLE_ROUND") for v in _TIE_VALUES)
    assert a == b


def test_rescale_double_round_matches_spec_formula():
    mult, shift = 2**30, 32
    for v in _TIE_VALUES:
        prod = v * mult
        r = 1 << (shift - 1)
        r += (1 << 30) if v >= 0 else -(1 << 30)
        expected = (prod + r) >> shift
        assert apply_scale_32(v, mult, shift, "DOUBLE_ROUND") == expected


def test_rescale_inference_equals_single_round():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = int(rng.integers(-(2**20), 2**20))
        mult = int(rng.integers(0, 2**31))
        shift = int(rng.integers(2, 62))
        assert apply_scale_32(v, mult, shift, "INFERENCE") == apply_scale_32(v, mult, shift, "SINGLE_ROUND")


def test_rescale_clamps_to_i8():
    params = _rescale_params(2**30, 31, "SINGLE_ROUND")
    x = np.array([1000], dtype=np.int64)
    assert rescale_array(x, params, "i8").tolist() == [127]


# --------------------------------------------------------------------------
# rescale: numpy path == scalar path (readable spec vs fast path)
# --------------------------------------------------------------------------


def test_rescale_array_matches_scalar_reference_per_tensor():
    rng = np.random.default_rng(0)
    for rounding in ("SINGLE_ROUND", "DOUBLE_ROUND", "INFERENCE"):
        for _ in range(20):
            n = 30
            values = rng.integers(-(2**30), 2**30, size=n).astype(np.int64)
            mult = int(rng.integers(0, 2**31))
            shift = int(rng.integers(2, 62))
            in_zp = int(rng.integers(-5, 5))
            out_zp = int(rng.integers(-5, 5))
            params = _rescale_params(mult, shift, rounding, in_zp=in_zp, out_zp=out_zp)
            got = rescale_array(values, params, "i8")
            expected = [
                min(max(apply_scale_32(int(v) - in_zp, mult, shift, rounding) + out_zp, -128), 127) for v in values
            ]
            assert got.tolist() == expected


def test_rescale_array_matches_scalar_reference_per_channel():
    rng = np.random.default_rng(1)
    c = 6
    for rounding in ("SINGLE_ROUND", "DOUBLE_ROUND"):
        for _ in range(10):
            values = rng.integers(-(2**30), 2**30, size=(2, 3, c)).astype(np.int64)
            mults = tuple(int(rng.integers(0, 2**31)) for _ in range(c))
            shifts = tuple(int(rng.integers(2, 62)) for _ in range(c))
            params = RescaleParams(
                multiplier=mults,
                shift=shifts,
                per_channel=True,
                in_zp=0,
                out_zp=0,
                rounding=rounding,
                scale32=True,
                input_unsigned=False,
                output_unsigned=False,
            )
            got = rescale_array(values, params, "i8")
            expected = np.empty_like(values)
            it = np.nditer(values, flags=["multi_index"])
            for v in it:
                idx = it.multi_index
                ch = idx[-1]
                s = apply_scale_32(int(v), mults[ch], shifts[ch], rounding)
                expected[idx] = min(max(s, -128), 127)
            assert got.tolist() == expected.tolist()


# --------------------------------------------------------------------------
# conv2d
# --------------------------------------------------------------------------


def _conv_attrs(pad=(0, 0, 0, 0), stride=(1, 1), in_zp=0, w_zp=0) -> ConvAttrs:
    return ConvAttrs(pad=pad, stride=stride, dilation=(1, 1), in_zp=in_zp, w_zp=w_zp, acc_dtype="i32")


def test_conv2d_identity_1x1():
    x = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int8)  # 1x2x2x1
    w = np.array([[[[1]]]], dtype=np.int8)  # 1x1x1x1, OHWI
    bias = np.zeros((1,), dtype=np.int32)
    out = interp._conv2d_int64(x, w, bias, _conv_attrs(), "%t")
    assert out.tolist() == x.astype(np.int64).tolist()


def test_conv2d_centre_delta_kernel_is_identity():
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=(1, 4, 4, 1)).astype(np.int8)
    w = np.zeros((1, 3, 3, 1), dtype=np.int8)
    w[0, 1, 1, 0] = 1
    bias = np.zeros((1,), dtype=np.int32)
    out = interp._conv2d_int64(x, w, bias, _conv_attrs(pad=(1, 1, 1, 1)), "%t")
    assert out.tolist() == x.astype(np.int64).tolist()


def test_conv2d_shifted_delta_kernel_shifts_with_zero_fill():
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=(1, 4, 4, 1)).astype(np.int8)
    w = np.zeros((1, 3, 3, 1), dtype=np.int8)
    w[0, 0, 0, 0] = 1  # top-left tap
    bias = np.zeros((1,), dtype=np.int32)
    out = interp._conv2d_int64(x, w, bias, _conv_attrs(pad=(1, 1, 1, 1)), "%t")
    expected = np.zeros_like(x, dtype=np.int64)
    expected[:, 1:, 1:, :] = x.astype(np.int64)[:, :-1, :-1, :]
    assert out.tolist() == expected.tolist()


def test_conv2d_in_zp_pads_with_zero_point():
    # 3x3 all-ones weights, all-ones input, in_zp=3, pad=1: interior taps
    # contribute (1-3)=-2 each; padded taps contribute (in_zp-in_zp)=0.
    x = np.ones((1, 3, 3, 1), dtype=np.int8)
    w = np.ones((1, 3, 3, 1), dtype=np.int8)
    bias = np.zeros((1,), dtype=np.int32)
    out = interp._conv2d_int64(x, w, bias, _conv_attrs(pad=(1, 1, 1, 1), in_zp=3), "%t")
    # corner (0,0): 4 valid taps -> -8; edge (0,1): 6 valid taps -> -12; centre (1,1): 9 valid taps -> -18
    assert out[0, 0, 0, 0] == -8
    assert out[0, 0, 1, 0] == -12
    assert out[0, 1, 1, 0] == -18


def _ref_conv2d(x: np.ndarray, w: np.ndarray, bias: np.ndarray, pad, stride, in_zp: int, w_zp: int) -> np.ndarray:
    """Pure-Python triple-nested-loop reference for `conv2d` (the readable
    spec `_conv2d_int64`'s vectorised tensordot loop is cross-checked
    against)."""
    n, ih, iw, cin = x.shape
    oc, kh, kw, _ = w.shape
    pad_t, pad_b, pad_l, pad_r = pad
    sh, sw = stride
    oh = (ih + pad_t + pad_b - (kh - 1) - 1) // sh + 1
    ow = (iw + pad_l + pad_r - (kw - 1) - 1) // sw + 1
    out = np.zeros((n, oh, ow, oc), dtype=np.int64)
    for b in range(n):
        for oy in range(oh):
            for ox in range(ow):
                for oc_i in range(oc):
                    acc = 0
                    for ky in range(kh):
                        iy = oy * sh + ky - pad_t
                        for kx in range(kw):
                            ix = ox * sw + kx - pad_l
                            for ic in range(cin):
                                xv = int(x[b, iy, ix, ic]) if (0 <= iy < ih and 0 <= ix < iw) else in_zp
                                wv = int(w[oc_i, ky, kx, ic])
                                acc += (xv - in_zp) * (wv - w_zp)
                    out[b, oy, ox, oc_i] = acc + int(bias[oc_i])
    return out


def test_conv2d_matches_bruteforce_reference_random_shapes():
    rng = np.random.default_rng(0)
    zp_choices = (0, 3, -2)
    for h, w_, cin, cout, k, stride, pad in itertools.product(
        (4, 5), (4, 5), (1, 3), (1, 2), (1, 3), (1, 2), (0, 1)
    ):
        in_zp = int(rng.choice(zp_choices))
        w_zp = int(rng.choice(zp_choices))
        x = rng.integers(-128, 128, size=(1, h, w_, cin)).astype(np.int8)
        wt = rng.integers(-128, 128, size=(cout, k, k, cin)).astype(np.int8)
        bias = rng.integers(-1000, 1000, size=(cout,)).astype(np.int32)
        attrs = _conv_attrs(pad=(pad, pad, pad, pad), stride=(stride, stride), in_zp=in_zp, w_zp=w_zp)
        got = interp._conv2d_int64(x, wt, bias, attrs, "%t")
        expected = _ref_conv2d(x, wt, bias, attrs.pad, attrs.stride, in_zp, w_zp)
        assert got.tolist() == expected.tolist(), (h, w_, cin, cout, k, stride, pad, in_zp, w_zp)


# --------------------------------------------------------------------------
# overflow
# --------------------------------------------------------------------------


def test_conv2d_accumulator_overflow_raises():
    cin = 100_000
    x = np.full((1, 1, 1, cin), -128, dtype=np.int8)
    w = np.full((1, 1, 1, cin), -128, dtype=np.int8)  # (-128 - 0) * (-128 - 0) = 16384 per tap
    bias = np.array([600_000_000], dtype=np.int32)  # pushes 16384 * 100_000 = 1_638_400_000 past int32 max
    attrs = _conv_attrs()
    with pytest.raises(AccumulatorOverflow) as exc:
        interp._conv2d_int64(x, w, bias, attrs, "%4")
    assert exc.value.op_id == "%4"


# --------------------------------------------------------------------------
# fixture end-to-end
# --------------------------------------------------------------------------


def _fixture_input(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=(1, 8, 8, 4)).astype(np.int8)


def test_fixture_run_matches_manual_composition():
    graph = load_tosa_file(FIXTURE_PATH)
    x = _fixture_input()
    out = run(graph, {"arg0": x})
    assert out["10"].dtype == np.int8
    assert out["10"].shape == (1, 8, 8, 8)

    conv_op = graph.producer("4")
    rescale_op = graph.producer("9")
    clamp_op = graph.producer("10")
    w = np.asarray(graph.tensor("0").values, dtype=np.int64).reshape(graph.tensor("0").shape)
    b = np.asarray(graph.tensor("1").values, dtype=np.int64).reshape(graph.tensor("1").shape)

    acc = interp._conv2d_int64(x, w, b, conv_op.attrs, conv_op.id)
    resc = rescale_array(acc, rescale_op.attrs, "i8")
    clamped = interp._clamp(resc, clamp_op.attrs)
    assert out["10"].tolist() == clamped.astype(np.int8).tolist()


def test_fixture_fused_conv_matches_unfused():
    graph = load_tosa_file(FIXTURE_PATH)
    x = _fixture_input()
    unfused_out = run(graph, {"arg0": x})["10"]

    conv_op = graph.producer("4")
    rescale_op = graph.producer("9")
    clamp_op = graph.producer("10")
    out_tensor = graph.tensor("10")

    fused_attrs = FusedConvAttrs(conv=conv_op.attrs, rescale=rescale_op.attrs, clamp=clamp_op.attrs)
    fused_tensors = {
        "arg0": graph.tensor("arg0"),
        "0": graph.tensor("0"),
        "1": graph.tensor("1"),
        "100": Tensor(id="100", shape=out_tensor.shape, dtype=out_tensor.dtype),
    }
    fused_ops = (
        Op(id="%0", kind="const", inputs=(), outputs=("0",), attrs=None),
        Op(id="%1", kind="const", inputs=(), outputs=("1",), attrs=None),
        Op(id="%100", kind="fused_conv", inputs=("arg0", "0", "1"), outputs=("100",), attrs=fused_attrs),
    )
    fused_graph = Graph(name="main", tensors=fused_tensors, ops=fused_ops, inputs=("arg0",), outputs=("100",))
    fused_out = run(fused_graph, {"arg0": x})["100"]

    assert fused_out.tolist() == unfused_out.tolist()
