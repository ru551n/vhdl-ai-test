"""M5 tests: `cnnc.passes.fuse.FusePass` (doc/tosa_compiler_plan.md §9,
§13 M5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir import interp
from cnnc.gir.ir import ClampAttrs, ConvAttrs, Graph, Op, RescaleParams, Tensor, conv2d_output_shape
from cnnc.gir.printer import print_gir
from cnnc.gir.verify import verify
from cnnc.passes import FusePass, PassContext, default_pipeline, run_pipeline
from cnnc.target.load import load_target

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"
GOLDEN_PATH = Path(__file__).parent / "golden" / "conv_rescale_clamp.fused.gir.txt"


def _fixture_text() -> str:
    return FIXTURE_PATH.read_text()


def _graph_from_text(text: str) -> Graph:
    return import_tosa(parse_module(text))


def _fixture_graph() -> Graph:
    return _graph_from_text(_fixture_text())


def _clamp_5_100_graph() -> Graph:
    text = _fixture_text().replace("max_val = 127 : i8, min_val = 0 : i8", "max_val = 100 : i8, min_val = 5 : i8")
    return _graph_from_text(text)


def _no_clamp_graph() -> Graph:
    text = _fixture_text()
    lines = [ln for ln in text.splitlines() if "tosa.clamp" not in ln]
    text = "\n".join(lines).replace('"func.return"(%10)', '"func.return"(%9)')
    return _graph_from_text(text)


def _random_input(graph: Graph, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {tid: rng.integers(-128, 128, size=graph.tensor(tid).shape, dtype=np.int8) for tid in graph.inputs}


def _single_output(values: dict[str, np.ndarray]) -> np.ndarray:
    assert len(values) == 1
    return next(iter(values.values()))


# --------------------------------------------------------------------------
# Structural: fixture -> single fused_conv, golden dump
# --------------------------------------------------------------------------


def test_fixture_fuses_to_single_fused_conv():
    g = _fixture_graph()
    ctx = PassContext(target=load_target("cnn_accel"))
    g2 = FusePass().run(g, ctx)
    verify(g2)

    kinds = [op.kind for op in g2.ops]
    assert kinds.count("fused_conv") == 1
    assert kinds.count("const") == 2
    assert len(g2.ops) == 3

    assert print_gir(g2) == GOLDEN_PATH.read_text()


def test_fixture_fused_conv_matches_unfused_interp():
    g = _fixture_graph()
    ctx = PassContext(target=load_target("cnn_accel"))
    g2 = FusePass().run(g, ctx)
    inputs = _random_input(g)
    np.testing.assert_array_equal(_single_output(interp.run(g, inputs)), _single_output(interp.run(g2, inputs)))


# --------------------------------------------------------------------------
# Structural: inadmissible clamp stays standalone
# --------------------------------------------------------------------------


def test_clamp_5_100_stays_unfused_on_isa_v10_target():
    # Pre-H1 (ISA v1.0) target: only [-128,127] / [0,127] are fusable.
    from conftest import v10_target

    g = _clamp_5_100_graph()
    ctx = PassContext(target=v10_target(load_target("cnn_accel")))
    g2 = FusePass().run(g, ctx)
    verify(g2)

    fused = [op for op in g2.ops if op.kind == "fused_conv"]
    assert len(fused) == 1
    assert fused[0].attrs.clamp is None
    clamps = [op for op in g2.ops if op.kind == "clamp"]
    assert len(clamps) == 1


def test_clamp_5_100_fuses_on_real_isa_v11_target():
    # The real target is ISA v1.1 since H1 (`clamp_ranges: "any"`): the
    # general clamp is admissible and fused into the conv chain. (Lowering
    # it onto CLAMP_EN/clamp_min/clamp_max is M11; see test_to_hir.py.)
    g = _clamp_5_100_graph()
    ctx = PassContext(target=load_target("cnn_accel"))
    g2 = FusePass().run(g, ctx)
    verify(g2)

    fused = [op for op in g2.ops if op.kind == "fused_conv"]
    assert len(fused) == 1
    assert fused[0].attrs.clamp == ClampAttrs(min=5, max=100)
    assert not any(op.kind == "clamp" for op in g2.ops)

    inputs = _random_input(g)
    np.testing.assert_array_equal(_single_output(interp.run(g, inputs)), _single_output(interp.run(g2, inputs)))


# --------------------------------------------------------------------------
# Structural: no clamp at all -> fused_conv with clamp=None
# --------------------------------------------------------------------------


def test_no_clamp_variant_fuses_with_clamp_none():
    g = _no_clamp_graph()
    ctx = PassContext(target=load_target("cnn_accel"))
    g2 = FusePass().run(g, ctx)
    verify(g2)

    fused = [op for op in g2.ops if op.kind == "fused_conv"]
    assert len(fused) == 1
    assert fused[0].attrs.clamp is None
    assert g2.outputs == g.outputs

    inputs = _random_input(g)
    np.testing.assert_array_equal(_single_output(interp.run(g, inputs)), _single_output(interp.run(g2, inputs)))


# --------------------------------------------------------------------------
# Structural: conv output with two users -> not fused
# --------------------------------------------------------------------------


def test_conv_output_two_users_not_fused():
    g = _fixture_graph()
    conv_out = g.producer("9").inputs[0]  # "4": the conv2d accumulator tensor
    rescale_attrs: RescaleParams = g.producer("9").attrs
    tensors = dict(g.tensors)
    tensors["11"] = Tensor(id="11", shape=g.tensor("9").shape, dtype="i8")
    extra_rescale = Op(id="%11", kind="rescale", inputs=(conv_out,), outputs=("11",), attrs=rescale_attrs)
    g2 = g.replace(ops=g.ops + (extra_rescale,), tensors=tensors, outputs=g.outputs + ("11",))
    verify(g2)

    ctx = PassContext(target=load_target("cnn_accel"))
    g3 = FusePass().run(g2, ctx)
    assert all(op.kind != "fused_conv" for op in g3.ops)
    assert g3 is g2 or [op.kind for op in g3.ops] == [op.kind for op in g2.ops]


# --------------------------------------------------------------------------
# Randomized equivalence (50 graphs) built programmatically as GIR
# --------------------------------------------------------------------------


def _build_random_graph(rng: np.random.Generator) -> tuple[Graph, np.ndarray]:
    h = int(rng.integers(3, 7))
    w = int(rng.integers(3, 7))
    cin = int(rng.choice([1, 3, 8]))
    cout = int(rng.choice([8, 16]))
    k = int(rng.choice([1, 3]))
    stride = int(rng.choice([1, 2]))
    pad = int(rng.choice([0, 1]))

    conv_attrs = ConvAttrs(pad=(pad, pad, pad, pad), stride=(stride, stride), dilation=(1, 1), in_zp=0, w_zp=0, acc_dtype="i32")

    x = rng.integers(-128, 128, size=(1, h, w, cin), dtype=np.int8)
    weight_vals = tuple(int(v) for v in rng.integers(-128, 128, size=cout * k * k * cin, dtype=np.int8))
    bias_vals = tuple(int(v) for v in rng.integers(-1000, 1000, size=cout))

    out_shape = conv2d_output_shape(x.shape, (cout, k, k, cin), conv_attrs)

    if rng.random() < 0.3:
        mult, shift = 2**30, 31  # tie-provoking
    else:
        mult = int(rng.integers(0, 2**31))
        shift = int(rng.integers(15, 41))
    rescale_attrs = RescaleParams(
        multiplier=(mult,), shift=(shift,), per_channel=False, in_zp=0, out_zp=0,
        rounding="SINGLE_ROUND", scale32=True, input_unsigned=False, output_unsigned=False,
    )

    clamp_choice = rng.choice(["clamp_relu", "clamp_identity", "none"])

    tensors = {
        "arg0": Tensor(id="arg0", shape=x.shape, dtype="i8"),
        "w": Tensor(id="w", shape=(cout, k, k, cin), dtype="i8", values=weight_vals),
        "b": Tensor(id="b", shape=(cout,), dtype="i32", values=bias_vals),
        "4": Tensor(id="4", shape=out_shape, dtype="i32"),
        "9": Tensor(id="9", shape=out_shape, dtype="i8"),
    }
    ops = [
        Op(id="%w", kind="const", inputs=(), outputs=("w",), attrs=None),
        Op(id="%b", kind="const", inputs=(), outputs=("b",), attrs=None),
        Op(id="%4", kind="conv2d", inputs=("arg0", "w", "b"), outputs=("4",), attrs=conv_attrs),
        Op(id="%9", kind="rescale", inputs=("4",), outputs=("9",), attrs=rescale_attrs),
    ]
    outputs = ("9",)
    if clamp_choice != "none":
        clamp_bounds = (0, 127) if clamp_choice == "clamp_relu" else (-128, 127)
        tensors["10"] = Tensor(id="10", shape=out_shape, dtype="i8")
        ops.append(Op(id="%10", kind="clamp", inputs=("9",), outputs=("10",), attrs=ClampAttrs(min=clamp_bounds[0], max=clamp_bounds[1])))
        outputs = ("10",)

    g = Graph(name="main", tensors=tensors, ops=tuple(ops), inputs=("arg0",), outputs=outputs)
    verify(g)
    return g, x


def test_randomized_equivalence_50_graphs():
    rng = np.random.default_rng(2026)
    target = load_target("cnn_accel")
    for _ in range(50):
        g, x = _build_random_graph(rng)
        ctx = PassContext(target=target)
        g2 = run_pipeline(g, default_pipeline(target), ctx)

        before = _single_output(interp.run(g, {"arg0": x}))
        after = _single_output(interp.run(g2, {"arg0": x}))
        np.testing.assert_array_equal(before, after)
