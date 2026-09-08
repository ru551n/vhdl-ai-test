"""M4 tests: pass framework (`cnnc.passes.framework`), `NormalizePass`,
`LegalizeRescalePass` (doc/tosa_compiler_plan.md §11 M4)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from cnnc.errors import LegalizeError, VerifyError
from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir import interp
from cnnc.gir.interp import apply_scale_32
from cnnc.gir.ir import ClampAttrs, ConvAttrs, Graph, Op, RescaleParams, Tensor
from cnnc.gir.verify import verify
from cnnc.passes import (
    FusePass,
    LegalizeRescalePass,
    NormalizePass,
    PassContext,
    default_pipeline,
    run_pipeline,
)
from cnnc.target.load import load_target

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"


def _fixture_text() -> str:
    return FIXTURE_PATH.read_text()


def _graph_from_text(text: str) -> Graph:
    return import_tosa(parse_module(text))


def _fixture_graph() -> Graph:
    return _graph_from_text(_fixture_text())


def _fixture_graph_identity_clamp() -> Graph:
    # min_val 0 -> -128: [-128, 127] covers all of i8, an identity clamp.
    text = _fixture_text().replace("min_val = 0 : i8", "min_val = -128 : i8")
    return _graph_from_text(text)


def _random_input(graph: Graph, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        tid: rng.integers(-128, 128, size=graph.tensor(tid).shape, dtype=np.int8)
        for tid in graph.inputs
    }


def _single_output(values: dict[str, np.ndarray]) -> np.ndarray:
    assert len(values) == 1
    return next(iter(values.values()))


def _target_with_any_clamp(target):
    unit = target.units[0]
    new_epilogue = dataclasses.replace(unit.epilogue, clamp_ranges="any")
    new_unit = dataclasses.replace(unit, epilogue=new_epilogue)
    return dataclasses.replace(target, units=(new_unit,) + target.units[1:])


def _conv_rescale_graph(mult: int, shift: int, rounding: str = "SINGLE_ROUND") -> Graph:
    """Minimal 1x1 conv (weight=1, bias=0) + rescale, no clamp: `%5` is
    the graph output. Small enough for fast randomized interp checks."""
    conv_attrs = ConvAttrs(pad=(0, 0, 0, 0), stride=(1, 1), dilation=(1, 1), in_zp=0, w_zp=0, acc_dtype="i32")
    rescale_attrs = RescaleParams(
        multiplier=(mult,), shift=(shift,), per_channel=False, in_zp=0, out_zp=0,
        rounding=rounding, scale32=True, input_unsigned=False, output_unsigned=False,
    )
    tensors = {
        "arg0": Tensor(id="arg0", shape=(1, 2, 2, 1), dtype="i8"),
        "w": Tensor(id="w", shape=(1, 1, 1, 1), dtype="i8", values=(1,)),
        "b": Tensor(id="b", shape=(1,), dtype="i32", values=(0,)),
        "4": Tensor(id="4", shape=(1, 2, 2, 1), dtype="i32"),
        "5": Tensor(id="5", shape=(1, 2, 2, 1), dtype="i8"),
    }
    ops = (
        Op(id="%w", kind="const", inputs=(), outputs=("w",), attrs=None),
        Op(id="%b", kind="const", inputs=(), outputs=("b",), attrs=None),
        Op(id="%4", kind="conv2d", inputs=("arg0", "w", "b"), outputs=("4",), attrs=conv_attrs),
        Op(id="%5", kind="rescale", inputs=("4",), outputs=("5",), attrs=rescale_attrs),
    )
    return Graph(name="main", tensors=tensors, ops=ops, inputs=("arg0",), outputs=("5",))


# --------------------------------------------------------------------------
# NormalizePass
# --------------------------------------------------------------------------


def test_normalize_removes_identity_clamp_and_rewires_output():
    # Identity-clamp removal is for targets WITHOUT `clamp_ranges: "any"`
    # -- i.e. the pre-H1 ISA v1.0 target; the real target keeps it (see
    # test_normalize_keeps_identity_clamp_when_target_admits_any_clamp).
    from conftest import v10_target

    g = _fixture_graph_identity_clamp()
    ctx = PassContext(target=v10_target(load_target("cnn_accel")))
    g2 = NormalizePass().run(g, ctx)
    assert all(op.kind != "clamp" for op in g2.ops)
    assert g2.outputs == ("9",)  # rewired from the removed clamp's output to its input
    verify(g2)

    inputs = _random_input(g)
    before = _single_output(interp.run(g, inputs))
    after = _single_output(interp.run(g2, inputs))
    np.testing.assert_array_equal(before, after)


def test_normalize_keeps_non_identity_clamp():
    g = _fixture_graph()  # clamp [0, 127], not identity
    ctx = PassContext(target=load_target("cnn_accel"))
    g2 = NormalizePass().run(g, ctx)
    assert any(op.kind == "clamp" for op in g2.ops)
    assert g2.outputs == g.outputs


def test_normalize_keeps_identity_clamp_when_target_none():
    # No target information at all: still safe/harmless to remove, since
    # removal is unconditionally semantics-preserving; the target-aware
    # exception only ever *keeps* a clamp, never forces one to be dropped
    # incorrectly. Documented default: identity clamps are removed.
    g = _fixture_graph_identity_clamp()
    g2 = NormalizePass().run(g, PassContext(target=None))
    assert all(op.kind != "clamp" for op in g2.ops)


def test_normalize_keeps_identity_clamp_when_target_admits_any_clamp():
    fake_target = _target_with_any_clamp(load_target("cnn_accel"))
    g = _fixture_graph_identity_clamp()
    g2 = NormalizePass().run(g, PassContext(target=fake_target))
    assert any(op.kind == "clamp" for op in g2.ops)
    assert g2 is g or g2.outputs == g.outputs


def test_normalize_drops_dead_const():
    g = _fixture_graph()
    tensors = dict(g.tensors)
    tensors["dead_w"] = Tensor(id="dead_w", shape=(1,), dtype="i8", values=(5,))
    dead_op = Op(id="%dead_w", kind="const", inputs=(), outputs=("dead_w",), attrs=None)
    g2 = g.replace(ops=(dead_op,) + g.ops, tensors=tensors)
    verify(g2)  # dead consts are legal GIR, just wasteful

    g3 = NormalizePass().run(g2, PassContext(target=load_target("cnn_accel")))
    assert all(op.id != "%dead_w" for op in g3.ops)
    assert "dead_w" not in g3.tensors
    verify(g3)


def test_normalize_preserves_op_order():
    from conftest import v10_target

    g = _fixture_graph_identity_clamp()  # its clamp gets removed (v1.0 target)
    g2 = NormalizePass().run(g, PassContext(target=v10_target(load_target("cnn_accel"))))
    kinds_before = [op.kind for op in g.ops if op.kind != "clamp"]
    kinds_after = [op.kind for op in g2.ops]
    assert kinds_after == kinds_before


# --------------------------------------------------------------------------
# LegalizeRescalePass: pure-function property test (10k triples)
# --------------------------------------------------------------------------


def test_legalize_shift_rewrite_exactness_property():
    """`apply_scale_32(v, m, s) == apply_scale_32(v, m<<k, s+k)` for the
    `k = 15 - s` legalization step, whenever `m << k` still fits int32."""
    rng = np.random.default_rng(1234)
    checked = 0
    for _ in range(10_000):
        if rng.random() < 0.4:
            v = int(rng.integers(-8, 9))  # near-tie-provoking small values
        else:
            v = int(rng.integers(-(2**31), 2**31))
        m = int(rng.integers(0, 2**31))
        s = int(rng.integers(2, 15))
        k = 15 - s
        m2 = m << k
        if m2 >= 2**31:
            continue
        checked += 1
        a = apply_scale_32(v, m, s, "SINGLE_ROUND")
        b = apply_scale_32(v, m2, s + k, "SINGLE_ROUND")
        assert a == b
    assert checked > 500  # sanity: a substantial fraction pass the overflow filter


# --------------------------------------------------------------------------
# LegalizeRescalePass: graph-level
# --------------------------------------------------------------------------


def test_legalize_rewrites_shift_below_min():
    g = _conv_rescale_graph(mult=2**20, shift=10)
    ctx = PassContext(target=load_target("cnn_accel"))
    g2 = LegalizeRescalePass().run(g, ctx)
    rescale_op = next(op for op in g2.ops if op.kind == "rescale")
    assert rescale_op.attrs.shift == (15,)
    assert rescale_op.attrs.multiplier == (2**20 << 5,)
    verify(g2)

    inputs = _random_input(g)
    before = _single_output(interp.run(g, inputs))
    after = _single_output(interp.run(g2, inputs))
    np.testing.assert_array_equal(before, after)


def test_legalize_raises_on_multiplier_overflow():
    g = _conv_rescale_graph(mult=2**30, shift=10)  # 2**30 << 5 == 2**35, overflows
    ctx = PassContext(target=load_target("cnn_accel"))
    with pytest.raises(LegalizeError):
        LegalizeRescalePass().run(g, ctx)


def test_legalize_inference_becomes_single_round_with_note():
    g = _conv_rescale_graph(mult=2**20, shift=20, rounding="INFERENCE")
    ctx = PassContext(target=load_target("cnn_accel"))
    g2 = LegalizeRescalePass().run(g, ctx)
    rescale_op = next(op for op in g2.ops if op.kind == "rescale")
    assert rescale_op.attrs.rounding == "SINGLE_ROUND"
    assert any("INFERENCE" in note for note in ctx.notes)


def test_legalize_rejects_double_round():
    g = _conv_rescale_graph(mult=2**20, shift=20, rounding="DOUBLE_ROUND")
    ctx = PassContext(target=load_target("cnn_accel"))
    with pytest.raises(LegalizeError, match="DOUBLE_ROUND"):
        LegalizeRescalePass().run(g, ctx)


def test_legalize_noop_when_target_is_none():
    g = _conv_rescale_graph(mult=2**20, shift=10)
    ctx = PassContext(target=None)
    g2 = LegalizeRescalePass().run(g, ctx)
    assert g2 is g


def test_legalize_noop_when_shift_already_at_minimum():
    g = _conv_rescale_graph(mult=2**20, shift=15)
    ctx = PassContext(target=load_target("cnn_accel"))
    g2 = LegalizeRescalePass().run(g, ctx)
    assert g2 is g


# --------------------------------------------------------------------------
# Pipeline: dumps + verify-after-each
# --------------------------------------------------------------------------


def test_run_pipeline_writes_numbered_dumps(tmp_path):
    g = _fixture_graph()
    target = load_target("cnn_accel")
    ctx = PassContext(target=target, dump_dir=tmp_path)
    run_pipeline(g, default_pipeline(target), ctx)

    for idx, name in ((2, "normalize"), (3, "legalize_rescale"), (4, "fuse")):
        txt = tmp_path / f"{idx:02d}_{name}.txt"
        js = tmp_path / f"{idx:02d}_{name}.json"
        assert txt.is_file(), f"missing {txt}"
        assert js.is_file(), f"missing {js}"
        json.loads(js.read_text())  # valid JSON


def test_run_pipeline_verifies_after_every_pass():
    class _BreaksGraph:
        name = "broken"

        def run(self, graph, ctx):
            bad_op = Op(id="%bad", kind="clamp", inputs=("does_not_exist",), outputs=("bad_out",), attrs=ClampAttrs(min=0, max=1))
            tensors = dict(graph.tensors)
            tensors["bad_out"] = Tensor(id="bad_out", shape=(1,), dtype="i8")
            return graph.replace(ops=graph.ops + (bad_op,), tensors=tensors)

    g = _fixture_graph()
    ctx = PassContext(target=load_target("cnn_accel"))
    with pytest.raises(VerifyError):
        run_pipeline(g, [_BreaksGraph()], ctx)


def test_default_pipeline_contains_normalize_legalize_fuse():
    passes = default_pipeline(load_target("cnn_accel"))
    assert [p.name for p in passes] == ["normalize", "legalize_rescale", "fuse"]
    assert isinstance(passes[2], FusePass)
