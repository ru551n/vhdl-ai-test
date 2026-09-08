"""Tests for `accel_v2.reference`: the bit-exact executor.

Two concerns, per the task's anti-fork guard:

1. `reference.py` must actually delegate to `cnn_accel_model.py`'s own
   `conv2d`/`pool_avg`/rounding primitives rather than a second,
   hand-rolled implementation -- checked here by building the SAME
   layer twice (once through `Model`/`Planner`/`run_reference`, once by
   constructing a `cnn_accel_model.LayerDesc` directly from the built
   op's own fields and calling `cnn_accel_model.conv2d`/`pool_avg`
   directly) and asserting the two outputs are identical.
2. The planner's *predicted* `DdrTraffic` must equal what running the
   program actually moves (`reference.run_reference`'s own accounting)
   -- this is the regression test for the two `planner.py` bugs found
   while integrating this module (see `test_planner.py`'s docstring).
"""

from __future__ import annotations

import cnn_accel_model as golden

from accel_v2.memimage import MemoryImage
from accel_v2.model import Activation, AddOp, Conv2dOp, Model, PoolOp
from accel_v2.planner import Planner
from accel_v2.reference import _exec_add, run_reference


def test_conv2d_matches_golden_model_directly() -> None:
    m = Model(seed=11)
    x = m.input(6, 6, 3, scale=1.0)
    y = m.conv2d(
        x,
        out_channels=5,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        activation=Activation.RELU,
    )
    m.output(y)
    planned = Planner().plan(m)
    result = run_reference(planned, MemoryImage())

    op = y.producer
    assert isinstance(op, Conv2dOp)
    desc = golden.LayerDesc(
        opcode=golden.OPCODE_CONV2D,
        flags=op.flags(),
        in_width=x.width,
        in_height=x.height,
        in_channels=x.channels,
        out_channels=y.channels,
        kernel_h=op.kernel[0],
        kernel_w=op.kernel[1],
        stride_h=op.stride[0],
        stride_w=op.stride[1],
        pad_top=op.padding[0],
        pad_bottom=op.padding[1],
        pad_left=op.padding[2],
        pad_right=op.padding[3],
        requant_scale=op.requant_scale,
        requant_shift=op.requant_shift,
    )
    expected = golden.conv2d(x.data, op.weight, op.bias, desc)
    assert result.tensor_data[y.name] == expected


def test_pool_avg_matches_golden_model_directly() -> None:
    m = Model(seed=5)
    x = m.input(8, 8, 4, scale=1.0)
    y = m.pool_avg(x, kernel=(2, 2), stride=(2, 2))
    m.output(y)
    planned = Planner().plan(m)
    result = run_reference(planned, MemoryImage())

    op = y.producer
    assert isinstance(op, PoolOp)
    desc = golden.LayerDesc(
        opcode=golden.OPCODE_POOL_AVG,
        flags=op.flags(),
        in_width=x.width,
        in_height=x.height,
        in_channels=x.channels,
        pool_kernel_h=op.kernel[0],
        pool_kernel_w=op.kernel[1],
        pool_stride_h=op.stride[0],
        pool_stride_w=op.stride[1],
        requant_scale=op.requant_scale,
        requant_shift=op.requant_shift,
    )
    expected = golden.pool_avg(x.data, desc)
    assert result.tensor_data[y.name] == expected


def _add_op(requant_scale: int, requant_shift: int) -> AddOp:
    # inputs/output are never read by _exec_add -- only requant_scale/
    # requant_shift matter -- so dummy placeholders are fine here.
    return AddOp(name="add", inputs=[], output=None, requant_scale=requant_scale, requant_shift=requant_shift)


def test_add_requant_exact_tie_rounds_towards_positive_infinity() -> None:
    # requant_scale=2**14, requant_shift=0 => requant(v) = round(v / 2),
    # ties towards +infinity (cnn_accel_model.round_shift_right_signed's
    # documented default, non-convergent rounding rule).
    op = _add_op(requant_scale=1 << 14, requant_shift=0)
    assert _exec_add(op, [3], [0]) == [2]  # 3/2 = 1.5 -> 2
    assert _exec_add(op, [-3], [0]) == [-1]  # -3/2 = -1.5 -> -1 (towards +inf)
    assert _exec_add(op, [2], [0]) == [1]  # 2/2 = 1.0, exact, no tie


def test_add_saturates_at_int8_boundaries() -> None:
    # requant_scale=2**15, requant_shift=0 => requant(v) == v exactly.
    op = _add_op(requant_scale=1 << 15, requant_shift=0)
    assert _exec_add(op, [127], [0]) == [127]  # exact boundary, no clamp
    assert _exec_add(op, [-128], [0]) == [-128]  # exact boundary, no clamp
    assert _exec_add(op, [100], [100]) == [127]  # 200 saturates to INT8_MAX
    assert _exec_add(op, [-100], [-100]) == [-128]  # -200 saturates to INT8_MIN


def _linear_chain(seed: int = 1) -> Model:
    m = Model(seed=seed)
    x = m.input(8, 8, 4)
    h1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    h2 = m.conv2d(h1, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    m.output(h2)
    return m


def _fanout_chain(seed: int = 7) -> Model:
    m = Model(seed=seed)
    x = m.input(8, 8, 4)
    h1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    h2a = m.conv2d(h1, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.NONE)
    h2b = m.conv2d(h1, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.NONE)
    h3 = m.add(h2a, h2b)
    m.output(h3)
    return m


def test_traffic_prediction_matches_actual_local_only() -> None:
    planned = Planner(tensor_mem_bytes=1 << 16).plan(_linear_chain())
    result = run_reference(planned, MemoryImage())
    assert result.traffic == planned.traffic
    assert planned.traffic.spill_count == 0


def test_traffic_prediction_matches_actual_with_spill() -> None:
    planned = Planner(tensor_mem_bytes=1024).plan(_fanout_chain())
    result = run_reference(planned, MemoryImage())
    assert result.traffic == planned.traffic
    assert planned.traffic.spill_count >= 1
    assert planned.traffic.reload_count >= 1


def test_all_op_kinds_execute_without_error_and_traffic_matches() -> None:
    m = Model(seed=3)
    x = m.input(8, 8, 4)
    c1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    p1 = m.pool_max(c1, kernel=(2, 2), stride=(2, 2))
    u1 = m.upsample2x(p1)
    cp = m.copy(u1)
    a1 = m.act(cp)
    add1 = m.add(a1, c1)
    m.output(add1)

    planned = Planner().plan(m)
    result = run_reference(planned, MemoryImage())
    assert result.traffic == planned.traffic
    assert len(result.tensor_data[add1.name]) == add1.height * add1.width * add1.channels


def test_determinism_same_seed_identical_output() -> None:
    def run(seed: int) -> list[int]:
        planned = Planner().plan(_linear_chain(seed=seed))
        result = run_reference(planned, MemoryImage())
        out_name = planned.model.outputs[0].name
        return result.tensor_data[out_name]

    assert run(42) == run(42)
    assert run(42) != run(43)
