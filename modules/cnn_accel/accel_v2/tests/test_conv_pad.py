"""Guards for `cases_conv_pad.py` -- the convolution `pad_value` cases.

These do not re-verify the arithmetic (`test_cnn_accel_model.py` does
that against an independently written reference, and `tb_cnn_accel_top`
does it against the real RTL). They verify that the CASES ARE CAPABLE OF
FAILING.

That is not a theoretical concern. The pool half of this feature (commit
4915955) first shipped a case that passed against a deliberately broken
RTL, because with ordinary random int8 data the wrong pad value produced
the same int8 output anyway. A hardware test that cannot distinguish
right from wrong is worse than no test, and nothing in the VUnit run
would ever say so. So: execute each case's program through the reference
with its convolutions' `pad_value` forced back to 0 -- exactly the
mutation of tying `cnn_accel_conv_core`'s `cfg_pad_value` to zero -- and
require the graph output to change.
"""

from __future__ import annotations

import pytest

from accel_v2 import cases_conv_pad
from accel_v2.memimage import MemoryImage
from accel_v2.model import Conv2dOp, PoolOp
from accel_v2.planner import Planner
from accel_v2.reference import run_reference
from accel_v2.tbcase import WORD_BYTES


def _outputs(case, model) -> list[list[int]]:
    """Every graph output of `model`, executed through the reference at
    `case`'s own scratchpad geometry (so a mutated copy is planned the
    same way the case itself was)."""
    planned = Planner(
        tensor_mem_bytes=case.tensor_mem_bytes,
        bank_bytes=case.bank_words * WORD_BYTES,
    ).plan(model)
    result = run_reference(planned, MemoryImage())
    return [list(result.tensor_data[t.name]) for t in model.outputs]


@pytest.mark.parametrize("builder", cases_conv_pad.CASE_BUILDERS, ids=lambda b: b.__name__)
def test_case_detects_a_conv_that_ignores_pad_value(builder) -> None:
    """Zeroing every conv `pad_value` must change the case's output.

    `case_conv_pad_default_stays_zero`'s LAST conv deliberately leaves
    `pad_value` at 0; the mutation still changes the earlier one, so the
    case as a whole is still sensitive -- which is the property under
    test here.
    """
    case = builder()
    reference_out = _outputs(case, case.model)

    # A second, identical build rather than a copy: every case is fully
    # determined by its seed, and `Model` holds an `itertools.count` for
    # auto-naming that `copy.deepcopy` cannot copy.
    mutated = builder().model
    mutated_any = False
    for op in mutated.ops:
        if isinstance(op, Conv2dOp) and op.pad_value != 0:
            op.pad_value = 0
            mutated_any = True
    assert mutated_any, f"{case.name}: no convolution sets a non-zero pad_value"

    assert _outputs(case, mutated) != reference_out, (
        f"{case.name}: forcing every conv pad_value to 0 does not change the "
        "output, so this case would pass against hardware that ignores the "
        "field entirely -- pick data that makes the fill observable"
    )


def test_conv_and_pool_case_uses_two_different_pad_values() -> None:
    """`case_conv_and_pool_pad_values_differ` exists to catch a single
    shared pad-value wire. That only works while the two values actually
    differ, so it is asserted rather than left to a reader."""
    case = cases_conv_pad.case_conv_and_pool_pad_values_differ()
    # The leading clamped 1x1 conv that shapes the activations is
    # unpadded and leaves `pad_value` at 0; only the padded conv matters.
    conv_values = {
        op.pad_value for op in case.model.ops if isinstance(op, Conv2dOp) and any(op.padding)
    }
    pool_values = {op.pad_value for op in case.model.ops if isinstance(op, PoolOp)}
    assert conv_values == {-128}
    assert pool_values == {80}

    # And swapping them changes the answer, in BOTH directions -- so
    # neither engine can be reading the other's value undetected.
    for swap in (True, False):
        mutated = cases_conv_pad.case_conv_and_pool_pad_values_differ().model
        for op in mutated.ops:
            if isinstance(op, Conv2dOp) and any(op.padding) and swap:
                op.pad_value = 80
            if isinstance(op, PoolOp) and not swap:
                op.pad_value = -128
        assert _outputs(case, mutated) != _outputs(case, case.model), f"swap={swap}"
