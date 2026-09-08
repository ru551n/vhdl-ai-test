"""`tb_cnn_accel_top` cases for the ISA v2.1 pooling extensions.

A second catalogue alongside `cases.py`, registered by
`module_cnn_accel.py` in the same loop and following exactly the same
contract (one function per case, each returning a `TbCase`, each naming
its own seed). It is a separate file only so the two can be edited
independently; there is nothing structurally different about these
cases.

What is under test here is the pair of hardware limits raised for
YOLOv8n:

* pooling has padding at all, and a padded tap takes the descriptor's
  `pad_value` (the tensor's quantization zero-point), not 0 -- see
  `cnn_accel_model._pool_windows`;
* the pool kernel bound is 5, separate from the convolution bound of 3 --
  the SPPF block's 5x5/stride-1/pad-2 max pool.

Both are checked end to end the same way every other top-level case is:
the DUT's own DDR writeback is compared against `reference.py`'s
execution of the same program, byte for byte.
"""

from __future__ import annotations

from accel_v2.model import Model
from accel_v2.tbcase import TbCase, build_case

# Same small shapes as `cases.py`: these are integration tests, and the
# pooling arithmetic itself is pinned bit-exactly by `tb_cnn_accel_pool`.
_H = 8
_W = 8
_C = 8


def case_pool_max_padded_zero_point() -> TbCase:
    """3x3/1 max pooling with 'same' padding and a zero-point pad value.

    The minimal case that separates "pooling has padding" from "padding
    is filled with the zero-point": at `pad_value = -128` every border
    window's padded taps are the smallest int8 there is, so they can
    never win the max, whereas the old zero fill would have won it
    wherever the real activations are negative.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        # CLAMP the conv output into the all-negative range, so that the
        # pad value is *observable*: with every real activation below
        # -40, a zero-filled pad tap wins every border window's max and a
        # -128 one never does. Without this the case would silently pass
        # against a hardware that ignored pad_value entirely -- a random
        # int8 tensor almost always has a real tap above 0 in every 3x3
        # window, which is exactly the trap this comment exists to stop
        # the next person falling into (it caught this case's first draft
        # under a deliberate RTL mutation).
        h = model.conv2d(
            x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), clamp=(-128, -40), name="h"
        )
        y = model.pool_max(
            h, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_value=-128, name="y"
        )
        model.output(y)

    return build_case("pool_max_padded_zero_point", build, seed=101)


def case_pool_max_5x5_sppf() -> TbCase:
    """YOLOv8n's SPPF shape exactly: 5x5 max pool, stride 1, padding 2.

    5x5 is above the convolution datapath's `MAX_KERNEL_SIZE` of 3 and
    only the pool path is sized for it, so this case is what proves the
    separate pool kernel bound reaches all the way through `cmd_proc`'s
    geometry validation, the pool `window_gen` instance and the pool
    lanes. Stride 1 with pad 2 is shape-preserving, so the output is the
    same 8x8x8 tensor the input is.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        # Clamped all-negative, for the same reason as
        # `case_pool_max_padded_zero_point`: it makes the pad value
        # observable in the output instead of being masked by some real
        # tap that happens to be larger.
        h = model.conv2d(
            x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), clamp=(-128, -40), name="h"
        )
        y = model.pool_max(
            h, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="y"
        )
        model.output(y)

    return build_case("pool_max_5x5_sppf", build, seed=102)


def case_sppf_chain() -> TbCase:
    """The SPPF block's actual structure: the same 5x5/1/2 max pool
    applied three times in a row, all local, with the results consumed by
    a following op.

    SPPF pools its input repeatedly and concatenates; this repeats the
    pool three times and adds two of the results, which exercises the
    same thing that matters to the hardware -- back-to-back pooling
    commands with no conv in between, each reconfiguring the pool
    window generator -- without needing a concat the ISA does not have.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        stem = model.conv2d(
            x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), clamp=(-128, -40), name="stem"
        )
        p1 = model.pool_max(
            stem, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="p1"
        )
        p2 = model.pool_max(
            p1, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="p2"
        )
        p3 = model.pool_max(
            p2, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="p3"
        )
        y = model.add(p1, p3, name="y")
        model.output(y)

    return build_case("sppf_chain", build, seed=103)


def case_pool_avg_padded() -> TbCase:
    """Padded average pooling, count-include-pad.

    `POOL_AVG` sums the same padded window `POOL_MAX` maxes over and
    divides by a fixed `requant_scale`, so the padded taps are part of
    the sum. This case exists to make that documented semantics a tested
    one rather than an asserted one -- the reference model and the RTL
    must agree on it, whatever it is.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="h")
        y = model.pool_avg(
            h, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_value=-128, name="y"
        )
        model.output(y)

    return build_case("pool_avg_padded", build, seed=104)


#: Every case, in the order they are registered as VUnit configs.
CASE_BUILDERS = (
    case_pool_max_padded_zero_point,
    case_pool_max_5x5_sppf,
    case_sppf_chain,
    case_pool_avg_padded,
)


def all_cases() -> list[TbCase]:
    """Build every case. Each call builds fresh objects (each with its own
    `DdrMap`), so cases never share DDR allocation state."""
    return [builder() for builder in CASE_BUILDERS]


__all__ = ["CASE_BUILDERS", "all_cases"]
