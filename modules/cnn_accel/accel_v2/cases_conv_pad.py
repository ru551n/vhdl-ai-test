"""`tb_cnn_accel_top` cases for quantization-correct CONVOLUTION padding.

A catalogue alongside `cases.py`, registered by `module_cnn_accel.py` in
the same loop and following exactly the same contract (one function per
case, each returning a `TbCase`, each naming its own seed). It is a
separate file only so the two can be edited independently.

What is under test here is one thing: a convolution fills its padded taps
with the descriptor's ISA v2.1 `pad_value` -- the input tensor's
quantization zero-point -- and not with a literal 0. Padding a quantized
int8 tensor with 0 is not "adding nothing": 0 is the real value
`(0 - zero_point) * scale`, so every padded tap contributes
`w * (0 - zero_point)` to the accumulator, i.e. about
`sum(w) * zero_point * scale` of pure bias on every border output.
YOLOv8n convolves 3x3 with padding 1 in essentially every layer, so that
error lands on the border of every feature map in the network.

The same field reached the POOL path one commit earlier (4915955); these
cases are the convolution half, checked end to end the same way every
other top-level case is -- the DUT's own DDR writeback compared against
`reference.py`'s execution of the same program, byte for byte.

THE TRAP, inherited from the pool work and re-hit here. With ordinary
random int8 activations and weights, a wrong pad value very often does
NOT change the final int8 output: the accumulator difference is
requantized and clamped away, and the case then passes against hardware
that ignores `pad_value` entirely. Every case below therefore

  * drives the padded convolution with an ALL-NEGATIVE input (clamped
    into a narrow band near the -128 zero-point, which is what a real
    quantized ReLU-style activation tensor looks like), and
  * picks a requant scale that lands the result in the middle of the
    int8 range rather than on a rail,

so the injected bias is both large and visible. Each case was then
mutation-tested: with `cnn_accel_conv_core`'s `cfg_pad_value` tied back
to zero, each of these FAILS on a data mismatch, and passes again once it
is restored. `test_cases_conv_pad.py` keeps that property from rotting by
asserting it at the reference level.
"""

from __future__ import annotations

from accel_v2.model import Activation, Model
from accel_v2.tbcase import TbCase, build_case

# Same small shapes as `cases.py` and `cases_pool_pad.py`: these are
# integration tests, and the convolution arithmetic itself is pinned
# bit-exactly by `tb_cnn_accel_conv_core`.
_H = 8
_W = 8
_C = 8

#: The int8 zero-point of a realistically quantized activation tensor
#: (YOLOv8n's activations quantize with `zero_point = -128`).
_ZP = -128

#: The band a "quantized activation" is clamped into: entirely below
#: zero, so a 0-valued pad tap is nowhere near any real value in the
#: tensor -- precisely the situation zero padding gets wrong, and
#: precisely the situation random int8 data never produces. Same trick,
#: and the same band, as `cases_pool_pad.py`.
#:
#: The band is deliberately WIDE (88 of the 256 int8 codes) rather than a
#: tight cluster at the zero-point. A tight band would flatten a clamped
#: intermediate into a constant, and a constant plane cannot carry the
#: effect of a wrong pad value forward into the next op -- which is how
#: `case_conv_pad_default_stays_zero` first failed to be sensitive at
#: all, caught by `tests/test_conv_pad.py`.
_ACTIVATION_CLAMP = (-128, -40)

#: Requant scale/shift for the padded convolutions below. A 3x3x8 window
#: of values near -128 against random int8 weights accumulates into the
#: tens of thousands, so it needs shifting down by ~2**9 to sit inside
#: int8 instead of saturating; `1 << 15` is Q15 unity, so this is exactly
#: `1/512`. Passed explicitly rather than derived from `QuantParams`,
#: because the whole point is to control where the result lands.
_REQUANT = {"requant_scale": 1 << 15, "requant_shift": 9}


def _activations(model: Model, name: str):
    """A `_H x _W x _C` tensor whose values all sit in `_ACTIVATION_CLAMP`.

    Produced by a real clamped convolution rather than by handing the
    model a crafted input buffer: the input tensor of a case is filled by
    the seeded RNG, and clamping the FIRST op's output is how
    `cases_pool_pad.py` solves the same problem. That first convolution
    is unpadded, so it is not itself under test here.
    """
    x = model.input(_H, _W, _C, name=f"{name}_x")
    return model.conv2d(
        x,
        _C,
        kernel=(1, 1),
        padding=(0, 0, 0, 0),
        activation=Activation.NONE,
        clamp=_ACTIVATION_CLAMP,
        name=name,
    )


def case_conv_pad_zero_point() -> TbCase:
    """The minimal case: one 3x3 / stride-1 / pad-1 convolution over a
    tensor whose zero-point is -128, with `pad_value = -128`.

    This is YOLOv8n's convolution shape, and the difference between
    padding with 0 and padding with the zero-point is a bias of
    `sum(w) * 128` on every border output -- most of the 8x8 plane.
    """

    def build(model: Model) -> None:
        h = _activations(model, "h")
        y = model.conv2d(
            h,
            _C,
            kernel=(3, 3),
            padding=(1, 1, 1, 1),
            pad_value=_ZP,
            activation=Activation.NONE,
            name="y",
            **_REQUANT,
        )
        model.output(y)

    return build_case("conv_pad_zero_point", build, seed=201)


def case_conv_pad_asymmetric() -> TbCase:
    """A positive pad value with all four pad counts different.

    `pad_top`/`pad_bottom`/`pad_left`/`pad_right` are four independent
    ISA fields, and the fill is a single value shared by all of them. A
    fill that only reached the top/left taps (the ones the window
    generator writes first) would still pass a symmetric case.
    """

    def build(model: Model) -> None:
        h = _activations(model, "h")
        y = model.conv2d(
            h,
            _C,
            kernel=(3, 3),
            padding=(1, 0, 0, 1),
            pad_value=100,
            activation=Activation.NONE,
            name="y",
            **_REQUANT,
        )
        model.output(y)

    return build_case("conv_pad_asymmetric", build, seed=202)


def case_conv_pad_value_per_descriptor() -> TbCase:
    """Two padded convolutions back to back with DIFFERENT pad values,
    the second reading the first's local output.

    `cnn_accel_window_gen` latches `cfg_pad_value` at `start`, once per
    command. A value that was latched but never re-latched -- or one held
    in a CSR instead of the descriptor -- gives the second convolution
    the first's fill and fails here, while a single-conv case cannot tell
    the difference.
    """

    def build(model: Model) -> None:
        h = _activations(model, "h")
        a = model.conv2d(
            h,
            _C,
            kernel=(3, 3),
            padding=(1, 1, 1, 1),
            pad_value=_ZP,
            activation=Activation.NONE,
            clamp=_ACTIVATION_CLAMP,
            name="a",
            **_REQUANT,
        )
        y = model.conv2d(
            a,
            _C,
            kernel=(3, 3),
            padding=(1, 1, 1, 1),
            pad_value=64,
            activation=Activation.NONE,
            name="y",
            **_REQUANT,
        )
        model.output(y)

    return build_case("conv_pad_value_per_descriptor", build, seed=203)


def case_conv_pad_default_stays_zero() -> TbCase:
    """A padded convolution that does NOT set `pad_value`, immediately
    after one that does.

    The compatibility half of the change: `pad_value` defaults to 0 and a
    descriptor that leaves it there must zero-pad exactly as every
    pre-v2.1 program did. Placed second so it also proves the previous
    command's non-zero fill does not leak forward -- the opposite
    direction of `case_conv_pad_value_per_descriptor`.
    """

    def build(model: Model) -> None:
        h = _activations(model, "h")
        a = model.conv2d(
            h,
            _C,
            kernel=(3, 3),
            padding=(1, 1, 1, 1),
            pad_value=_ZP,
            activation=Activation.NONE,
            clamp=_ACTIVATION_CLAMP,
            name="a",
            **_REQUANT,
        )
        y = model.conv2d(
            a,
            _C,
            kernel=(3, 3),
            padding=(1, 1, 1, 1),
            activation=Activation.NONE,
            name="y",
            **_REQUANT,
        )
        model.output(y)

    return build_case("conv_pad_default_stays_zero", build, seed=204)


def case_conv_and_pool_pad_values_differ() -> TbCase:
    """A padded convolution and a padded max pool in the same program,
    with DIFFERENT pad values.

    The conv and the pool have their own `cnn_accel_window_gen` instances
    and their own `cfg_pad_value` wires, fed from separate `cmd_proc`
    outputs. This is the case a shared value -- one signal driving both
    generators, or `conv_cfg_pad_value` accidentally wired to
    `pool_cfg_pad_value` -- cannot satisfy: the conv pads with -128 and
    the pool with +80, and either engine taking the other's value changes
    its result.
    """

    def build(model: Model) -> None:
        h = _activations(model, "h")
        a = model.conv2d(
            h,
            _C,
            kernel=(3, 3),
            padding=(1, 1, 1, 1),
            pad_value=_ZP,
            activation=Activation.NONE,
            clamp=_ACTIVATION_CLAMP,
            name="a",
            **_REQUANT,
        )
        y = model.pool_max(
            a, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_value=80, name="y"
        )
        model.output(y)

    return build_case("conv_and_pool_pad_values_differ", build, seed=205)


def case_conv_pad_stride2_zero_point() -> TbCase:
    """A padded, strided convolution with a zero-point fill.

    Stride 2 with pad 1 makes the bottom/right windows exist only
    *because* of the padding, so their padded taps outnumber their real
    ones. It is also the shape a downsampling YOLOv8n stage uses.
    """

    def build(model: Model) -> None:
        h = _activations(model, "h")
        y = model.conv2d(
            h,
            _C,
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1, 1, 1),
            pad_value=_ZP,
            activation=Activation.NONE,
            name="y",
            **_REQUANT,
        )
        model.output(y)

    return build_case("conv_pad_stride2_zero_point", build, seed=206)


#: Every case, in the order they are registered as VUnit configs.
CASE_BUILDERS = (
    case_conv_pad_zero_point,
    case_conv_pad_asymmetric,
    case_conv_pad_value_per_descriptor,
    case_conv_pad_default_stays_zero,
    case_conv_and_pool_pad_values_differ,
    case_conv_pad_stride2_zero_point,
)


def all_cases() -> list[TbCase]:
    """Build every case. Each call builds fresh objects (each with its own
    `DdrMap`), so cases never share DDR allocation state."""
    return [builder() for builder in CASE_BUILDERS]


__all__ = ["CASE_BUILDERS", "all_cases"]
