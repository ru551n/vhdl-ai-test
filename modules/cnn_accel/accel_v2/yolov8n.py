"""YOLOv8n at its real size, as a `Model` -- shapes only.

The tiling design's traffic tables (sections 4.3 and 5) are statements
about *this* network at 640x640, and the only honest way to check that
the selection rule reproduces them is to run the rule on the network
itself: 63 convolutions, 4.371 GMAC, 3.17 MB of weights.

Built with `Model(shapes_only=True)`, so no weight, bias or input values
are generated. Everything the planner and the tiler consult -- shapes,
kernels, strides, padding, channel counts, the alias structure of every
`split`/`concat` -- is exact; only the numbers are absent. Do not hand
this model to `reference.py` or `program.py`.

The topology is `ultralytics`' `yolov8n.yaml` as the gap analysis
enumerates it: backbone `Conv/C2f` stages at 640->320->160->80->40->20,
SPPF, an FPN/PAN neck with two upsample merges and two stride-2
downsample merges, and a three-scale Detect head whose two branches per
scale end in graph outputs (the softmax/decode runs on the host).
"""

from __future__ import annotations

from accel_v2.model import Activation, Model, Tensor

#: `(top, bottom, left, right)` for a shape-preserving 3x3 convolution.
_PAD3 = (1, 1, 1, 1)


def _conv(model: Model, x: Tensor, out_channels: int, *, k: int = 3, stride: int = 1, name: str):
    padding = _PAD3 if k == 3 else (0, 0, 0, 0)
    return model.conv2d(
        x,
        out_channels,
        kernel=(k, k),
        stride=(stride, stride),
        padding=padding,
        activation=Activation.RELU,
        name=name,
    )


def _bottleneck(model: Model, x: Tensor, channels: int, *, shortcut: bool, prefix: str) -> Tensor:
    h = _conv(model, x, channels, name=f"{prefix}_cv1")
    h = _conv(model, h, channels, name=f"{prefix}_cv2")
    return model.add(h, x, name=f"{prefix}_add") if shortcut else h


def _c2f(model: Model, x: Tensor, c2: int, *, n: int, shortcut: bool, prefix: str) -> Tensor:
    c = c2 // 2
    cv1 = _conv(model, x, 2 * c, k=1, name=f"{prefix}_cv1")
    a, b = model.split(cv1, [c, c], names=[f"{prefix}_a", f"{prefix}_b"])
    branches = [a, b]
    tail = b
    for i in range(n):
        tail = _bottleneck(model, tail, c, shortcut=shortcut, prefix=f"{prefix}_m{i}")
        branches.append(tail)
    cat = model.concat(branches, name=f"{prefix}_cat")
    return _conv(model, cat, c2, k=1, name=f"{prefix}_cv2")


def _sppf(model: Model, x: Tensor, c2: int, *, prefix: str) -> Tensor:
    c = x.channels // 2
    cv1 = _conv(model, x, c, k=1, name=f"{prefix}_cv1")
    parts = [cv1]
    src = cv1
    for i in range(3):
        src = model.pool_max(
            src, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128,
            name=f"{prefix}_p{i + 1}",
        )
        parts.append(src)
    return _conv(model, model.concat(parts, name=f"{prefix}_cat"), c2, k=1, name=f"{prefix}_cv2")


def _detect(model: Model, x: Tensor, *, prefix: str) -> list[Tensor]:
    """One Detect scale: a 64-channel box branch and an 80-channel class
    branch, `3x3 -> 3x3 -> 1x1` each, both graph outputs. 80 channels is
    what makes R6 (a final channel tile with padding lanes) real."""
    outputs = []
    for branch, width in (("cv2", 64), ("cv3", 80)):
        h = _conv(model, x, width, name=f"{prefix}_{branch}_0")
        h = _conv(model, h, width, name=f"{prefix}_{branch}_1")
        outputs.append(model.output(_conv(model, h, width, k=1, name=f"{prefix}_{branch}_2")))
    return outputs


def build(seed: int = 0) -> Model:
    """The whole network. `shapes_only`: planner and tiler input, never
    an executable program."""
    model = Model(seed=seed, name="yolov8n", shapes_only=True)
    image = model.input(640, 640, 3, name="image")

    l0 = _conv(model, image, 16, stride=2, name="L0")  # 320
    l1 = _conv(model, l0, 32, stride=2, name="L1")  # 160
    l2 = _c2f(model, l1, 32, n=1, shortcut=True, prefix="L2")
    l3 = _conv(model, l2, 64, stride=2, name="L3")  # 80
    l4 = _c2f(model, l3, 64, n=2, shortcut=True, prefix="L4")
    l5 = _conv(model, l4, 128, stride=2, name="L5")  # 40
    l6 = _c2f(model, l5, 128, n=2, shortcut=True, prefix="L6")
    l7 = _conv(model, l6, 256, stride=2, name="L7")  # 20
    l8 = _c2f(model, l7, 256, n=1, shortcut=True, prefix="L8")
    l9 = _sppf(model, l8, 256, prefix="L9")

    up10 = model.upsample2x(l9, name="L10_up")  # 40
    cat11 = model.concat([up10, l6], name="L11_cat")
    l12 = _c2f(model, cat11, 128, n=1, shortcut=False, prefix="L12")

    up13 = model.upsample2x(l12, name="L13_up")  # 80
    cat14 = model.concat([up13, l4], name="L14_cat")
    l15 = _c2f(model, cat14, 64, n=1, shortcut=False, prefix="L15")

    l16 = _conv(model, l15, 64, stride=2, name="L16")  # 40
    cat17 = model.concat([l16, l12], name="L17_cat")
    l18 = _c2f(model, cat17, 128, n=1, shortcut=False, prefix="L18")

    l19 = _conv(model, l18, 128, stride=2, name="L19")  # 20
    cat20 = model.concat([l19, l9], name="L20_cat")
    l21 = _c2f(model, cat20, 256, n=1, shortcut=False, prefix="L21")

    _detect(model, l15, prefix="D0")
    _detect(model, l18, prefix="D1")
    _detect(model, l21, prefix="D2")
    return model


def conv_count(model: Model) -> int:
    from accel_v2.model import Conv2dOp

    return sum(1 for op in model.ops if isinstance(op, Conv2dOp))


def total_macs(model: Model) -> int:
    from accel_v2.model import Conv2dOp

    return sum(
        op.output.height * op.output.width * op.output.channels
        * op.inputs[0].channels * op.kernel[0] * op.kernel[1]
        for op in model.ops
        if isinstance(op, Conv2dOp)
    )


__all__ = ["build", "conv_count", "total_macs"]
