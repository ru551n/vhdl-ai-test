"""`cnnc.lower.layout` vs the golden model's own S6/D10/D11 pack functions.

The compiler builds DDR byte images from `Target` tiling parameters alone;
`cnn_accel_model` builds the same images from its own constants. Any drift
between the two would make every emitted program feed the model (and the
RTL) a misordered byte image, so pin them against each other here, over
partial tiles (Cin=3, Cout=12) as well as exact ones.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cnnc.lower.layout import (
    activation_bytes,
    pack_activation_planes,
    pack_bias_tiled,
    pack_weights_tiled,
    packed_bias_bytes,
    packed_weight_bytes,
    unpack_activation_planes,
)


def _load(root: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model(target):
    root = Path(target.provenance["accel_root"])
    _load(root, "cnn_accel_constants")
    return _load(root, "cnn_accel_model")


@pytest.fixture(scope="module")
def tiling(target):
    unit = target.units[0]
    return target.memory.activation_plane_channels, unit.internal_tiling.cin, unit.internal_tiling.cout


@pytest.mark.parametrize("shape", [(1, 8, 8, 4), (1, 5, 7, 3), (1, 4, 4, 8), (1, 3, 2, 17)])
def test_activation_planes_match_model(model, tiling, shape):
    plane_channels, _, _ = tiling
    rng = np.random.default_rng(sum(shape))
    arr = rng.integers(-128, 128, size=shape).astype(np.int8)
    _n, h, w, c = shape

    ours = pack_activation_planes(arr, plane_channels)
    theirs = model.pack_activation_planes([int(v) for v in arr.reshape(-1)], w, h, c)
    assert len(ours) == activation_bytes(shape, plane_channels) == model.activation_bytes(w, h, c)
    assert list(np.frombuffer(ours, dtype=np.int8)) == theirs

    back = unpack_activation_planes(ours, shape, np.int8, plane_channels)
    assert back.shape == shape and np.array_equal(back, arr)


@pytest.mark.parametrize("shape", [(8, 3, 3, 4), (12, 3, 3, 3), (8, 1, 1, 16), (20, 2, 3, 9)])
def test_weights_tiled_match_model(model, tiling, shape):
    _, tile_channels, pe_rows = tiling
    oc, kh, kw, ic = shape
    rng = np.random.default_rng(oc * ic)
    values = tuple(int(v) for v in rng.integers(-128, 128, size=oc * kh * kw * ic))
    desc = model.LayerDesc(
        opcode=model.OPCODE_CONV2D, in_channels=ic, out_channels=oc, kernel_h=kh, kernel_w=kw
    )

    ours = pack_weights_tiled(values, shape, tile_channels, pe_rows)
    theirs = model.pack_weights_for_hw(list(values), desc, tile_channels, pe_rows)
    assert len(ours) == packed_weight_bytes(shape, tile_channels, pe_rows)
    assert len(ours) == model.packed_weight_count(desc, tile_channels, pe_rows)
    assert list(np.frombuffer(ours, dtype=np.int8)) == theirs
    assert model.unpack_weights_from_hw(theirs, desc, tile_channels, pe_rows) == list(values)


@pytest.mark.parametrize("oc", [8, 3, 12, 17])
def test_bias_tiled_matches_model(model, tiling, oc):
    _, _, pe_rows = tiling
    rng = np.random.default_rng(oc)
    values = tuple(int(v) for v in rng.integers(-(2**31), 2**31, size=oc))
    desc = model.LayerDesc(opcode=model.OPCODE_CONV2D, out_channels=oc)

    ours = pack_bias_tiled(values, pe_rows)
    theirs = model.pack_bias_for_hw(list(values), desc, pe_rows)
    assert len(ours) == packed_bias_bytes(oc, pe_rows) == 4 * model.packed_bias_count(desc, pe_rows)
    assert list(np.frombuffer(ours, dtype="<i4")) == theirs


def test_model_plane_constant_is_what_target_discovered(model, tiling):
    plane_channels, tile_channels, pe_rows = tiling
    assert plane_channels == model.ACTIVATION_PLANE_CHANNELS
    assert tile_channels == model.TILE_CHANNELS
    assert pe_rows == model.PE_ROWS
