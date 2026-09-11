"""VUnit python_bridge entry point for the DEPTH_TO_SPACE live-checker
pilot (tb_cnn_accel_elementwise_depth_to_space_pyffi.vhd).

Loaded into the simulator's embedded Python interpreter via
`python_execute(file_name => ...)`; called from VHDL via `python_call`.
This is the SAME `cnn_accel_model.depth_to_space` function every other
opcode's golden-model cross-check already treats as the single source of
truth (see that module's own anti-fork-rule docstring) -- the pilot's
whole point is that the RTL testbench can now call it live, instead of a
Python-generated file being loaded by VHDL and compared after the fact.
"""

import sys
from pathlib import Path

# modules/cnn_accel/ (two levels up from this file: test/python_bridge/).
_CNN_ACCEL_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CNN_ACCEL_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_ACCEL_DIR))

import numpy as np

from cnn_accel_model import (
    LayerDesc,
    OPCODE_DEPTH_TO_SPACE,
    depth_to_space,
    pack_activation_planes,
    unpack_activation_planes,
)


def depth_to_space_check(packed_bytes, in_width, in_height, in_channels, out_channels, dts_factor):
    """`packed_bytes`: a VUnit `integer_array_t` (NumPy array) of the RAW
    decision-S6 channel-tiled-plane DDR bytes -- exactly what the
    testbench's own 'src0_mem'/'dst_mem' hold, and exactly what the RTL
    actually reads/writes. NOT the logical HWC tensor
    `cnn_accel_model.depth_to_space` itself operates on -- unpacked here
    first, then repacked on the way out, the same two calls
    `cnn_accel_model.run_layer` wraps every opcode in. Returns the
    expected OUTPUT bytes in that same packed domain, so the testbench
    can compare byte-for-byte against what it captured with no VHDL-side
    tiling math of its own."""
    factor = int(dts_factor)
    in_width, in_height, in_channels, out_channels = (
        int(in_width),
        int(in_height),
        int(in_channels),
        int(out_channels),
    )
    packed_in = [(int(b) + 256) if int(b) < 0 else int(b) for b in packed_bytes]
    hwc_in = unpack_activation_planes(packed_in, in_width, in_height, in_channels)

    desc = LayerDesc(
        opcode=OPCODE_DEPTH_TO_SPACE,
        in_width=in_width,
        in_height=in_height,
        in_channels=in_channels,
        out_channels=out_channels,
        dts_factor=factor,
    )
    hwc_out = depth_to_space(hwc_in, desc, factor=factor)

    out_width, out_height = in_width * factor, in_height * factor
    packed_out = pack_activation_planes(hwc_out, out_width, out_height, out_channels)
    return np.array([(b - 256) if b >= 128 else b for b in packed_out], dtype=np.int8)
