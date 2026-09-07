"""`cnnc.gir`: Graph IR -- the WHAT (doc/tosa_compiler_plan.md §2.1)."""

from cnnc.gir.interp import AccumulatorOverflow, apply_scale_32, evaluate_all, rescale_array, run
from cnnc.gir.ir import (
    DTYPES,
    ClampAttrs,
    ConvAttrs,
    FusedConvAttrs,
    Graph,
    Op,
    RescaleParams,
    Tensor,
    conv2d_output_shape,
    dtype_range,
)
from cnnc.gir.printer import print_gir, to_json
from cnnc.gir.verify import verify

__all__ = [
    "DTYPES",
    "AccumulatorOverflow",
    "ClampAttrs",
    "ConvAttrs",
    "FusedConvAttrs",
    "Graph",
    "Op",
    "RescaleParams",
    "Tensor",
    "apply_scale_32",
    "conv2d_output_shape",
    "dtype_range",
    "evaluate_all",
    "print_gir",
    "rescale_array",
    "run",
    "to_json",
    "verify",
]
