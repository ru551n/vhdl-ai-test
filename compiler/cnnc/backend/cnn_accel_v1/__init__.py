"""`cnn_accel_v1` backend: HIR (stage='planned') -> `Program` (`emit.py`),
program decode + dump (`decode.py`), and program execution against the
`cnn_accel_model` golden model (`run.py`). See doc/tosa_compiler_plan.md
§2.3, §5, §10 item 2, §13 M8.
"""

from __future__ import annotations

from .decode import decode_program, print_program
from .emit import Descriptor, Program, emit_program
from .run import run_program
from .vectors import VectorsResult, write_conv_core_vectors

__all__ = [
    "Descriptor",
    "Program",
    "emit_program",
    "decode_program",
    "print_program",
    "run_program",
    "VectorsResult",
    "write_conv_core_vectors",
]
