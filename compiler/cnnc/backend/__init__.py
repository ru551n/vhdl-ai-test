"""Backend emitters: HIR (planned) -> accelerator program (doc/tosa_compiler_plan.md §2.3, §13 M8).

One subpackage per target `program_model`; today only `cnn_accel_v1`
(layer-descriptor instruction stream) exists.
"""

from __future__ import annotations
