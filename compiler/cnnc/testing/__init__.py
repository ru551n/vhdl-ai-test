"""`cnnc.testing`: helpers used only by the test suite (doc/tosa_compiler_plan.md
§10 item 3, the IREE external oracle). `numpy` is used here and in
`cnnc.gir.interp` only."""

from cnnc.testing.iree_oracle import iree_available, run_iree

__all__ = [
    "iree_available",
    "run_iree",
]
