"""Hardware IR (HIR): the HOW (doc/tosa_compiler_plan.md §2.2/§7/§8)."""

from .ir import (
    KIND_TO_OP,
    LAYOUTS,
    ROLES,
    STAGES,
    Buffer,
    BufferView,
    HirModule,
    HirOp,
    TileInfo,
)
from .printer import print_hir, to_json
from .verify import HirVerifyError, MemoryPlanError, verify_hir

__all__ = [
    "KIND_TO_OP",
    "LAYOUTS",
    "ROLES",
    "STAGES",
    "Buffer",
    "BufferView",
    "HirModule",
    "HirOp",
    "TileInfo",
    "HirVerifyError",
    "MemoryPlanError",
    "verify_hir",
    "print_hir",
    "to_json",
]
