"""Typed exception hierarchy shared by every compiler stage.

Every diagnostic can carry the GIR/HIR op id it applies to and the
pipeline stage that raised it, so a failing test/CLI run says *which*
invariant and *which* op (doc/tosa_compiler_plan.md §11).
"""

from __future__ import annotations


class CompilerError(Exception):
    """Base class for all typed compiler diagnostics.

    `op_id` is the offending op's id (e.g. ``"%4"``); `stage` is the
    pipeline stage name (e.g. ``"import"``, ``"verify"``). When both are
    set the message is prefixed ``"[stage] op <id>: "``.
    """

    def __init__(self, message: str, *, op_id: str | None = None, stage: str | None = None):
        self.op_id = op_id
        self.stage = stage
        if stage is not None and op_id is not None:
            prefix = f"[{stage}] op {op_id}: "
        elif stage is not None:
            prefix = f"[{stage}]: "
        elif op_id is not None:
            prefix = f"op {op_id}: "
        else:
            prefix = ""
        super().__init__(f"{prefix}{message}")


class TosaImportError(CompilerError):
    """Raised by `frontend.tosa_import` for TOSA IR the importer cannot
    translate into GIR: unsupported constructs, invalid attributes,
    declared/computed shape mismatches, etc."""


class UnsupportedOp(TosaImportError):
    """A dialect op has no importer support (e.g. `tosa.add` in M2)."""


class UnsupportedAttribute(TosaImportError):
    """An op was recognised, but one of its attributes/operands falls
    outside the subset this milestone's importer implements (e.g.
    `dilation != (1, 1)`, `scale32 = false`)."""


class VerifyError(CompilerError):
    """Raised by `gir.verify.verify` when a `Graph` violates a GIR
    invariant."""


class CapabilityError(CompilerError):
    """Raised by `lower.to_hir` (M6) when a GIR op cannot be lowered onto
    the target's advertised capabilities: unit selection, dtype/kernel/
    stride/dilation admissibility, rescale/clamp epilogue limits,
    `Unit.constraints` violations, or a target-wide gate such as the
    rounding mode. `unit` is the offending `Target.Unit.name` (`None` for
    checks that are not unit-specific); `constraint` is the capability
    name or constraint expression that failed (`None` if not applicable).

    Distinct from `cnnc.target.contract.CapabilityError` (a plain
    `Exception` reserved for target/backend-internal concerns); this one
    is a `CompilerError` so it carries `op_id`/`stage` like every other
    compiler diagnostic.
    """

    def __init__(
        self,
        message: str,
        *,
        op_id: str | None = None,
        stage: str | None = None,
        unit: str | None = None,
        constraint: str | None = None,
    ) -> None:
        self.unit = unit
        self.constraint = constraint
        super().__init__(message, op_id=op_id, stage=stage)


class LegalizeError(CompilerError):
    """Raised by `passes.legalize_rescale.LegalizeRescalePass` (M4) when a
    `rescale` cannot be rewritten onto the target's rescale capability:
    multiplier overflow after the `shift < shift_min` rewrite, `shift`
    left above `shift_max`, or a rounding mode (`DOUBLE_ROUND`) with no
    target-expressible equivalent."""
