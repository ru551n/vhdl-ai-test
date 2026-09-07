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


# Later milestones add further typed errors on top of `CompilerError`
# (e.g. `CapabilityError` for HIR capability checks -- already present in
# `cnnc.target.contract` for target/backend concerns; schedule/memplan
# errors for M7). Not added here: out of scope for M2.
