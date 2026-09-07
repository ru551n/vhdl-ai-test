"""GIR passes (doc/tosa_compiler_plan.md §11 M4/M5): `normalize`,
`legalize_rescale`, `fuse`, and the shared pass framework."""

from __future__ import annotations

from cnnc.target.contract import Target

from .fuse import FusePass
from .framework import Pass, PassContext, run_pipeline
from .legalize_rescale import LegalizeRescalePass
from .normalize import NormalizePass

__all__ = [
    "Pass",
    "PassContext",
    "run_pipeline",
    "NormalizePass",
    "LegalizeRescalePass",
    "FusePass",
    "default_pipeline",
]


def default_pipeline(target: Target | None) -> list[Pass]:
    """The M4/M5 GIR pipeline, in order: normalize, legalize_rescale, fuse.

    `target` is accepted (rather than a fixed no-arg pipeline) so future
    milestones can vary pass selection/configuration per target; today
    every pass takes its target from `PassContext.target` at `run()` time,
    so this list is the same regardless of `target`.
    """
    del target
    return [NormalizePass(), LegalizeRescalePass(), FusePass()]
