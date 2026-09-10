"""GIR passes (doc/tosa_compiler_plan.md §11 M4/M5): `normalize`,
`legalize_rescale`, `fuse`, and the shared pass framework."""

from __future__ import annotations

from cnnc.target.contract import Target

from .depth_to_space_channels import PermuteDepthToSpaceChannelsPass
from .fuse import FusePass
from .framework import Pass, PassContext, run_pipeline
from .legalize_rescale import LegalizeRescalePass
from .normalize import NormalizePass

__all__ = [
    "Pass",
    "PassContext",
    "run_pipeline",
    "NormalizePass",
    "PermuteDepthToSpaceChannelsPass",
    "LegalizeRescalePass",
    "FusePass",
    "default_pipeline",
]


def default_pipeline(target: Target | None) -> list[Pass]:
    """The M4/M5 GIR pipeline, in order: depth_to_space_channels,
    normalize, legalize_rescale, fuse.

    `PermuteDepthToSpaceChannelsPass` runs FIRST, on the graph exactly as
    imported. It rewrites the constants of the convolution that produces a
    pixel shuffle's input, and it is far simpler to find that convolution
    while it is still a plain `conv2d` -> `rescale` -> `clamp` sequence
    than after `FusePass` has folded the three into a `fused_conv` (it
    handles both, but only the first shape is guaranteed to exist).
    Running before `NormalizePass` also means no clamp it would have to
    look through has been deleted out from under it.

    `target` is accepted (rather than a fixed no-arg pipeline) so future
    milestones can vary pass selection/configuration per target; today
    every pass takes its target from `PassContext.target` at `run()` time,
    so this list is the same regardless of `target`.
    """
    del target
    return [
        PermuteDepthToSpaceChannelsPass(),
        NormalizePass(),
        LegalizeRescalePass(),
        FusePass(),
    ]
