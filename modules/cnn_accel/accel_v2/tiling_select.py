"""Choosing the strip height, and what to do when nothing fits
(design sections 4.2 and 7, implementation plan step 6).

`tiler.py` rewrites a group into `S` strips for a *given* `S`. This
module decides `S`. The rule (section 4.2) is an exhaustive search with
no tuning knob:

    for every strip height S, and weights resident or not:
        build the strip sub-program and PLAN IT with the real Planner
        reject it if any group-internal buffer fell back to DDR
        reject it if the recomputed halos exceed the cap (25 % of MACs)
        keep the (S, residency) with the least DDR traffic; ties -> fewer strips
    if nothing survives: dissolve the group into single-op groups

Two things about that are deliberate and worth keeping.

**The fit test is a real plan, not an estimate.** The question "does a
strip of `R` rows fit in the scratchpad?" is answered by the same
first-fit, bank-confining, evicting allocator that will place the
buffers, so there is no estimate-versus-reality gap to be wrong about --
in particular no fragmentation margin to guess. Only *one* strip is
planned per candidate (`tiler.probe_strip`): every strip of a group is
the same sub-program on the same buffer sizes, and the last is merely
shorter.

**The search is over distinct strip heights, not over `1..H`.** `S` and
`R = ceil(H/S)` are the same statement, and `H` values of `S` produce
only about `2*sqrt(H)` distinct `R`s -- 49 of them at `H = 640`, not 640.
Enumerating `R` instead is the identical search with the duplicates
removed, and it is what makes the rule affordable on a real network.

Weight residency
----------------

`weights_resident` means the group's packed weight images are LOADed
into the scratchpad once and every strip's convolutions fetch their tiles
from `space_wgt = LOCAL_TENSOR` rather than re-reading them from DDR per
strip. The whole path is built and simulated: `Conv2dOp.weights_resident`
carries the request, `planner.py` places the images and emits the
`ConstLoadStep`, `program.py` emits the `LOAD`s and the
`space_wgt = LOCAL_TENSOR` descriptor, and `cases_tiling.py` runs it on
the DUT. `allow_resident_weights` therefore defaults to **True**, and
`Selection.resident_groups` is the set `tiler.tile` should be handed
alongside `strips` and `groups`.

The one thing residency is still modelled rather than planned is the fit
test: `_probe` shrinks the probe's scratchpad by the image bytes instead
of allocating them, because a probe builds *one* strip while a real
image is live across all `S` of them. If the final plan then cannot place
an image after all, `planner.py` falls back to DDR weights for it -- the
program stays correct and its predicted traffic stays honest (the charge
follows the decision actually made), it is merely the slower plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cnn_accel_model as golden

from accel_v2.model import Conv2dOp, Model, PoolOp, RowRange, Tensor, alias_root
from accel_v2.ddrmap import DdrMap
from accel_v2.planner import Planner
from accel_v2.tiler import (
    Group,
    TilingError,
    form_groups,
    dissolve,
    group_inputs,
    group_outputs,
    probe_strip,
    propagate_rows,
    strip_bounds,
)

#: Section 4.3's one policy constant: a group may recompute at most this
#: fraction of its own MACs in halo rows. It is where, on YOLOv8n, the
#: marginal megabyte saved per marginal GMAC of recompute falls below the
#: point at which bus time and array time trade evenly -- below it, more
#: strips cost more than they save. Raising it to 0.5 fuses three more
#: groups at 256 KiB and lands 8 MB lower for 12 % recompute; the default
#: stays at 25 % because that trade flips once compute scales ~3x.
DEFAULT_RECOMPUTE_CAP = 0.25

#: Bytes of DDR read for every 64-byte descriptor fetched.
_DESC_BYTES = 64


# ---------------------------------------------------------------------------
# Per-candidate cost (design section 4.1).
# ---------------------------------------------------------------------------


@dataclass
class GroupCost:
    """What one `(S, residency)` candidate costs a group."""

    strips: int
    rows: int  # R = ceil(H_out / S)
    weights_resident: bool
    ddr_read: int
    ddr_write: int
    weight_bytes: int
    recompute: float
    fits: bool
    #: Why it does not fit, for the message the fallback prints.
    reason: str = ""

    @property
    def ddr_total(self) -> int:
        return self.ddr_read + self.ddr_write


def _conv_macs_per_row(op: Conv2dOp) -> int:
    """MACs one output row of `op` costs."""
    return (
        op.output.width
        * op.output.channels
        * op.inputs[0].channels
        * op.kernel[0]
        * op.kernel[1]
    )


def group_macs(group: Group) -> int:
    return sum(
        _conv_macs_per_row(op) * op.output.height
        for op in group.ops
        if isinstance(op, Conv2dOp)
    )


def group_weight_bytes(group: Group) -> int:
    """Packed weight + bias + scale bytes for every distinct convolution
    in the group, using the same `cnn_accel_model` packers `planner.py`
    charges traffic with, so the two can never disagree."""
    total = 0
    for op in group.ops:
        if not isinstance(op, Conv2dOp):
            continue
        desc = op.weight_layer_desc()
        total += golden.packed_weight_count(desc, golden.TILE_CHANNELS, golden.PE_ROWS)
        if op.bias is not None:
            total += golden.packed_bias_count(desc, golden.PE_ROWS) * 4
        if op.per_channel_scale is not None:
            total += golden.packed_scale_table_bytes(desc, golden.PE_ROWS)
    return total


def _window_bytes(tensor: Tensor, rows: int) -> int:
    return rows * tensor.row_bytes * tensor.plane_count


def evaluate(
    model: Model,
    groups: list[Group],
    group: Group,
    strips: int,
    *,
    weights_resident: bool,
    planner: Planner,
    planewise_add: bool = True,
) -> GroupCost:
    """Cost and feasibility of cutting `group` into `strips` strips.

    Traffic is the closed form of section 4.1 evaluated on the *actual*
    row ranges the recurrence produces per strip -- not an approximation
    of them -- so a group input read is `bytes(X) + (S-1)*halo_X*hb_X`
    with the halo whatever the receptive fields really are, stride-2
    phase and clipping included. Feasibility is a real plan of one
    strip."""
    outs = group_outputs(model, groups, group)
    bounds = strip_bounds(group.height, min(strips, group.height))

    inputs = group_inputs(model, groups, group)
    read = 0
    write = 0
    recomputed_rows: dict[str, int] = {}
    #: The strip with the largest working set, which is the one whose fit
    #: decides the candidate. It is NOT the first: strip 0's halo is
    #: clipped by the top of the tensor and the last strip's by the
    #: bottom, so an *interior* strip materializes more rows of every
    #: tensor than either -- at `S = 3` on a 16-row C2f, 80 rows against
    #: strip 0's 73, which is the difference between two 8-plane buffers
    #: fitting a 16 KiB scratchpad and not. Probing strip 0 and calling
    #: the answer "the tallest strip" was wrong about `R` being the whole
    #: story, and the DUT found it: the plan fell back to DDR for two
    #: buffers the rule had already declared feasible.
    worst_rows: int = -1
    worst_bound = bounds[0]

    for out_rows in bounds:
        need = propagate_rows(group, out_rows, outs)
        total_rows = sum(rows.rows for rows in need.values())
        if total_rows > worst_rows:
            worst_rows, worst_bound = total_rows, out_rows
        for tensor in inputs:
            read += _window_bytes(tensor, need[tensor.name].rows)
        for tensor in outs:
            write += _window_bytes(tensor, out_rows.rows)
        for op in group.ops:
            if isinstance(op, Conv2dOp):
                recomputed_rows[op.name] = recomputed_rows.get(op.name, 0) + need[op.output.name].rows

    weight_bytes = group_weight_bytes(group)
    read += weight_bytes if weights_resident else weight_bytes * len(bounds)

    total_macs = group_macs(group)
    extra = sum(
        (rows - _op_by_name(group, name).output.height) * _conv_macs_per_row(_op_by_name(group, name))
        for name, rows in recomputed_rows.items()
    )
    recompute = extra / total_macs if total_macs else 0.0

    fits, reason, descriptors = _probe(
        model,
        groups,
        group,
        worst_bound,
        planner,
        planewise_add=planewise_add,
        # Resident weight images occupy the scratchpad for the whole
        # group, so the strip's own buffers only get what is left. Until
        # `space_wgt = LOCAL_TENSOR` is emitted (step 2) they cannot be
        # allocated for real, so the probe is given a correspondingly
        # smaller scratchpad -- which is the same question asked the same
        # way, one allocation short of asking it literally.
        reserved=weight_bytes if weights_resident else 0,
    )
    read += descriptors * len(bounds) * _DESC_BYTES

    return GroupCost(
        strips=len(bounds),
        rows=bounds[0].rows,
        weights_resident=weights_resident,
        ddr_read=read,
        ddr_write=write,
        weight_bytes=weight_bytes,
        recompute=recompute,
        fits=fits,
        reason=reason,
    )


def _op_by_name(group: Group, name: str):
    return next(op for op in group.ops if op.name == name)


def _probe(
    model: Model,
    groups: list[Group],
    group: Group,
    out_rows: RowRange,
    planner: Planner,
    *,
    planewise_add: bool,
    reserved: int = 0,
) -> tuple[bool, str, int]:
    """Plan one strip for real. Returns `(fits, reason, descriptors)`.

    `out_rows` is the strip with the largest working set, chosen by the
    caller from the row-range recurrence itself (see `evaluate`) rather
    than assumed to be the first. "Fits" means the plan put nothing but
    pinned group boundaries in DDR: a group-internal buffer that fell
    back (section 7 level 2) is exactly the strip height not being
    viable.

    One strip is still enough. Every strip is the same sub-program, only
    the row counts differ, and the buffers of one strip are all dead
    before the next begins -- so the strip that needs the most rows is
    the one that decides, and planning the other `S-1` would ask the same
    question with more slack."""
    from accel_v2.planner import descriptor_count

    capacity = planner.tensor_mem_bytes - reserved
    if capacity < planner.bank_bytes:
        return False, f"resident weights leave only {capacity} bytes of scratchpad", 0
    capacity -= capacity % planner.bank_bytes
    try:
        probe = probe_strip(
            model, group, out_rows, planewise_add=planewise_add, groups=groups
        )
        # A fresh `DdrMap` per probe, at the caller's scale. `DdrMap`'s
        # per-region bump allocators are stateful and never reset, and a
        # search runs thousands of throwaway probes; sharing one map
        # would walk it off the end of its regions and report "does not
        # fit" for a reason that has nothing to do with the scratchpad.
        # Probe addresses are discarded anyway -- only the *placements*
        # are the answer.
        planned = Planner(
            tensor_mem_bytes=capacity,
            bank_bytes=planner.bank_bytes,
            ddr_map=DdrMap(scale=planner.ddr_map.scale),
        ).plan(probe)
    except (TilingError, ValueError) as error:
        return False, str(error), 0
    descriptors = sum(descriptor_count(step) for step in planned.steps)
    if planned.ddr_placements:
        names = [name for name, _, _ in planned.ddr_placements]
        return False, f"buffer(s) {names} did not fit in the scratchpad", descriptors
    if planned.traffic.spill_count:
        # A spilled buffer is the same verdict reached one step later:
        # the strip's live set does not fit, and the planner paid a
        # STORE + LOAD round trip to DDR for a group-internal buffer.
        # Section 7 calls that level 2, the thing the chosen `S` is
        # supposed to make unnecessary -- and unlike a DDR *placement* it
        # is invisible in `ddr_placements`, so it has to be asked for
        # separately or a "fallback-free" case quietly writes DDR.
        return (
            False,
            f"{planned.traffic.spill_count} buffer(s) had to be spilled mid-strip",
            descriptors,
        )
    return True, "", descriptors


# ---------------------------------------------------------------------------
# The rule (design section 4.2).
# ---------------------------------------------------------------------------


def candidate_strip_counts(height: int) -> list[int]:
    """Every `S` in `1..height` that produces a *distinct* strip height
    `R`, smallest `S` first.

    `S` and `R` say the same thing, and searching `S` directly evaluates
    the same candidate many times over (at `H = 640`, `S = 22..26` all
    mean `R = 26..30`... and `S = 321..640` all mean `R = 1` or `2`).
    Keeping the smallest `S` per `R` is also exactly the rule's tie-break
    ("ties -> fewer strips") applied one level earlier."""
    by_rows: dict[int, int] = {}
    for strips in range(1, height + 1):
        rows = -(-height // strips)
        by_rows.setdefault(rows, strips)
    return sorted(by_rows.values())


@dataclass
class GroupChoice:
    """The rule's verdict for one group."""

    group: int
    height: int
    #: The candidate that will actually be built -- never `None`. When
    #: no candidate satisfied section 4.2 this is the section 7 level 3
    #: *fallback* choice (see `fell_back`), which still has a real cost
    #: and must still be counted: a group that fell back is the most
    #: expensive one in the network, not a free one.
    cost: GroupCost | None
    #: True when no `(S, residency)` satisfied the rule and this is the
    #: layer-wise fallback rather than a fused choice.
    fell_back: bool = False
    #: Every candidate considered, for a test or a report to explain the
    #: choice rather than assert it blindly.
    considered: list[GroupCost] = field(default_factory=list)

    @property
    def strips(self) -> int | None:
        return None if self.cost is None else self.cost.strips


def choose_group(
    model: Model,
    groups: list[Group],
    group: Group,
    *,
    planner: Planner,
    recompute_cap: float = DEFAULT_RECOMPUTE_CAP,
    allow_resident_weights: bool = True,
    planewise_add: bool = True,
) -> GroupChoice:
    """Section 4.2, for one group."""
    residencies = (True, False) if allow_resident_weights else (False,)
    considered: list[GroupCost] = []
    best: GroupCost | None = None
    for strips in candidate_strip_counts(group.height):
        for resident in residencies:
            cost = evaluate(
                model,
                groups,
                group,
                strips,
                weights_resident=resident,
                planner=planner,
                planewise_add=planewise_add,
            )
            considered.append(cost)
            if not cost.fits or cost.recompute > recompute_cap:
                continue
            if best is None or (cost.ddr_total, cost.strips) < (best.ddr_total, best.strips):
                best = cost
    if best is not None:
        return GroupChoice(
            group=group.index, height=group.height, cost=best, considered=considered
        )

    # Section 7 level 3. Nothing satisfied both the fit test and the
    # recompute cap, so the group is dissolved: it still runs strip by
    # strip -- the ifmap is still LOADed once instead of being
    # re-streamed by the output-channel loop, which is the 202 -> 48 MB
    # floor the design guarantees -- but at whatever height does fit,
    # cap or no cap. Cheapest feasible candidate wins, ties to fewer
    # strips, exactly as above.
    feasible = [c for c in considered if c.fits]
    fallback = (
        min(feasible, key=lambda c: (c.ddr_total, c.strips)) if feasible else None
    )
    return GroupChoice(
        group=group.index,
        height=group.height,
        cost=fallback,
        fell_back=True,
        considered=considered,
    )


@dataclass
class Selection:
    """The rule's verdict for a whole model, and the `strips` argument
    `tiler.tile` should be given."""

    choices: list[GroupChoice]
    #: group index -> chosen `S`, ready to hand to `tiler.tile`.
    strips: dict[int, int] = field(default_factory=dict)
    #: The grouping these choices are for -- `form_groups`', unless a
    #: group had to be dissolved (section 7 level 3). Pass it to
    #: `tiler.tile` alongside `strips`, or the indices will not line up.
    groups: list = field(default_factory=list)
    #: Groups no `(S, residency)` satisfied, which are dissolved into
    #: single-op groups (section 7 level 3).
    fallbacks: list[int] = field(default_factory=list)
    #: Group indices whose weights the rule chose to keep resident in the
    #: scratchpad -- hand it to `tiler.tile(..., resident_groups=...)`.
    resident_groups: set[int] = field(default_factory=set)

    @property
    def ddr_read(self) -> int:
        return sum(c.cost.ddr_read for c in self.choices if c.cost)

    @property
    def ddr_write(self) -> int:
        return sum(c.cost.ddr_write for c in self.choices if c.cost)


def select(
    model: Model,
    *,
    planner: Planner | None = None,
    tensor_mem_bytes: int | None = None,
    bank_bytes: int | None = None,
    recompute_cap: float = DEFAULT_RECOMPUTE_CAP,
    allow_resident_weights: bool = True,
    planewise_add: bool = True,
    ddr_scale: int = 1,
    strict_fallback: bool = False,
) -> Selection:
    """Choose a strip height for every fusion group of `model`.

    Pass either a configured `planner` or the scratchpad geometry to
    build one from; the planner is used only to *probe* candidate
    strips, never to plan the final program (the caller does that, on
    `tiler.tile(model, selection.strips, groups=selection.groups)`).

    `strict_fallback` is the design's section 7 level 3 as literally
    written: a group that satisfies no `(S, residency)` is *always*
    dissolved into single-op groups. Measured on YOLOv8n that is worse
    than the thing it replaces -- dissolving L8+SPPF, L12 and L15 costs
    5.5 MB more per frame at 256 KiB than simply running them at the
    tallest strip that fits and accepting the over-cap recompute, because
    every intermediate of a dissolved group becomes a DDR round trip
    while the recompute it avoids is compute, not traffic. So the default
    takes whichever of the two the rule's own objective prefers: dissolve
    when dissolving really is cheaper in DDR bytes, otherwise keep the
    group fused at the tallest feasible strip and report it as a
    fallback. Either way the fit requirement is never relaxed -- only the
    recompute cap is, and only when that saves bandwidth."""
    if planner is None:
        planner = (
            Planner(
                tensor_mem_bytes=tensor_mem_bytes,
                bank_bytes=bank_bytes,
                ddr_map=DdrMap(scale=ddr_scale),
            )
            if tensor_mem_bytes is not None
            else Planner(ddr_map=DdrMap(scale=ddr_scale))
        )

    groups = form_groups(model)
    selection = Selection(choices=[])
    index = 0
    while index < len(groups):
        group = groups[index]
        # A fresh planner per probe: `Planner` holds a `DdrMap` whose
        # bump allocators are stateful, and thousands of throwaway probes
        # would otherwise walk it off the end of its regions.
        probe_planner = Planner(
            tensor_mem_bytes=planner.tensor_mem_bytes,
            bank_bytes=planner.bank_bytes,
            ddr_map=DdrMap(scale=planner.ddr_map.scale),
        )
        choice = choose_group(
            model,
            groups,
            group,
            planner=probe_planner,
            recompute_cap=recompute_cap,
            allow_resident_weights=allow_resident_weights,
            planewise_add=planewise_add,
        )
        if choice.fell_back and len(group.ops) > 1:
            trial_groups = dissolve(groups, index)
            trial = [
                choose_group(
                    model,
                    trial_groups,
                    trial_groups[i],
                    planner=Planner(
                        tensor_mem_bytes=planner.tensor_mem_bytes,
                        bank_bytes=planner.bank_bytes,
                        ddr_map=DdrMap(scale=planner.ddr_map.scale),
                    ),
                    recompute_cap=recompute_cap,
                    allow_resident_weights=allow_resident_weights,
                    planewise_add=planewise_add,
                )
                for i in range(index, index + len(group.ops))
            ]
            dissolved_cost = (
                sum(c.cost.ddr_total for c in trial)
                if all(c.cost is not None for c in trial)
                else None
            )
            # `choice.cost is None` means the group does not fit at ANY
            # strip height as one group, so dissolving is not a
            # cost comparison but the only lowering left. (Before the
            # probe learned to plan the *widest* strip rather than the
            # first, no group ever reached this state, and asking the
            # comparison anyway raised `AttributeError` instead of
            # falling back.)
            take_it = dissolved_cost is not None and (
                strict_fallback
                or choice.cost is None
                or dissolved_cost < choice.cost.ddr_total
            )
            if take_it:
                groups = trial_groups
                for sub_choice in trial:
                    selection.choices.append(sub_choice)
                    if sub_choice.fell_back:
                        selection.fallbacks.append(sub_choice.group)
                    selection.strips[sub_choice.group] = sub_choice.cost.strips
                    if sub_choice.cost.weights_resident:
                        selection.resident_groups.add(sub_choice.group)
                index += len(group.ops)
                continue
        selection.choices.append(choice)
        if choice.fell_back:
            selection.fallbacks.append(group.index)
        if choice.cost is None:
            raise TilingError(
                f"tiler: group {group.index} does not fit at ANY strip height "
                f"(1..{group.height}). Last reason: "
                f"{choice.considered[-1].reason if choice.considered else 'unknown'}. "
                "Even one output row needs more scratchpad than there is; the group must "
                "be split, or its buffers left in DDR by the planner's own fallback."
            )
        selection.strips[group.index] = choice.cost.strips
        if choice.cost.weights_resident:
            selection.resident_groups.add(group.index)
        index += 1
    selection.groups = groups
    return selection


__all__ = [
    "DEFAULT_RECOMPUTE_CAP",
    "GroupCost",
    "GroupChoice",
    "Selection",
    "candidate_strip_counts",
    "choose_group",
    "evaluate",
    "group_macs",
    "group_weight_bytes",
    "select",
]
