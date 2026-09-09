"""Spatial tiling with cross-layer fusion: a `Model -> Model` pass.

This is the whole of the tiling logic. It takes an ordinary graph, cuts
it into **fusion groups**, and rewrites each group into `S` independent
**strip sub-programs** built entirely out of the operations that already
exist -- `conv2d`, `pool_*`, `add`, `upsample2x`, `split`, `concat`, plus
`RowCopyOp` for moving row windows. The result is a plain (if larger)
`Model`; `planner.py` plans it, `reference.py` executes it and
`program.py` lowers it with no knowledge that tiling happened, and the
ISA does not change at all.

Why strips have to be whole tensors
-----------------------------------

Two hardware facts shape everything here, and neither is negotiable.

**F2 -- a plane stride is derived from the descriptor's own height.**
`cmd_proc.vhd` computes `in_plane_bytes = in_width * in_height * 8` from
the descriptor, and the ifmap feeder fetches row `r` of channel tile
`t` at `in_addr + t*in_plane_bytes + r*in_row_bytes`. There is no field
that says "this buffer is really taller than the part I am reading". So
a strip **cannot** be "the same tensor at an offset address" whenever it
has more than one channel plane: it must be a *compact* tensor of its
own height. That is why this pass rewrites the graph instead of
annotating it, why `RowCopyOp` exists at all, and why a group boundary
has to be assembled plane-row by plane-row.

**F4 -- the output-channel loop re-streams the whole ifmap once per
tile.** Reading a conv's input from DDR therefore costs `ceil(Cout/8)`
times the ifmap, up to 32x. Loading a strip into the scratchpad *once*
and letting the OT loop re-read it locally is where most of the traffic
win comes from -- more than fusion itself.

How a strip is built (design section 2.5)
-----------------------------------------

A strip is anchored on the group's **output** rows `[o0, o1)`. Working
backwards through the group's ops, each tensor gets the row range its
consumers need:

===========================  ==========================================
op producing rows `[a, b)`   rows of its input it reads
===========================  ==========================================
conv/pool `k x k` stride `s`
pad `p`                      `[a*s - p, (b-1)*s + k - p)`
1x1 stride-1, ADD, ACT,
COPY, split, concat          `[a, b)`
UPSAMPLE 2x                  `[a//2, ceil(b/2))`
===========================  ==========================================

clipped to the tensor's real rows -- and whatever the clip removed is
exactly what the strip's `pad_top`/`pad_bottom` must supply. Halo rows
are **recomputed** by each neighbouring strip rather than carried
between them, which is what keeps a strip an independent, stateless
sub-program (design D2, and section 2.7's costing of the alternatives).

Then, forwards, each op is cloned onto the strip tensors: same weights
object (never `WEIGHT_REUSE` -- see R8 in `program.py`), padding recut
per strip, and a `RowCopyOp` window inserted wherever the buffer a
consumer has is taller than the rows that consumer actually wants (a
join whose operands come from different chain depths, or an UPSAMPLE
whose output had to be rounded out to an even row boundary).

What this module deliberately does not do
-----------------------------------------

Choose `S`. That is `tiling_select.py`'s job, and separating them is
what makes the correctness of the rewrite testable on its own: the
oracle in `tests/test_tiler.py` runs *every* strip height from 1 to
`H_out` on every block builder in the catalogue and requires the tiled
program's `reference.py` result to equal the untiled program's, bit for
bit. No simulator is involved, and that is where the real risk lives --
the DUT and `reference.py` both execute the planner's addresses, so a
halo off by one row is self-consistently wrong on both sides and only a
comparison against the *untiled* result can see it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from accel_v2.model import (
    ActOp,
    AddOp,
    Conv2dOp,
    CopyOp,
    Model,
    Op,
    PoolOp,
    RowRange,
    StripOrigin,
    Tensor,
    UpsampleOp,
    alias_plane_offset_total,
    alias_root,
)


class TilingError(ValueError):
    """A graph this pass cannot tile. Always actionable: the message says
    which construct and why, so the caller can either restructure the
    graph or leave the group untiled."""


# ---------------------------------------------------------------------------
# Group formation (design section 3.1).
# ---------------------------------------------------------------------------


@dataclass
class Group:
    """A maximal run of ops that one strip sub-program computes together.

    `height` is the row count of the group's *outputs*, the coordinate
    system a strip is anchored in. Every op in the group produces a
    tensor of that height -- an op that does not (a stride-2 conv, an
    UPSAMPLE) starts a new group, which is what makes the anchor
    well defined without any per-group configuration."""

    index: int
    ops: list[Op]
    height: int

    @property
    def name(self) -> str:
        return f"g{self.index}"


def _has_stride(op: Op) -> bool:
    return isinstance(op, (Conv2dOp, PoolOp)) and any(s > 1 for s in op.stride)


def form_groups(model: Model) -> list[Group]:
    """Cut `model.ops` into fusion groups, design section 3.1.

    The rule, in order of application:

    * an op with any stride > 1 is a group **of its own** -- fusing
      across a downsample means carrying two row-coordinate systems
      through one strip, and section 3.4 costs that out at ~0.6 % of the
      traffic for double the geometry;
    * an op whose output height differs from the run's starts a new
      group. This is what puts an `UPSAMPLE` at the *head* of the group
      that follows it (its output is already at the group's resolution),
      which is precisely what section 3.3 asks for, without a special
      case for it;
    * otherwise the op joins the current run.

    Deterministic, needs no hints, and yields exactly the 19 groups of
    section 5 on YOLOv8n."""
    groups: list[Group] = []
    current: list[Op] = []
    height = None

    def flush() -> None:
        nonlocal current, height
        if current:
            groups.append(Group(index=len(groups), ops=current, height=height))
        current, height = [], None

    for op in model.ops:
        out_h = op.output.height
        if _has_stride(op):
            flush()
            groups.append(Group(index=len(groups), ops=[op], height=out_h))
            continue
        if current and out_h != height:
            flush()
        if not current:
            height = out_h
        current.append(op)
    flush()
    return groups


def _plane_span(tensor: Tensor) -> tuple[int, int]:
    """`tensor`'s planes as a half-open range inside its alias root's
    buffer."""
    start = alias_plane_offset_total(tensor)
    return start, start + tensor.plane_count


def _family(model: Model, root: Tensor) -> list[Tensor]:
    return [t for t in model.tensors if alias_root(t) is root]


def is_locally_built(group: Group, tensor: Tensor) -> bool:
    """Can this group produce `tensor`'s bytes itself, or must it load
    them?

    A tensor is locally built when one of its own ops writes it, when it
    is a `concat` **any** of whose parts is locally built (the group
    assembles the buffer and loads whatever slots it does not fill), or
    when it is a `split` **slice** of a locally built parent. A *part*
    never consults its parent: a part's bytes come from its producer, not
    from the buffer it happens to sit in -- which is also what stops this
    walking in circles between a concat and its parts."""
    produced = {op.output.name for op in group.ops}

    def walk(t: Tensor) -> bool:
        if t.name in produced:
            return True
        if t.alias_parts:
            return any(walk(part) for part in t.alias_parts)
        if t.alias_role == "slice" and t.alias_parent is not None:
            return walk(t.alias_parent)
        return False

    return walk(tensor)


def dissolve(groups: list[Group], index: int) -> list[Group]:
    """Replace `groups[index]` with one group per op -- design section 7
    level 3, the group-level fallback.

    Taken when no `(S, residency)` at all satisfies the selection rule:
    the fusion is given up, every intermediate of the group becomes a
    pinned DDR boundary, and each op is tiled on its own. That is
    strictly worse than fusing and strictly better than not tiling -- the
    ifmap is still strip-LOADed into the scratchpad once instead of being
    re-streamed by the output-channel loop, which is the 202 -> 48 MB
    floor the whole design rests on.

    A single-op group cannot fail in turn: one input row plus one output
    plane always fit, and if they somehow did not, `planner.py`'s own
    per-buffer fallback is still underneath."""
    replacement = [Group(index=0, ops=[op], height=op.output.height) for op in groups[index].ops]
    merged = groups[:index] + replacement + groups[index + 1 :]
    return [Group(index=i, ops=g.ops, height=g.height) for i, g in enumerate(merged)]


def group_inputs(model: Model, groups: list[Group], group: Group) -> list[Tensor]:
    """The tensors this group must LOAD from DDR, at **member**
    granularity, in first-use order.

    Not simply "the operands produced elsewhere". Two cases make it more
    than that, and both are the neck:

    * a stride-2 conv reads `L4_cv2`'s output, which is a *part* of the
      neck's 128-channel concat family. The strip wants that part's 64
      channels, not the family;
    * the neck's own C2f reads the assembled `[up | skip]` concat, of
      which its group builds the `up` half itself -- so the `skip` half
      is loaded straight into its slot in the strip's concat buffer
      (design section 3.3), and the `up` half is not loaded at all.
    """
    produced = {op.output.name for op in group.ops}
    loads: list[Tensor] = []
    seen: set[str] = set()

    def sources(t: Tensor) -> None:
        if t.name in produced:
            return
        if is_locally_built(group, t):
            if t.alias_parts:
                for part in t.alias_parts:
                    sources(part)
                return
            if t.alias_role == "slice" and t.alias_parent is not None:
                sources(t.alias_parent)
                return
        if t.name in seen:
            return
        seen.add(t.name)
        loads.append(t)

    for op in group.ops:
        for operand in op.inputs:
            sources(operand)
    return loads


def group_outputs(model: Model, groups: list[Group], group: Group) -> list[Tensor]:
    """The tensors this group produces that anything outside it reads,
    at **member** granularity: the exact planes a strip must store back.

    Root granularity is not good enough. `L4_cv2`'s output is a part of
    the neck's concat family, and its group produces only that part;
    storing the whole family would write over a slot its neighbour owns.
    So the question asked of each produced tensor `M` is: *do the planes
    M occupies get read outside this group?* -- which is true when M
    itself has an outside consumer or is a graph output, and also when
    some larger view of the same buffer (the assembled concat) is read
    outside and overlaps M's planes."""
    group_of_op = {id(op): g.index for g in groups for op in g.ops}

    def read_outside(t: Tensor) -> bool:
        return t.is_output or any(
            group_of_op[id(model.ops[c])] != group.index for c in t.consumers
        )

    outs: list[Tensor] = []
    seen: set[str] = set()
    for op in group.ops:
        member = op.output
        if member.name in seen:
            continue
        lo, hi = _plane_span(member)
        crosses = read_outside(member) or any(
            read_outside(other)
            and max(lo, _plane_span(other)[0]) < min(hi, _plane_span(other)[1])
            for other in _family(model, alias_root(member))
            if other is not member
        )
        if crosses:
            seen.add(member.name)
            outs.append(member)
    return outs


# ---------------------------------------------------------------------------
# The row-range recurrence (design section 2.5 step 1).
# ---------------------------------------------------------------------------


def input_rows(op: Op, out_rows: RowRange) -> list[RowRange]:
    """The rows of each of `op`'s inputs needed to produce `out_rows` of
    its output, **unclipped** -- a range running below 0 or past the
    tensor's last row is not an error, it is the padding the strip's
    `pad_top`/`pad_bottom` will supply, and losing that information is
    exactly how a strip ends up padding an interior boundary.

    The one place the recurrence table of section 2.5 is implemented."""
    a, b = out_rows.r0, out_rows.r1
    if isinstance(op, (Conv2dOp, PoolOp)):
        kh = op.kernel[0]
        sh = op.stride[0]
        pad_top = op.padding[0]
        return [RowRange(a * sh - pad_top, (b - 1) * sh + kh - pad_top)]
    if isinstance(op, UpsampleOp):
        f = op.factor
        return [RowRange(a // f, -(-b // f))]
    if isinstance(op, (AddOp, CopyOp, ActOp)):
        return [out_rows for _ in op.inputs]
    raise TilingError(
        f"tiler: no row-range rule for op '{op.name}' of type {type(op).__name__}. Every "
        "op in a fused group must say which input rows one output row depends on; add it "
        "to `input_rows` (and to the table in this module's docstring) before tiling a "
        "graph that uses it."
    )


def output_rows(op: Op, in_rows: RowRange, *, pad_top: int, pad_bottom: int) -> RowRange:
    """The rows this op's clone will actually produce, given a strip
    input holding `in_rows` (already clipped) and the recut padding.

    Computed from the *descriptor fields* -- `in_height`, `pad_top`,
    `pad_bottom`, kernel, stride -- exactly as `cmd_proc` will, rather
    than from what the recurrence hoped for. Comparing the two is the
    geometry assertion that catches an off-by-one in `input_rows` at
    build time, before any address exists."""
    if isinstance(op, (Conv2dOp, PoolOp)):
        kh, sh = op.kernel[0], op.stride[0]
        rows = (in_rows.rows + pad_top + pad_bottom - kh) // sh + 1
        # `a*sh - pad_top == in_rows.r0 - pad_top` gives the first output
        # row this strip computes, in the FULL tensor's coordinates.
        first = (in_rows.r0 - pad_top + op.padding[0]) // sh
        return RowRange(first, first + rows)
    if isinstance(op, UpsampleOp):
        f = op.factor
        return RowRange(in_rows.r0 * f, in_rows.r1 * f)
    return in_rows


def _recut_padding(op: Op, unclipped: RowRange, in_height: int) -> tuple[int, int]:
    """`(pad_top, pad_bottom)` for a strip clone of `op` whose input
    strip is `unclipped` clipped to `[0, in_height)`.

    The padded taps are precisely the rows the recurrence asked for that
    do not exist, so the pad counts are the sizes of the two clipped-off
    ends. That is strictly more accurate than "the original padding on
    the first/last strip, zero in between": for a 5x5/pad-2 pool the
    strip starting at output row 1 needs *one* pad row, not two and not
    none, and SPPF at `S >= 2` hits that on the very first interior
    boundary."""
    return max(0, -unclipped.r0), max(0, unclipped.r1 - in_height)


def propagate_rows(
    group: Group, out_rows: RowRange, outs: list[Tensor]
) -> dict[str, RowRange]:
    """`tensor name -> rows of it this strip must materialize`.

    Backwards over the group's ops, unioning what every consumer
    asks for. Keyed per **tensor**, not per alias family, because the
    two alias roles pull in opposite directions:

    * a `split` **slice** genuinely *is* its parent's rows, so a
      requirement on the slice is a requirement on the parent
      (`require` walks the edge), and the slice ends up covering
      whatever the parent does;
    * a concat **part** is not. Every part of a concat buffer must be
      the same height, so the concat's rows are pushed *down* onto
      each part as a lower bound -- but a part whose own consumers
      need a taller halo (the C2f bottleneck chain reads `m0`'s
      output two 3x3 convolutions deeper than the concat does) keeps
      its own taller buffer, and a window of it becomes the part.
      Forcing the concat to the hull instead would make every branch
      of the block as tall as the deepest one, which for a C2f at
      `n = 2` is four extra rows on eight buffers.

    Both directions are handled inside `require`, so the single
    reverse sweep is enough: a concat's consumers, a slice's
    consumers and a part's own consumers are all ops that come
    *after* the producer whose requirement they set.
    """
    need: dict[str, RowRange] = {}

    def require(tensor: Tensor, rows: RowRange) -> None:
        clipped = rows.clip(tensor.height)
        previous = need.get(tensor.name)
        merged = clipped if previous is None else previous.hull(clipped)
        if merged == previous:
            return
        need[tensor.name] = merged
        if tensor.alias_role == "slice" and tensor.alias_parent is not None:
            # The slice is the parent's own rows; the parent must
            # hold them.
            require(tensor.alias_parent, merged)
        for part in tensor.alias_parts:
            # Every part of this buffer must be exactly this tall.
            require(part, merged)

    for root in outs:
        require(root, out_rows)

    for op in reversed(group.ops):
        want = need.get(op.output.name)
        if want is None:
            # Nothing downstream (inside or outside the group) reads
            # this op's result for this strip. Still cloned, on the
            # strip's own output rows, so the program stays a
            # faithful rewrite of a graph that computes dead values.
            want = out_rows.clip(op.output.height)
            need[op.output.name] = want
        for original, rows in zip(op.inputs, input_rows(op, want)):
            require(original, rows)
    return need


# ---------------------------------------------------------------------------
# The pass itself.
# ---------------------------------------------------------------------------


@dataclass
class StripRecord:
    """What one strip of one group turned out to be -- the tiler's own
    account of its geometry, for tests and dumps to check against an
    independently derived one."""

    group: int
    strip: int
    out_rows: RowRange
    #: original tensor name -> the rows of it this strip materialized.
    rows: dict[str, RowRange] = field(default_factory=dict)
    #: original op name -> `(pad_top, pad_bottom)` this strip's clone got.
    padding: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class TiledModel:
    """Result of `tile`. `model` is the rewritten graph; everything else
    is provenance, never consulted by the planner."""

    model: Model
    source: Model
    groups: list[Group]
    strips: list[StripRecord]
    #: original alias-root name -> the pinned full-resolution tensor in
    #: `model` that carries it across group boundaries.
    pinned: dict[str, Tensor] = field(default_factory=dict)
    #: group index -> number of strips it was cut into.
    strip_counts: dict[int, int] = field(default_factory=dict)
    #: Every pinned buffer (or plane slice of one) that strip stores
    #: write. What R11's partition check should be asked about.
    stored: list[Tensor] = field(default_factory=list)

    def rows_of(self, group: int, strip: int, tensor_name: str) -> RowRange:
        record = next(r for r in self.strips if r.group == group and r.strip == strip)
        return record.rows[tensor_name]


def strip_bounds(height: int, strips: int) -> list[RowRange]:
    """Cut `[0, height)` into row strips of `R = ceil(height/strips)`.

    Deliberately not a balanced cut: `R` is the number the cost model and
    the working-set table of section 3.2 are written in terms of, so the
    strip a given `S` produces must be the one those numbers describe.
    The last strip is short, and a `strips` that would leave an empty one
    simply yields fewer (which is also the selection rule's "ties ->
    fewer strips")."""
    if not 1 <= strips <= height:
        raise TilingError(
            f"tiler: {strips} strips is not a legal cut of {height} rows -- S must be "
            "between 1 and the group's output height"
        )
    rows_per_strip = -(-height // strips)
    return [
        RowRange(start, min(start + rows_per_strip, height))
        for start in range(0, height, rows_per_strip)
    ]


def tile(
    model: Model,
    strips: int | dict[int, int] = 1,
    *,
    planewise_add: bool = True,
    name: str | None = None,
    groups: list[Group] | None = None,
    resident_groups: set[int] | None = None,
) -> TiledModel:
    """Rewrite `model` into an equivalent strip-tiled model.

    `strips` is `S`: one number for every group, or a `{group index: S}`
    mapping (missing groups default to 1). Choosing it is not this
    function's job -- see `tiling_select.py`.

    `planewise_add` emits a multi-plane `ADD` as one `ADD` per plane
    (`Model.add_planewise`). On by default because
    `cnn_accel_elementwise` issues an `ADD` as a *single* request per
    operand, while a multi-plane strip buffer is allowed to span banks at
    plane granularity; a whole-tensor request over one would be silently
    clamped. Turning it off is only useful for isolating that from the
    rest of the rewrite in a test.

    `groups` overrides the grouping `form_groups` would derive -- pass
    the list `tiling_select.select` settled on, which may have dissolved
    a group that did not fit (design section 7 level 3).

    `resident_groups` is the set of group indices whose convolutions
    should serve their packed weight/bias/scale images out of the
    scratchpad (`Conv2dOp.weights_resident`). Every strip's clone of one
    conv shares that conv's `weight` list, which is what lets
    `planner.py` give the `S` clones ONE image and load it once instead
    of once per strip -- the whole of the saving. Pass
    `tiling_select.Selection.resident_groups`; an unplaceable image
    silently falls back to DDR weights in the planner, so this is a
    request rather than a promise."""
    return _Tiler(
        model,
        strips,
        planewise_add=planewise_add,
        name=name,
        groups=groups,
        resident_groups=resident_groups,
    ).run()


def probe_strip(
    model: Model,
    group: Group,
    out_rows: RowRange,
    *,
    planewise_add: bool = True,
    groups: list[Group] | None = None,
) -> Model:
    """Build a `Model` containing **one** strip of one group -- the
    sub-program the selection rule plans to find out whether a strip
    height actually fits.

    Section 4.2 makes the fit decision with the real allocator rather
    than an estimate, which removes the estimate-versus-reality gap
    entirely -- but only one strip need be built to answer it: every
    strip of a group is the same sub-program on the same buffer sizes
    (the last is shorter, i.e. strictly easier). Building all `S` of them
    for each of `H` candidate heights would be quadratic in the group's
    height for no extra information."""
    tiler = _Tiler(
        model, 1, planewise_add=planewise_add, name=f"{model.name}_probe", groups=groups
    )
    outs = group_outputs(model, tiler.groups, group)
    produced = {alias_root(op.output).name for op in group.ops}
    tiler._tile_strip(group, 0, out_rows, outs, produced)
    return tiler.out


class _Tiler:
    """The rewrite, one group at a time. Split out of `tile` only so the
    per-strip bookkeeping can live in attributes instead of in a closure
    stack five deep."""

    def __init__(
        self,
        source: Model,
        strips,
        *,
        planewise_add: bool,
        name: str | None,
        groups: list[Group] | None = None,
        resident_groups: set[int] | None = None,
    ) -> None:
        self.source = source
        self.resident_groups = set(resident_groups or ())
        #: Whether the group currently being tiled wants resident
        #: weights, read by `_clone_shaped` (which is four calls deep and
        #: has no other way to know which group it is in).
        self._resident = False
        self.planewise_add = planewise_add
        self.out = Model(seed=source.seed, name=name or f"{source.name}_tiled")
        self.groups = form_groups(source) if groups is None else groups
        self.strips_of = (
            {g.index: strips for g in self.groups} if isinstance(strips, int) else dict(strips)
        )
        self.pinned: dict[str, Tensor] = {}
        self.records: list[StripRecord] = []
        self.strip_counts: dict[int, int] = {}
        #: The pinned tensors (or plane slices of them) strip stores
        #: actually write, for the R11 partition check to be asked about
        #: the right thing -- a pinned concat family may have a slot no
        #: group ever writes to DDR, because the group that reads it also
        #: produces it.
        self.stored: list[Tensor] = []

        #: original op -> index of the group it belongs to.
        self.group_of_op: dict[int, int] = {
            id(op): g.index for g in self.groups for op in g.ops
        }
        #: alias-root name -> index of the group whose ops write it, or
        #: `None` for a graph input (written by nobody).
        self.written_in: dict[str, int | None] = {}
        for group in self.groups:
            for op in group.ops:
                self.written_in.setdefault(alias_root(op.output).name, group.index)

    # -- boundaries ------------------------------------------------------

    def _pin(self, root: Tensor) -> Tensor:
        """The full-resolution DDR tensor that carries `root`'s buffer
        between groups, created on first use.

        Flat, never aliased: an alias family is a *scratchpad* layout
        decision, and re-creating it at a group boundary would tie two
        groups' buffer shapes together for no benefit. Each side of the
        boundary reconstructs whatever views it needs inside its own
        strips."""
        existing = self.pinned.get(root.name)
        if existing is not None:
            return existing

        if root.is_input:
            pinned = self.out.input(
                root.height, root.width, root.channels, root.quant.scale, name=root.name
            )
            # Byte-identical input data, not a fresh draw: the oracle
            # compares the tiled program's output against the untiled
            # one's, which is only meaningful if both ran on the same
            # bytes. (`Model.input` consumed an RNG draw to make them;
            # that draw is simply discarded.)
            pinned.data = list(root.data)
        else:
            pinned = Tensor(
                name=root.name,
                height=root.height,
                width=root.width,
                channels=root.channels,
                quant=root.quant,
                pin_ddr=True,
            )
            self.out.tensors.append(pinned)
            if root.is_output:
                self.out.output(pinned)

        self.pinned[root.name] = pinned
        return pinned

    # -- the run ---------------------------------------------------------

    def _pinned_view(self, member: Tensor, pinned_root: Tensor) -> Tensor:
        """A read-only view of `pinned_root` covering `member`'s planes.

        The pinned buffer is flat, so an alias member of it is a plane
        range at the member's own total plane offset -- the same
        arithmetic `planner.py` already does for any `split` view, and
        the reason a group boundary does not have to reproduce the alias
        structure that produced it."""
        key = f"{pinned_root.name}::{member.name}"
        existing = next((t for t in self.out.tensors if t.name == key), None)
        if existing is not None:
            return existing
        view = Tensor(
            name=key,
            height=pinned_root.height,
            width=pinned_root.width,
            channels=member.channels,
            quant=member.quant,
            alias_parent=pinned_root,
            alias_plane_offset=alias_plane_offset_total(member),
            alias_role="slice",
        )
        self.out.tensors.append(view)
        return view

    #: Ops whose hardware requests are NOT bounded by one activation
    #: plane, and whose operands therefore cannot be confined at plane
    #: granularity (design section 2.4's own proviso, "as long as the
    #: planner never emits a `T = 1`-style whole-tensor request or an
    #: `ADD`/`COPY` bigger than a plane").
    #:
    #: * `COPY` and `ACT` issue **one** request of `xfer_bytes` -- the
    #:   whole tensor (`cnn_accel_elementwise.vhd`, `cur_len_q <=
    #:   xfer_len_q`).
    #: * `UPSAMPLE` issues 8-byte reads and 16-byte writes, walking the
    #:   buffer sequentially, so a request straddles a plane boundary
    #:   whenever the plane size is not a multiple of the beat pair.
    #:
    #: Found by the DUT, not by inspection: `cnn_accel_tensor_mem`
    #: asserts on a straddling request, and the first tiled case with an
    #: `UPSAMPLE` in it stopped the simulation on that assertion. A
    #: buffer these ops touch is therefore left with the strict rule
    #: (whole buffer in one bank), which is what every untiled tensor
    #: uses; if it then does not fit, the planner's ordinary DDR fallback
    #: takes it.
    _UNPLANNED_REQUEST_OPS = (CopyOp, ActOp, UpsampleOp)

    def _relax_unconfinable(self) -> None:
        """Undo `tag`'s plane confinement wherever an op would issue a
        request bigger than one plane against the buffer."""
        family: dict[int, list[Tensor]] = {}
        for tensor in self.out.tensors:
            family.setdefault(id(alias_root(tensor)), []).append(tensor)

        def unconfine(tensor: Tensor) -> None:
            root = alias_root(tensor)
            for member in family.get(id(root), [root]):
                member.confine_unit_bytes = None
            root.confine_unit_bytes = None

        for op in self.out.ops:
            unbounded = isinstance(op, self._UNPLANNED_REQUEST_OPS) or (
                # A multi-plane ADD is one request per operand over the
                # whole tensor; `Model.add_planewise` is what normally
                # keeps it to a plane, and this catches the case where it
                # was turned off.
                isinstance(op, AddOp) and op.inputs[0].plane_count > 1
            )
            if not unbounded:
                continue
            for tensor in list(op.inputs) + [op.output]:
                unconfine(tensor)

    def run(self) -> TiledModel:
        for group in self.groups:
            self._tile_group(group)
        self._relax_unconfinable()
        return TiledModel(
            model=self.out,
            source=self.source,
            groups=self.groups,
            strips=self.records,
            pinned=self.pinned,
            strip_counts=self.strip_counts,
            stored=list({id(t): t for t in self.stored}.values()),
        )

    def _tile_group(self, group: Group) -> None:
        produced = {alias_root(op.output).name for op in group.ops}
        outs = group_outputs(self.source, self.groups, group)
        if not outs:
            raise TilingError(
                f"tiler: group {group.index} ({[op.name for op in group.ops]}) produces "
                "nothing anything else reads. A group with no output has no row anchor to "
                "tile against; mark its result a graph output or drop the dead ops."
            )
        for member in outs:
            if member.height != group.height:
                raise TilingError(
                    f"tiler: group {group.index} outputs '{member.name}' of height "
                    f"{member.height} but the group's rows are {group.height} -- group "
                    "formation should have split this; that is a bug in `form_groups`."
                )

        # A group shorter than the requested `S` is simply cut into as
        # many strips as it has rows. That keeps a single uniform `S`
        # meaningful across a graph whose groups sit at different
        # resolutions (a stride-2 conv's group is half as tall as its
        # neighbours), which is what the oracle sweeps and what a
        # per-group selection would arrive at anyway.
        bounds = strip_bounds(
            group.height, min(self.strips_of.get(group.index, 1), group.height)
        )
        self.strip_counts[group.index] = len(bounds)
        self._resident = group.index in self.resident_groups
        for strip_index, out_rows in enumerate(bounds):
            self._tile_strip(group, strip_index, out_rows, outs, produced)

    # -- one strip -------------------------------------------------------

    def _tile_strip(
        self,
        group: Group,
        strip_index: int,
        out_rows: RowRange,
        group_outputs: list[Tensor],
        produced: set[str],
    ) -> None:
        record = StripRecord(group=group.index, strip=strip_index, out_rows=out_rows)

        need = propagate_rows(group, out_rows, group_outputs)
        record.rows.update(need)

        #: original tensor name -> its clone in this strip, and the rows
        #: (in ORIGINAL coordinates) that clone holds.
        made: dict[str, Tensor] = {}
        covers: dict[str, RowRange] = {}

        def tag(clone: Tensor, original: Tensor, rows: RowRange) -> Tensor:
            clone.origin = StripOrigin(full=original, rows=rows, strip=strip_index)
            clone.confine_unit_bytes = clone.plane_bytes
            return clone

        def suffix(t: Tensor) -> str:
            return f"{t.name}_{group.name}s{strip_index}"

        def load_group_input(t: Tensor) -> Tensor:
            """Bring the rows of `t` this strip needs in from DDR.

            `t` is whatever the consuming op names, which need not be a
            whole pinned buffer: a stride-2 conv reads `L4_cv2`'s output,
            and that output is a *part* of the neck's concat family, so
            its bytes are planes `[off, off+n)` of the pinned family
            buffer. Loading the member rather than the family is both
            correct and cheaper -- the strip gets a compact tensor of
            `t`'s own channels, and the other parts are never read."""
            root = alias_root(t)
            source = self._pin(root)
            if t is not root:
                source = self._pinned_view(t, source)
            rows = need[t.name]
            clone = self.out.copy_rows(source, rows, name=suffix(t))
            made[t.name] = tag(clone, t, rows)
            covers[t.name] = rows
            return clone

        def view_of(t: Tensor) -> Tensor:
            """Reconstruct `t`'s alias edge on top of its parent's clone
            -- a `split` view, or a `concat` that collapsed into a
            re-view. Free, exactly as in the untiled graph: the strip
            changes heights, never channels, so every plane offset
            carries over unchanged."""
            parent = clone(t.alias_parent)
            view = Tensor(
                name=suffix(t),
                height=parent.height,
                width=parent.width,
                channels=t.channels,
                quant=t.quant,
                alias_parent=parent,
                alias_plane_offset=t.alias_plane_offset,
                alias_role="slice",
            )
            self.out.tensors.append(view)
            # A view is literally its parent's rows, so it covers exactly
            # what the parent covers -- never what `need` recorded for
            # the view alone, which is only the part of it some consumer
            # asked for.
            made[t.name] = view
            covers[t.name] = covers[t.alias_parent.name]
            return view

        def concat_root(t: Tensor) -> Tensor:
            """A concat result that owns a buffer: created empty, its
            parts attached as each producer is cloned. Same shape as the
            untiled one but `need` rows tall."""
            rows = need[t.name]
            root_clone = Tensor(
                name=suffix(t),
                height=rows.rows,
                width=t.width,
                channels=t.channels,
                quant=t.quant,
            )
            self.out.tensors.append(root_clone)
            return tag(root_clone, t, rows)

        def clone(t: Tensor) -> Tensor:
            """The strip tensor standing for original tensor `t`."""
            existing = made.get(t.name)
            if existing is not None:
                return existing

            if not is_locally_built(group, t):
                # Produced outside this group (or a graph input): this
                # strip's copy of it comes in as one row-window load of
                # exactly the tensor the consumer named.
                return load_group_input(t)
            if t.alias_parts:
                built = concat_root(t)
                made[t.name] = built
                covers[t.name] = need[t.name]
                for part in t.alias_parts:
                    if not is_locally_built(group, part):
                        # Design section 3.3: a slot this group does not
                        # produce is LOADed straight into the concat
                        # buffer, with the row copy as the part's
                        # producer. Nothing new -- it is the existing
                        # "the producer writes into the slot" rule with a
                        # `RowCopyOp` as the producer.
                        loaded = load_group_input(part)
                        attach(loaded, part)
                return built
            if t.alias_parent is not None and t.alias_role == "slice":
                return view_of(t)
            raise TilingError(  # pragma: no cover - a produced tensor is cloned by its op
                f"tiler: '{t.name}' is produced inside group {group.index} but was "
                "asked for before its producer ran -- the ops are not in dataflow order"
            )

        def attach(slot: Tensor, part: Tensor) -> None:
            parent = clone(part.alias_parent)
            slot.alias_parent = parent
            slot.alias_plane_offset = part.alias_plane_offset
            slot.alias_role = "part"
            parent.alias_parts.append(slot)
            parent.alias_parts.sort(key=lambda p: p.alias_plane_offset)

        def place_in_concat(out_clone: Tensor, part: Tensor, rows: RowRange) -> None:
            """Make `part`'s strip value occupy its concat's slot.

            Two ways, and which one applies is the whole of design
            section 2.5 step 4. When the producer's strip happens to hold
            exactly the concat's rows, its clone simply *is* the part and
            writes straight into the slot -- producer-directed placement,
            unchanged. When it holds more (because something deeper in
            the chain needed a taller halo of it), the extra rows cannot
            live in the concat buffer, whose every part must be the same
            height; a `RowCopyOp` window of the concat's rows becomes the
            part instead, and the producer keeps its own taller buffer
            for the deep consumer."""
            want = need[part.alias_parent.name]
            attach(out_clone if rows == want else window(out_clone, part, want), part)

        def window(operand: Tensor, original: Tensor, want: RowRange) -> Tensor:
            """Narrow a strip buffer to exactly the rows one consumer
            wants. Needed wherever two consumers of a tensor want
            different rows (a join between chain depths) or where an op
            can only produce a rounded-out range (UPSAMPLE)."""
            have = covers[original.name]
            if have == want:
                return operand
            local = RowRange(want.r0 - have.r0, want.r1 - have.r0)
            if local.r0 < 0 or local.r1 > operand.height:
                raise TilingError(  # pragma: no cover - `need` is a hull of every consumer
                    f"tiler: strip {strip_index} of group {group.index} wants rows {want} "
                    f"of '{original.name}' but only materialized {have}"
                )
            return tag(
                self.out.copy_rows(
                    operand, local, name=f"{suffix(original)}_w{len(self.out.ops)}"
                ),
                original,
                want,
            )

        # -- clone the ops, in order ------------------------------------
        for op in group.ops:
            want_out = need[op.output.name]
            requested = input_rows(op, want_out)

            operands: list[Tensor] = []
            for original, unclipped in zip(op.inputs, requested):
                clipped = unclipped.clip(original.height)
                operands.append(window(clone(original), original, clipped))

            out_clone, produced_rows = self._clone_op(
                op, operands, requested, want_out, suffix(op.output), record
            )
            if produced_rows.rows != out_clone.height:  # pragma: no cover - defensive
                raise TilingError(
                    f"tiler: clone of '{op.name}' claims rows {produced_rows} but its "
                    f"tensor is {out_clone.height} rows tall"
                )

            tag(out_clone, op.output, produced_rows)
            made[op.output.name] = out_clone
            covers[op.output.name] = produced_rows
            if op.output.alias_role == "part" and op.output.alias_parent.name in need:
                # ...but only when *this* group is the one that
                # assembles that concat. The same tensor is a concat part
                # in the group that reads the concat and an ordinary
                # standalone output in the group that produces it -- the
                # neck's skip connection is produced two groups before
                # the merge that concatenates it, and there its buffer is
                # its own. `need` names exactly the tensors this strip
                # materializes, so a parent missing from it is a concat
                # that lives somewhere else.
                place_in_concat(out_clone, op.output, produced_rows)

        # -- store the group's outputs back to their pinned buffers -----
        for member in group_outputs:
            operand = made[member.name]
            have = covers[member.name]
            local = RowRange(out_rows.r0 - have.r0, out_rows.r1 - have.r0)
            root = alias_root(member)
            destination = self._pin(root)
            if member is not root:
                # The member is a slot of a pinned concat family: store
                # into its planes, at its row offset. `case_c2f_concat_
                # in_ddr` already proves "producers write slices of a DDR
                # concat"; a strip adds the row offset and nothing else.
                destination = self._pinned_view(member, destination)
            self.stored.append(destination)
            self.out.copy_rows(
                operand,
                local,
                into=destination,
                at=out_rows,
                name=f"{suffix(member)}_store",
            )

        self.records.append(record)

    # -- backward pass ---------------------------------------------------

    # -- op cloning ------------------------------------------------------

    def _clone_op(
        self,
        op: Op,
        operands: list[Tensor],
        requested: list[RowRange],
        want_out: RowRange,
        name: str,
        record: StripRecord,
    ) -> tuple[Tensor, RowRange]:
        """Build the strip's copy of `op` and return `(output tensor,
        the rows of the original output it holds)`."""
        if isinstance(op, (Conv2dOp, PoolOp)):
            pad_top, pad_bottom = _recut_padding(op, requested[0], op.inputs[0].height)
            record.padding[op.name] = (pad_top, pad_bottom)
            produced = output_rows(op, requested[0].clip(op.inputs[0].height),
                                   pad_top=pad_top, pad_bottom=pad_bottom)
            # The geometry assertion (design section 2.5 step 3): what
            # the descriptor fields will make the hardware produce must
            # be what the recurrence said this strip needs. An
            # off-by-one in `input_rows` shows up here, at build time,
            # with no address in sight.
            if produced != want_out:
                raise TilingError(
                    f"tiler: clone of '{op.name}' on input rows "
                    f"{requested[0].clip(op.inputs[0].height)} with pad "
                    f"({pad_top}, {pad_bottom}) produces output rows {produced}, but the "
                    f"row-range recurrence says this strip needs {want_out}. One of the "
                    "two is wrong; they are computed independently precisely so that this "
                    "cannot pass silently."
                )
            return self._clone_shaped(op, operands[0], pad_top, pad_bottom, name), produced

        if isinstance(op, AddOp):
            builder = self.out.add_planewise if self.planewise_add else self.out.add
            out = builder(
                operands[0],
                operands[1],
                requant_scale=op.requant_scale,
                requant_shift=op.requant_shift,
                name=name,
            )
            return out, want_out

        if isinstance(op, UpsampleOp):
            out = self.out.upsample2x(operands[0], name=name)
            produced = output_rows(op, requested[0].clip(op.inputs[0].height), pad_top=0, pad_bottom=0)
            return out, produced

        if isinstance(op, ActOp):
            return self.out.act(operands[0], op.lut, name=name), want_out

        if isinstance(op, CopyOp):
            return self.out.copy(operands[0], name=name), want_out

        raise TilingError(  # pragma: no cover - `input_rows` rejects first
            f"tiler: cannot clone op '{op.name}' of type {type(op).__name__}"
        )

    def _clone_shaped(
        self, op: Op, operand: Tensor, pad_top: int, pad_bottom: int, name: str
    ) -> Tensor:
        """Clone a `CONV2D`/`POOL_*` with recut vertical padding.

        Built as a dataclass directly rather than through
        `Model.conv2d`, because that method draws fresh weights from the
        RNG and a strip must reuse the *original* weight/bias/scale
        objects byte for byte -- both so that every strip of a group
        computes the same convolution, and so that the tiled program can
        be compared against the untiled one at all. `weight_reuse` is
        never set: it skips the weight refill for every output-channel
        pass, not just the first (R8), so it is only correct for a
        single-pass conv and is worth exactly nothing here."""
        _, _, pad_left, pad_right = op.padding
        padding = (pad_top, pad_bottom, pad_left, pad_right)
        if isinstance(op, Conv2dOp):
            out_h = (operand.height + pad_top + pad_bottom - op.kernel[0]) // op.stride[0] + 1
            out_w = (operand.width + pad_left + pad_right - op.kernel[1]) // op.stride[1] + 1
            out_t = Tensor(
                name=name,
                height=out_h,
                width=out_w,
                channels=op.output.channels,
                quant=op.output.quant,
            )
            clone = Conv2dOp(
                name=name,
                inputs=[operand],
                output=out_t,
                weight=op.weight,
                bias=op.bias,
                kernel=op.kernel,
                stride=op.stride,
                padding=padding,
                activation=op.activation,
                clamp=op.clamp,
                requant_scale=op.requant_scale,
                requant_shift=op.requant_shift,
                output_offset=op.output_offset,
                per_channel_scale=op.per_channel_scale,
                pad_value=op.pad_value,
                # Every strip's clone shares `op.weight`, so `planner.py`
                # gives all `S` of them one scratchpad image and loads it
                # once -- which is the point of the flag. See `tile`'s
                # `resident_groups`.
                weights_resident=self._resident,
            )
        else:
            assert isinstance(op, PoolOp)
            out_h = (operand.height + pad_top + pad_bottom - op.kernel[0]) // op.stride[0] + 1
            out_w = (operand.width + pad_left + pad_right - op.kernel[1]) // op.stride[1] + 1
            out_t = Tensor(
                name=name,
                height=out_h,
                width=out_w,
                channels=op.output.channels,
                quant=op.output.quant,
            )
            clone = PoolOp(
                name=name,
                inputs=[operand],
                output=out_t,
                mode=op.mode,
                kernel=op.kernel,
                stride=op.stride,
                padding=padding,
                pad_value=op.pad_value,
                activation=op.activation,
                clamp=op.clamp,
                requant_scale=op.requant_scale,
                requant_shift=op.requant_shift,
                output_offset=op.output_offset,
            )
        out_t.producer = clone
        self.out.register_prebuilt_op(clone)
        return out_t


__all__ = [
    "TilingError",
    "Group",
    "StripRecord",
    "TiledModel",
    "dissolve",
    "form_groups",
    "group_inputs",
    "group_outputs",
    "is_locally_built",
    "input_rows",
    "output_rows",
    "propagate_rows",
    "probe_strip",
    "strip_bounds",
    "tile",
]
