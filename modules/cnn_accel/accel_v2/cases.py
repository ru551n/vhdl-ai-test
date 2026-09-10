"""The `tb_cnn_accel_top` test-case catalogue.

One function per case, each returning a `TbCase`. `module_cnn_accel.py`
turns every entry of `ALL_CASES` into one VUnit config of the single
testbench's single test, so adding a test means adding a function here
and never touching VHDL (arch doc section 11).

Every case names its own `seed`, so a failure is reproducible from the
config name alone.

Shapes are kept deliberately small. These are *integration* tests: the
arithmetic of each engine is already pinned bit-exactly by its own
unit-level testbench (`tb_cnn_accel_conv_core`, `tb_cnn_accel_pool`,
`tb_cnn_accel_bias_requant`, ...) against the same golden model. What is
under test here is the command processor, the storage model and the
residency policy, and none of that gets more thoroughly exercised by a
224x224 tensor -- only slower.
"""

from __future__ import annotations

from accel_v2.model import Activation, Model
from accel_v2.tbcase import TbCase, TrafficPolicy, build_case

# Small enough to keep simulation short, but not degenerate: 8 input
# channels is exactly one activation-plane word (TILE_CHANNELS), and an
# 8x8 plane is wide enough that row tiling and padding both do real work.
_H = 8
_W = 8
_C = 8


# ---------------------------------------------------------------------------
# 1. Basic memory / control: the smallest thing that is still a program.
# ---------------------------------------------------------------------------


def case_single_conv() -> TbCase:
    """One 3x3 convolution, DDR in -> DDR out.

    The minimal end-to-end path: descriptor fetch, weight/bias fetch,
    a DDR-operand read, the conv engine, and a DDR writeback. Its value
    is that *every* later case's failure can be triaged against it: if
    this passes and a residency case fails, the bug is in the storage
    model, not in the datapath or the fetch path.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        y = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="y")
        model.output(y)

    return build_case("single_conv", build, seed=1)


# ---------------------------------------------------------------------------
# 2. Local residency: the reason rev 2 exists.
# ---------------------------------------------------------------------------


def case_local_chain_2op() -> TbCase:
    """Two chained convolutions whose intermediate stays in the
    scratchpad -- the rev-2 milestone case.

    The planner lowers this to exactly two compute descriptors and no
    `LOAD`/`STORE` at all: the first conv reads its DDR input operand
    directly and writes `LOCAL_TENSOR`, the second reads that local
    buffer and writes the graph output to DDR. So `DDR_WR_BYTES` must
    equal the final output's size *exactly* -- invariants R1, R2 and R4
    of arch doc section 10 in one program. `TrafficPolicy`'s default
    `write_bytes_exact` is what enforces it, cross-checked against the
    testbench's own AXI W-channel monitor.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="h")
        y = model.conv2d(h, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="y")
        model.output(y)

    return build_case("local_chain_2op", build, seed=2)


def case_local_chain_4op() -> TbCase:
    """Four chained convolutions. Same invariant as the two-op chain, but
    long enough that a per-command writeback bug (one that only shows up
    from the second intermediate onwards, e.g. a buffer that is written
    back when it is *reused* rather than when it is created) cannot hide:
    `DDR_WR_BYTES` still has to be exactly one output tensor."""

    def build(model: Model) -> None:
        t = model.input(_H, _W, _C, name="x")
        for i in range(4):
            t = model.conv2d(t, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name=f"h{i}")
        model.output(t)

    return build_case("local_chain_4op", build, seed=3)


def case_explicit_load() -> TbCase:
    """An input with two consumers, which makes the planner emit a real
    `LOAD` descriptor (a single-consumer input is read straight out of
    DDR instead -- see `Planner.resolve_input`). Proves the `LOAD` path
    and `TENSOR_LOAD_COUNT` independently of any spill."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        a = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="a")
        b = model.conv2d(x, _C, kernel=(1, 1), name="b")
        y = model.add(a, b, name="y")
        model.output(y)

    return build_case("explicit_load", build, seed=4)


# ---------------------------------------------------------------------------
# 3. Elementwise / residual (H5's dedicated ADD unit).
# ---------------------------------------------------------------------------


def case_residual_add() -> TbCase:
    """A residual block: the tensor produced at command `i` is consumed
    again at command `i+k` with an unrelated command in between, and must
    survive untouched in the scratchpad (invariants R3 and R4). Zero DDR
    writes until the closing output."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        skip = model.conv2d(x, _C, kernel=(1, 1), name="skip")
        mid = model.conv2d(skip, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="mid")
        y = model.add(mid, skip, name="y")
        model.output(y)

    return build_case("residual_add", build, seed=5)


# ---------------------------------------------------------------------------
# 4. Spill / reload: DDR round trips that are explicit in the program.
# ---------------------------------------------------------------------------


def case_forced_spill() -> TbCase:
    """A scratchpad deliberately too small for the live set, which forces
    the planner to insert an explicit spill `STORE` and a matching reload
    `LOAD`. Verifies invariant R5: bit-exact through the round trip, and
    `TENSOR_LOAD_COUNT`/`TENSOR_STORE_COUNT` equal to the program's own
    explicit counts -- no hidden traffic in either direction.

    The geometry is chosen to make the spill unavoidable rather than
    incidental: the scratchpad is 1024 bytes -- exactly two 8x8x8
    tensors -- while the residual keeps `skip` live across two
    intermediates. At the third allocation the planner must evict
    something, and Belady's rule picks `skip` (its next use is furthest
    away), so `skip` is spilled and then reloaded for the `add`.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        skip = model.conv2d(x, _C, kernel=(1, 1), name="skip")
        mid1 = model.conv2d(skip, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="mid1")
        mid2 = model.conv2d(mid1, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="mid2")
        y = model.add(mid2, skip, name="y")
        model.output(y)

    return build_case("forced_spill", build, seed=6, num_banks=1, bank_words=128)


def case_bank_boundary_straddle() -> TbCase:
    """Two live buffers whose size does not divide the bank size, in a
    scratchpad of two 1 KiB banks -- the geometry a *flat* (bank-unaware)
    allocator gets wrong.

    `h1` and `a` are 8x12x8 = 768 bytes each and both live across the
    closing `add`. A flat first-fit allocator puts `h1` at 0 and `a` at
    768, so `a` spans 768..1536 and straddles the boundary between bank 0
    and bank 1. `cnn_accel_tensor_mem` serves each transfer from exactly
    one bank and clamps anything that would run past its end, so every
    access to `a` would have been silently truncated to its first 256
    bytes -- with no error anywhere, because the descriptor's address and
    `reference.py`'s address agree (they come from the same planner) and
    only the hardware knows the boundary is there.

    The bank-aware planner places `a` in bank 1 instead; the 256-byte
    hole below the boundary stays on the free list. This case runs that
    placement against the real RTL, so a regression shows up as either a
    data mismatch or -- since the crossing assertion is severity
    'failure' -- an aborted simulation, not as a quietly passing test.

    `bank_words=128` (1 KiB banks, 2 KiB total) rather than the default
    8 KiB banks: the straddle has to be reachable with tensors small
    enough to keep the simulation short.
    """

    def build(model: Model) -> None:
        x = model.input(8, 12, _C, name="x")
        h1 = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="h1")
        a = model.conv2d(h1, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="a")
        y = model.add(a, h1, name="y")
        model.output(y)

    return build_case("bank_boundary_straddle", build, seed=77, num_banks=2, bank_words=128)


# ---------------------------------------------------------------------------
# 5. Feature coverage: one case per engine path, all local-resident.
# ---------------------------------------------------------------------------


def case_conv_stride2() -> TbCase:
    """Strided convolution, no padding -- exercises the output-geometry
    arithmetic in `cmd_proc`'s validation and in `window_gen`."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        y = model.conv2d(x, _C, kernel=(3, 3), stride=(2, 2), name="y")
        model.output(y)

    return build_case("conv_stride2", build, seed=7)


def case_conv_1x1_no_bias() -> TbCase:
    """1x1 convolution with the bias flag clear: the descriptor's
    `bias_addr` must be ignored rather than fetched, so
    `WEIGHT_LOAD_BYTES` covers weights only."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        y = model.conv2d(x, _C, kernel=(1, 1), bias=False, activation=Activation.NONE, name="y")
        model.output(y)

    return build_case("conv_1x1_no_bias", build, seed=8)


def case_conv_per_channel_scale() -> TbCase:
    """Per-channel requantization: a scale table is fetched from DDR and
    counted into `WEIGHT_LOAD_BYTES` alongside weights and bias."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        y = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), per_channel=True, name="y")
        model.output(y)

    return build_case("conv_per_channel_scale", build, seed=9)


def case_pool_max() -> TbCase:
    """2x2/2 max pooling on a local tensor."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="h")
        y = model.pool_max(h, name="y")
        model.output(y)

    return build_case("pool_max", build, seed=10)


def case_upsample() -> TbCase:
    """2x nearest-neighbour upsample of a local tensor (H3)."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="h")
        y = model.upsample2x(h, name="y")
        model.output(y)

    return build_case("upsample", build, seed=11)


def case_depth_to_space() -> TbCase:
    """`DEPTH_TO_SPACE` (ISA v2.2): ESPCN's sub-pixel-convolution tail --
    a conv that widens 8 channels to `factor**2 * 8 = 32`, then the
    pixel-shuffle that trades those back for a 2x larger frame.

    Exactly ONE output channel tile, so this isolates the inner
    `(dy, dx)` sweep and the destination row-base recurrence from the
    outer `c_tile_out` loop -- which `case_depth_to_space_two_tiles`
    below then adds on top."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, 4 * _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="h")
        y = model.depth_to_space(h, name="y")
        model.output(y)

    return build_case("depth_to_space", build, seed=15)


def case_depth_to_space_two_tiles() -> TbCase:
    """`DEPTH_TO_SPACE` over TWO output channel tiles, on a deliberately
    NON-SQUARE frame (4 rows x 8 columns).

    Two things `case_depth_to_space` structurally cannot catch:

     * the `c_tile_out` loop and the source plane stride. With one output
       tile the stride term `plane * n_tiles_out * in_h * in_w` is
       indistinguishable from `plane * in_h * in_w`, so a missing
       `n_tiles_out` factor passes; with two, it does not.
     * a transposed `in_width`/`in_height`. On the 8x8 frames every other
       case uses, swapping the two is invisible in both the address
       arithmetic and the golden comparison."""

    def build(model: Model) -> None:
        x = model.input(4, _W, _C, name="x")
        h = model.conv2d(x, 8 * _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="h")
        y = model.depth_to_space(h, name="y")
        model.output(y)

    return build_case("depth_to_space_two_tiles", build, seed=16)


def case_copy() -> TbCase:
    """`COPY` between two local buffers: pure data movement through the
    elementwise engine, with no arithmetic to hide a layout bug behind."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="h")
        y = model.copy(h, name="y")
        model.output(y)

    return build_case("copy", build, seed=12)


def case_act_lut() -> TbCase:
    """`ACT` with a 256-entry int8->int8 lookup table fetched from DDR
    (H1). The LUT address rides in the descriptor's `weight_addr` field
    (arch doc section 5.1)."""

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="h")
        y = model.act(h, name="y")
        model.output(y)

    return build_case("act_lut", build, seed=13)


# ---------------------------------------------------------------------------
# 6. Multi-op network: everything above in one program.
# ---------------------------------------------------------------------------


def case_multi_op_network() -> TbCase:
    """A small but genuinely mixed network -- conv, pool, upsample,
    residual add and an activation LUT -- run as a single program with
    every intermediate local. The only DDR write is the final output.

    This is the case that catches interaction bugs the single-feature
    cases cannot: an engine that leaves the scratchpad's arbitration in a
    bad state, a weight buffer not re-filled after a non-conv command, a
    counter that only increments for the first command of its kind.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        stem = model.conv2d(x, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="stem")
        down = model.pool_max(stem, name="down")
        deep = model.conv2d(down, _C, kernel=(3, 3), padding=(1, 1, 1, 1), name="deep")
        up = model.upsample2x(deep, name="up")
        fused = model.add(up, stem, name="fused")
        y = model.act(fused, name="y")
        model.output(y)

    return build_case("multi_op_network", build, seed=14)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

#: Every case, in the order they are registered as VUnit configs.
CASE_BUILDERS = (
    case_single_conv,
    case_local_chain_2op,
    case_local_chain_4op,
    case_explicit_load,
    case_residual_add,
    case_forced_spill,
    case_bank_boundary_straddle,
    case_conv_stride2,
    case_conv_1x1_no_bias,
    case_conv_per_channel_scale,
    case_pool_max,
    case_upsample,
    case_depth_to_space,
    case_depth_to_space_two_tiles,
    case_copy,
    case_act_lut,
    case_multi_op_network,
)


def all_cases() -> list[TbCase]:
    """Build every case. Each call builds fresh objects (each with its own
    `DdrMap`), so cases never share DDR allocation state."""
    return [builder() for builder in CASE_BUILDERS]


__all__ = ["CASE_BUILDERS", "TbCase", "TrafficPolicy", "all_cases"]
