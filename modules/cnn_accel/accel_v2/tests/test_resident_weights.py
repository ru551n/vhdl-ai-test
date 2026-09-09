"""`space_wgt = LOCAL_TENSOR`: a group's packed weight/bias/scale images
LOADed into the scratchpad once instead of re-read from DDR per strip
(design D6, hardware fact F7, implementation plan step 2).

The hardware has always supported this -- `cmd_proc.vhd`'s `st_wgt_req`
routes the refill to tensor-memory port `r1` when `space_wgt` names
`LOCAL_TENSOR` -- and nothing had ever emitted it. These tests pin the
lowering; `accel_v2/cases_tiling.py::case_resident_weights_strip_pair`
is the same path on the DUT, which is the half that can tell a wrong
local weight address from a right one (see `reference.py`'s
`ConstLoadStep` branch for why this side deliberately cannot).

What is asserted here is the *arithmetic and the geometry*: how many
images, how big, where, with what confinement unit, and what they cost --
each derived from the packers and the hardware's own request sizes, never
read back out of the plan.
"""

from __future__ import annotations

import cnn_accel_constants as const
import cnn_accel_model as golden

from accel_v2 import isa
from accel_v2.ddrmap import DdrMap
from accel_v2.memimage import MemoryImage
from accel_v2.model import Conv2dOp, Model
from accel_v2.planner import (
    ComputeStep,
    ConstLoadStep,
    Planner,
    const_images,
    descriptor_count,
)
from accel_v2.program import emit_program
from accel_v2.reference import run_reference
from accel_v2.tiler import tile
from accel_v2.tiling_checks import check_units_confined

#: 16 rows so a two-strip cut is exact; 32 output channels so the
#: convolution runs `n_ot = 4` output-channel passes, which is what makes
#: "one tile per pass" a real address sequence rather than a single fetch.
_H, _W, _CIN, _COUT = 16, 8, 8, 32


def _model() -> Model:
    m = Model(seed=77)
    x = m.input(_H, _W, _CIN, name="x")
    # `per_channel` on, so all three side tables exist and the scale
    # image is exercised too -- it is the one with the smallest
    # per-pass request and therefore the easiest to misplace.
    y = m.conv2d(
        x, _COUT, kernel=(3, 3), padding=(1, 1, 1, 1), per_channel=True, name="y"
    )
    m.output(y)
    return m


def _plan(strips: int, *, resident: bool, banks: int = 4, bank_bytes: int = 8 * 1024):
    model = _model()
    tiled = tile(model, strips, resident_groups={0} if resident else set())
    planner = Planner(
        tensor_mem_bytes=banks * bank_bytes,
        bank_bytes=bank_bytes,
        ddr_map=DdrMap(scale=4),
    )
    return tiled, planner.plan(tiled.model)


def _conv_of(planned) -> Conv2dOp:
    return next(
        s.op
        for s in planned.steps
        if isinstance(s, ComputeStep) and isinstance(s.op, Conv2dOp)
    )


def _image_bytes() -> int:
    """Total packed side-table bytes of one `y` convolution, from the
    packers themselves."""
    desc = golden.LayerDesc(
        opcode=golden.OPCODE_CONV2D,
        in_channels=_CIN,
        out_channels=_COUT,
        kernel_h=3,
        kernel_w=3,
    )
    return (
        golden.packed_weight_count(desc, golden.TILE_CHANNELS, golden.PE_ROWS)
        + golden.packed_bias_count(desc, golden.PE_ROWS) * (const.ACCUM_WIDTH // 8)
        + golden.packed_scale_table_bytes(desc, golden.PE_ROWS)
    )


def test_the_three_images_have_the_sizes_and_request_units_the_hardware_uses() -> None:
    """`const_images` is what the allocator confines against, so it has
    to describe the hardware's real request sequence: one weight *tile*
    per output-channel pass, one `PE_ROWS`-entry bias row, one
    `PE_ROWS`-entry scale row -- all from `st_wgt_req`'s own addressing,
    re-derived here rather than read from the function under test."""
    model = _model()
    conv = next(op for op in model.ops if isinstance(op, Conv2dOp))
    n_ot = _COUT // golden.PE_ROWS
    n_tiles = _CIN // golden.TILE_CHANNELS

    images = dict((kind, (total, unit)) for kind, total, unit in const_images(conv))
    assert set(images) == {"weight", "bias", "scale"}

    # wgt_tile_bytes = T * k_h * k_w * pe_rows * pe_cols (cmd_proc's
    # st_geom_out2), and there are n_ot of them.
    tile_bytes = n_tiles * 3 * 3 * golden.PE_ROWS * const.PE_COLS
    assert images["weight"] == (n_ot * tile_bytes, tile_bytes)
    assert images["bias"] == (
        n_ot * golden.PE_ROWS * (const.ACCUM_WIDTH // 8),
        golden.PE_ROWS * (const.ACCUM_WIDTH // 8),
    )
    assert images["scale"] == (
        n_ot * golden.PE_ROWS * const.SCALE_TABLE_ENTRY_BYTES,
        golden.PE_ROWS * const.SCALE_TABLE_ENTRY_BYTES,
    )
    assert sum(total for total, _ in images.values()) == _image_bytes()


def test_all_strips_share_one_image_loaded_once() -> None:
    """The saving *is* the sharing: `S` strip clones of one conv share
    its `weight` list, so there must be exactly one `ConstLoadStep` no
    matter how many strips, and it must move the image exactly once."""
    for strips in (2, 4, 8):
        _, planned = _plan(strips, resident=True)
        loads = [s for s in planned.steps if isinstance(s, ConstLoadStep)]
        assert len(loads) == 1, strips
        assert loads[0].nbytes == _image_bytes()
        assert [kind for kind, _, _ in loads[0].images] == ["weight", "bias", "scale"]

        convs = [
            s
            for s in planned.steps
            if isinstance(s, ComputeStep) and isinstance(s.op, Conv2dOp)
        ]
        assert len(convs) == strips
        assert all(s.weight_space == isa.SPACE_LOCAL_TENSOR for s in convs)
        # Every strip fetches from the same three addresses -- one image.
        assert len({tuple(sorted(s.weight_addrs.items())) for s in convs}) == 1


def test_residency_removes_exactly_the_per_strip_weight_re_read() -> None:
    """Closed form: with `S` strips, DDR weight traffic is `S * W` when
    the images live in DDR and `W` when they are resident. Nothing else
    about the two programs differs, so the read difference is exactly
    `(S-1) * W` -- asserted against `W` computed from the packers, not
    from either plan."""
    image = _image_bytes()
    for strips in (2, 4):
        _, ddr = _plan(strips, resident=False)
        _, local = _plan(strips, resident=True)

        # ...less the program fetch of the three extra `LOAD` descriptors
        # the resident plan carries, which is the only other difference
        # between the two programs.
        extra_descs = len(const_images(_conv_of(local))) * isa.INSTR_WORD_BYTES
        assert (
            ddr.traffic.read_bytes - local.traffic.read_bytes
            == (strips - 1) * image - extra_descs
        )
        # `WEIGHT_LOAD_BYTES` is unchanged: the hardware refills the
        # weight buffer once per pass whichever space it reads, and
        # `cnt_wgt_bytes_q` increments before the space mux. Residency
        # moves the bytes from AXI to the scratchpad's r1 port, it does
        # not remove the refill.
        assert local.traffic.weight_bytes == ddr.traffic.weight_bytes == strips * image
        assert (
            local.traffic.local_read_bytes - ddr.traffic.local_read_bytes
            == strips * image
        )


def test_the_planner_and_the_reference_agree_on_resident_weight_traffic() -> None:
    for strips in (1, 2, 4):
        _, planned = _plan(strips, resident=True)
        actual = run_reference(planned, MemoryImage())
        assert vars(planned.traffic) == vars(actual.traffic), strips


def test_the_emitted_descriptors_name_local_tensor_and_the_reserved_addresses() -> None:
    """The lowering, checked on the encoded descriptors: three `LOAD`s
    into the scratchpad, then convolutions whose `space_wgt` is
    `LOCAL_TENSOR` and whose weight/bias/scale addresses are the ones
    those `LOAD`s wrote to."""
    _, planned = _plan(2, resident=True)
    program = emit_program(planned)

    load = next(s for s in planned.steps if isinstance(s, ConstLoadStep))
    want = {kind: addr for kind, addr, _ in load.images}

    # The image `LOAD`s are the descriptors of that one step, found by
    # position -- a strip's own input load is also a DDR->LOCAL `LOAD`,
    # so filtering by opcode alone would sweep those in too.
    first = sum(descriptor_count(s) for s in planned.steps[: planned.steps.index(load)])
    loads = program.descs[first : first + len(load.images)]
    assert all(d.opcode == isa.OPCODE_LOAD for d in loads)
    assert all(d.space_src0 == isa.SPACE_DDR for d in loads)
    assert all(d.space_dst == isa.SPACE_LOCAL_TENSOR for d in loads)
    assert [d.out_addr for d in loads] == [addr for _, addr, _ in load.images]
    assert sum(d.xfer_bytes for d in loads) == _image_bytes()

    convs = [d for d in program.descs if d.opcode == isa.OPCODE_CONV2D]
    assert convs
    for d in convs:
        assert d.space_wgt == isa.SPACE_LOCAL_TENSOR
        assert (d.weight_addr, d.bias_addr, d.scale_addr) == (
            want["weight"],
            want["bias"],
            want["scale"],
        )
        # Every address `st_wgt_req` will form must be 8-byte aligned:
        # the descriptor's own alignment check covers only pass 0.
        assert d.weight_addr % 8 == d.bias_addr % 8 == d.scale_addr % 8 == 0


def test_every_per_pass_weight_request_stays_inside_one_bank() -> None:
    """R5 for the weight image. `cnn_accel_tensor_mem` clamps a request
    that runs past the end of the bank its address decodes to, so a tile
    straddling a boundary would be silently truncated -- and
    `reference.py` never touches these bytes at all, so nothing but this
    geometry check can see it."""
    for bank_words in (256, 512, 1024):
        bank = bank_words * 8
        _, planned = _plan(2, resident=True, banks=8, bank_bytes=bank)
        check_units_confined(planned)
        # ...and independently: walk the requests the hardware will make.
        by_name = dict(planned.local_confine_units)
        for name, addr, size in planned.local_placements:
            if "#const#" not in name:
                continue
            unit = by_name.get(name, size)
            for k in range(size // unit):
                lo = addr + k * unit
                assert lo // bank == (lo + unit - 1) // bank, (name, bank, k)


def test_an_image_that_cannot_be_placed_falls_back_to_ddr_weights() -> None:
    """Residency is a request, not a promise. A scratchpad too small for
    the image must produce a correct DDR-weight program, with the traffic
    charged where it is actually paid -- not a plan that claims a saving
    it did not get."""
    image = _image_bytes()
    _, planned = _plan(2, resident=True, banks=2, bank_bytes=1024)
    assert not [s for s in planned.steps if isinstance(s, ConstLoadStep)]
    convs = [
        s for s in planned.steps if isinstance(s, ComputeStep) and isinstance(s.op, Conv2dOp)
    ]
    assert convs and all(s.weight_space == isa.SPACE_DDR for s in convs)

    _, honest = _plan(2, resident=False, banks=2, bank_bytes=1024)
    assert planned.traffic.read_bytes == honest.traffic.read_bytes
    assert planned.traffic.read_bytes >= 2 * image

    actual = run_reference(planned, MemoryImage())
    assert vars(planned.traffic) == vars(actual.traffic)
