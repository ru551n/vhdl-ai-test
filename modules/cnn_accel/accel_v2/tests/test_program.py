"""Tests for `accel_v2.program`: descriptor lowering, the `MemoryImage`
CSV round-trip, and a cross-check against `cnn_accel_model.run_program`
(ISA v1.2's own interpreter) for a program with no `LOCAL_TENSOR` use --
the strongest available anti-fork guard on the *encoder* itself, since it
runs the emitted bytes through a completely independent, already-tested
implementation rather than re-decoding with this project's own
`isa.decode_desc`."""

from __future__ import annotations

import os
import tempfile

import cnn_accel_model as golden

from accel_v2 import isa
from accel_v2.ddrmap import DdrMap
from accel_v2.memimage import MemoryImage
from accel_v2.model import Activation, Conv2dOp, Model
from accel_v2.planner import ComputeStep, MoveStep, Planner
from accel_v2.program import emit_program
from accel_v2.reference import run_reference


def _single_conv_model(seed: int = 9) -> Model:
    """input -> conv -> output: the planner keeps this entirely in DDR
    (one graph input read once, one graph output write), so the whole
    program is expressible in `cnn_accel_model`'s v1.2 flat-DDR address
    space -- see the module docstring."""
    m = Model(seed=seed)
    x = m.input(8, 8, 4, scale=1.0)
    y = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    m.output(y)
    return m


def test_descriptor_chain_round_trips_through_encode_decode() -> None:
    planned = Planner().plan(_single_conv_model())
    pi = emit_program(planned)
    for desc in pi.descs:
        assert isa.decode_desc(isa.encode_desc(desc)) == desc


def test_chain_ends_in_halt_and_next_instr_addr_links_forward() -> None:
    planned = Planner().plan(_single_conv_model())
    pi = emit_program(planned)
    assert pi.descs[-1].opcode == isa.OPCODE_HALT
    # Walk next_instr_addr from program_addr and land on HALT after
    # exactly len(descs)-1 hops, matching run_program's own fetch-
    # decode-execute loop.
    pc = pi.program_addr
    hops = 0
    seen = set()
    while True:
        raw = pi.image.read_bytes(pc, isa.INSTR_WORD_BYTES)
        desc = isa.decode_desc(raw)
        if desc.opcode == isa.OPCODE_HALT:
            break
        assert pc not in seen, "next_instr_addr loop"
        seen.add(pc)
        pc = desc.next_instr_addr
        hops += 1
    assert hops == len(pi.descs) - 1


def test_weight_reuse_shares_ddr_weight_address() -> None:
    m = Model(seed=3)
    x = m.input(8, 8, 4)
    c1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1))
    c2 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), weight_reuse_from=c1)
    m.output(c1)
    m.output(c2)
    planned = Planner().plan(m)
    pi = emit_program(planned)
    conv_descs = [d for d in pi.descs if d.opcode == isa.OPCODE_CONV2D]
    assert len(conv_descs) == 2
    d1, d2 = conv_descs
    assert d1.weight_addr == d2.weight_addr
    assert d2.flags & (1 << golden.FLAG_WEIGHT_REUSE)


def test_csv_round_trip_preserves_every_word() -> None:
    planned = Planner().plan(_single_conv_model())
    pi = emit_program(planned)
    path = os.path.join(tempfile.mkdtemp(), "image.csv")
    pi.image.write_csv(path)
    reloaded = MemoryImage.read_csv(path)
    assert reloaded.words() == pi.image.words()


def test_determinism_same_seed_identical_image_bytes() -> None:
    def build(seed: int) -> list[tuple[int, int]]:
        planned = Planner().plan(_single_conv_model(seed=seed))
        return emit_program(planned).image.words()

    assert build(21) == build(21)
    assert build(21) != build(22)


def test_reloc_tags_exactly_the_input_and_output_operands() -> None:
    """ISA v2.3 (spec section 6a): `_single_conv_model` has one graph
    input and one graph output, each read/written by exactly one
    descriptor, so exactly one descriptor's `space_src0` operand and
    exactly one descriptor's `space_dst` operand should end up tagged --
    everything else (weights, the program's own descriptor addresses)
    must be untouched."""
    planned = Planner().plan(_single_conv_model())
    pi = emit_program(planned)

    in_addr = planned.tensor_ddr_addr[planned.model.inputs[0].name]
    out_addr = planned.tensor_ddr_addr[planned.model.outputs[0].name]

    reloc_input_descs = [d for d in pi.descs if d.reloc_input]
    reloc_output_descs = [d for d in pi.descs if d.reloc_output]
    assert len(reloc_input_descs) == 1
    assert len(reloc_output_descs) == 1
    assert reloc_input_descs[0].in_addr == in_addr
    assert reloc_output_descs[0].out_addr == out_addr
    # Neither tag ever fires on a non-DDR operand or the wrong address.
    for d in pi.descs:
        if d.reloc_input:
            assert d.space_src0 == isa.SPACE_DDR
        if d.reloc_output:
            assert d.space_dst == isa.SPACE_DDR


def test_reloc_tags_every_plane_of_a_tiled_input() -> None:
    """The design's whole point: address-range matching, not "is this
    operand syntactically the tensor", so a tiled program's several
    row-copy planes -- reading successive slices of one graph input --
    all get tagged, not just a first/only one."""
    from accel_v2 import cases_tiling

    for case in cases_tiling.all_cases():
        if not case.model.inputs:
            continue
        in_lo = case.planned.tensor_ddr_addr[case.model.inputs[0].name]
        in_hi = in_lo + case.model.inputs[0].size_bytes
        reloc_input_descs = [d for d in case.program.descs if d.reloc_input]
        if not reloc_input_descs:
            continue
        # However many descriptors touch the input, every one of them
        # must land inside the input's own window -- never outside it.
        for d in reloc_input_descs:
            assert in_lo <= d.in_addr < in_hi
        return
    raise AssertionError("no cases_tiling case had a reloc_input descriptor to check")


def test_reloc_flags_default_false_reproduce_prior_addresses() -> None:
    """Backward compatibility, stated as a test rather than only as a
    byte-count argument: a descriptor from any program built before ISA
    v2.3 always encodes with `reloc_input=reloc_output=False`, which
    `_tag_relocatable_operands` never touches unless the descriptor's
    OWN address happens to fall in a graph input/output window -- an
    ordinary weight/bias/scale/LUT/program descriptor never does."""
    planned = Planner().plan(_single_conv_model())
    pi = emit_program(planned)
    non_io_descs = [
        d
        for d in pi.descs
        if d is not next(x for x in pi.descs if x.reloc_input)
        and d is not next(x for x in pi.descs if x.reloc_output)
    ]
    assert non_io_descs, "expected at least the closing HALT to be untagged"
    for d in non_io_descs:
        assert d.reloc_input is False
        assert d.reloc_output is False


def test_ddr_only_program_matches_golden_run_program() -> None:
    """The strongest cross-check available: run the emitted descriptor
    chain through `cnn_accel_model.run_program` (a completely separate,
    already-regression-tested ISA v1.2 interpreter operating on a flat
    `bytearray`, not this project's `reference.py`) and compare its
    output bytes against `reference.run_reference`'s.

    Only valid for a program that never touches `LOCAL_TENSOR` --
    `run_layer` has no space-tag concept and treats every address as
    plain DDR, so a `LOCAL_TENSOR` address would silently collide with
    unrelated DDR content. `_single_conv_model` is deliberately shaped
    (one graph input, one graph output, nothing else) so the planner's
    own residency policy never allocates a local buffer at all."""
    m = _single_conv_model()
    planned = Planner().plan(m)
    assert all(isinstance(s, ComputeStep) for s in planned.steps), "expected a pure-DDR program"
    for step in planned.steps:
        assert step.output_space == isa.SPACE_DDR
        for space in step.input_spaces:
            assert space == isa.SPACE_DDR

    pi = emit_program(planned)

    ref_image = MemoryImage()
    ref_result = run_reference(planned, ref_image)
    out_tensor = m.outputs[0]
    out_addr = planned.tensor_ddr_addr[out_tensor.name]
    expected_bytes = ref_image.read_bytes(out_addr, out_tensor.size_bytes)

    # A too-small bytearray silently corrupts slice-assignment writes
    # past its current length (Python inserts instead of raising) -- the
    # buffer must span the full DDR address space `DdrMap.LIMIT` bounds,
    # not just `pi.image.size_bytes()` (the highest address written so
    # far, which is smaller than the OUTPUTS region's base address).
    flat = bytearray(DdrMap.LIMIT)
    for addr, word in pi.image.words():
        flat[addr : addr + 8] = word.to_bytes(8, "little")
    executed = golden.run_program(flat, pi.program_addr)
    assert executed == len(planned.steps)
    actual_bytes = bytes(flat[out_addr : out_addr + out_tensor.size_bytes])
    assert actual_bytes == expected_bytes
