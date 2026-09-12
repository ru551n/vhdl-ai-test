"""The guard that spatial tiling stays **opt-in**.

Every extension tiling needed -- unit-granular bank confinement,
DDR-pinned tensors, `RowCopyOp`, the tiler itself -- was added to modules
the whole existing catalogue already runs through. The acceptance
criterion for all of them is the same and is not negotiable: a graph that
asks for none of it must plan, place and lower *byte for byte* as it did
before. This file is the permanent statement of that.

It works by digest rather than by prose. For every case in every
catalogue it hashes the things a placement bug would move -- the ordered
local placements, the DDR placements, every tensor's DDR address, every
resolved step, the predicted traffic and the encoded bytes of every
emitted descriptor -- and compares against a constant recorded when the
untiled behaviour was last ratified.

**If this test fails**, the change under test moved something in the
untiled path. That may well be intended (a deliberate planner
improvement), but it is never a detail: re-run
`print_catalogue_digest()` below, diff the two dumps to see exactly which
case and which buffer moved, convince yourself it is what you meant, and
only then update `_RATIFIED_DIGEST` in the same commit that explains it.
Silently refreshing the constant defeats the entire point.
"""

from __future__ import annotations

import hashlib
import json

from accel_v2 import (
    cases,
    cases_concat_split,
    cases_conv_pad,
    cases_error,
    cases_pool_pad,
    cases_yolo,
    isa,
)
from accel_v2.planner import ComputeStep, MoveStep, RowCopyStep

_CATALOGUES = (cases, cases_concat_split, cases_conv_pad, cases_error, cases_pool_pad, cases_yolo)

#: sha256 of `catalogue_dump()`, re-ratified while landing the
#: output-channel-tile read accounting (`planner.ifmap_passes`) on top of
#: 7256b14.
#:
#: How each ratification was established, since a digest taken *after* a
#: change proves nothing on its own: the dump is taken on the previous
#: commit and on the change under test and diffed field by field across
#: all 64 cases.
#:
#: * 51fae78... (spatial tiling core, on top of d695c21): identical to
#:   d695c21 except for the two new `DdrTraffic.pinned_*_bytes` counters,
#:   which are `0` in every untiled case because nothing is pinned.
#: * this digest (the simulator half of spatial tiling, on top of
#:   7256b14): identical to 7256b14 in every `local_placements`,
#:   `ddr_placements`, `pinned_placements`, `tensor_ddr_addr`, `steps`,
#:   `descs`, `program_addr` and geometry entry of all 64 cases, and
#:   different in exactly two counters:
#:
#:   - `traffic.read_bytes` (9 cases) -- the convolutions whose ifmap the
#:     hardware streams once per output-channel tile
#:     (`planner.ifmap_passes`) and the `ACT`s whose 256-byte LUT is
#:     refetched per command (`isa.ACT_LUT_BYTES`). Both were real DDR
#:     reads the prediction did not charge, which is why `read_bytes`
#:     could only ever be a lower bound; with them charged it is exact
#:     for every case in the catalogue and `TrafficPolicy.
#:     read_bytes_exact` defaults to `True`.
#:   - `traffic.local_read_bytes` (9 cases) -- the same ifmap
#:     re-streaming, for the convolutions whose input is resident.
#:
#:   Nothing else moved, across resident weight images, per-plane
#:   spill lowering and the sharing of one DDR weight image between
#:   convolutions with the same weights: none of those three is reachable
#:   from an untiled graph, which is what the structural test above
#:   independently guarantees.
#: * this digest (ISA v2.2 `DEPTH_TO_SPACE`, on top of 22e3aea):
#:   re-ratified for ADDED CASES ONLY -- `cases.depth_to_space`,
#:   `cases.depth_to_space_two_tiles`, `cases_error.err_dts_bad_factor`
#:   and `cases_error.err_dts_bad_channels`, taking the catalogue from 64
#:   to 68. Established the way this file demands rather than by
#:   refreshing the constant: the dump was taken on the change under
#:   test, the four new keys removed, and the remaining 64 re-hashed --
#:   giving back 541f4304... exactly, i.e. every pre-existing case's
#:   `local_placements`, `ddr_placements`, `pinned_placements`,
#:   `tensor_ddr_addr`, `traffic`, `steps`, `descs` and `program_addr` is
#:   byte-identical. Nothing in the untiled path moved; the new opcode
#:   only added rows.
#: * this digest (ISA v2.3 streaming-inference interface -- INPUT_ADDR/
#:   OUTPUT_ADDR relocation, `program.py`'s `_tag_relocatable_operands`
#:   -- on top of the previous ratification): dump taken via a clean
#:   worktree at the prior commit vs. the working tree under test, same
#:   68 cases both sides. `local_placements`, `ddr_placements`,
#:   `pinned_placements`, `tensor_ddr_addr`, `traffic`, `steps` and
#:   `program_addr` identical in every case -- this change allocates no
#:   address and moves no placement, exactly as the feature design
#:   intends. `descs` differs in exactly 127 descriptors across the
#:   catalogue, every one of them at byte offset 3 only (the new
#:   `reloc_input`/`reloc_output` bits) and every value accounted for:
#:   54 descriptors gain `reloc_input` alone (0->1), 59 gain
#:   `reloc_output` alone (0->2), 13 single-op cases (one descriptor
#:   both reads the graph input and writes the graph output) gain both
#:   (0->3), and `cases_error.err_bad_reserved`'s deliberately-corrupted
#:   descriptor moves from 1->7 -- its own `reserved_w0=1` override
#:   re-encodes to bit 2 under the new packing (still inside the
#:   `reserved_w0(7 downto 2)` range `cnn_accel_cmd_proc` actually
#:   checks, per `test_isa.test_reserved_w0_still_triggers_bad_reserved_
#:   at_its_new_bit_position`) combined with that case also being a
#:   single-op shape, so both reloc bits set too. No other byte of any
#:   descriptor, and no other field of any case, changed.
_RATIFIED_DIGEST = "91c382bacfb4a2e5ff0846630e6bc97f3d08d031f7febd60fd59775060980c14"

#: The number of cases the digest covers, asserted separately so that
#: *deleting* a case cannot silently keep the digest meaningful.
_RATIFIED_CASE_COUNT = 68


def _step_record(step) -> list:
    if isinstance(step, MoveStep):
        return [
            "move",
            step.tensor.name,
            step.src_space,
            step.src_addr,
            step.dst_space,
            step.dst_addr,
            step.nbytes,
            step.kind,
        ]
    if isinstance(step, RowCopyStep):
        return ["rowcopy", step.op.name, step.src_space, step.dst_space, step.transfers]
    assert isinstance(step, ComputeStep)
    return [
        "compute",
        step.op.name,
        type(step.op).__name__,
        step.input_spaces,
        step.input_addrs,
        step.output_space,
        step.output_addr,
    ]


def catalogue_dump() -> dict:
    """Everything about the catalogue a placement or lowering change
    would perturb, in a form a human can diff."""
    dump: dict = {}
    for module in _CATALOGUES:
        for case in module.all_cases():
            planned = case.planned
            dump[f"{module.__name__}.{case.name}"] = {
                "local_placements": [list(p) for p in planned.local_placements],
                "ddr_placements": [list(p) for p in planned.ddr_placements],
                "pinned_placements": [list(p) for p in planned.pinned_placements],
                "tensor_ddr_addr": dict(planned.tensor_ddr_addr),
                "traffic": vars(planned.traffic),
                "tensor_mem_bytes": planned.tensor_mem_bytes,
                "bank_bytes": planned.bank_bytes,
                "steps": [_step_record(s) for s in planned.steps],
                "descs": [isa.encode_desc(d).hex() for d in case.program.descs],
                "program_addr": case.program.program_addr,
            }
    return dump


def catalogue_digest() -> str:
    payload = json.dumps(catalogue_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def print_catalogue_digest() -> None:  # pragma: no cover - developer helper
    """`python -c "from accel_v2.tests.test_tiling_guard import *;
    print_catalogue_digest()"` -- prints the digest and writes the full
    dump to `catalogue_dump.json` for diffing against another checkout."""
    with open("catalogue_dump.json", "w") as handle:
        json.dump(catalogue_dump(), handle, indent=1, sort_keys=True)
    print(catalogue_digest())


def test_no_catalogue_case_asks_for_any_tiling_feature() -> None:
    """The structural half of the guard, and the one that keeps failing
    for the right reason: not one existing tensor sets `pin_ddr` or
    `confine_unit_bytes`, and not one existing op is a `RowCopyOp`. So
    every tiling code path is reached only by a graph that asked for it,
    and the digest below is testing the *unmodified* lowering."""
    from accel_v2.model import RowCopyOp

    for module in _CATALOGUES:
        for case in module.all_cases():
            model = case.planned.model
            for tensor in model.tensors:
                assert not tensor.pin_ddr, f"{case.name}: '{tensor.name}' is pinned"
                assert tensor.confine_unit_bytes is None, (
                    f"{case.name}: '{tensor.name}' sets confine_unit_bytes"
                )
                assert tensor.origin is None, f"{case.name}: '{tensor.name}' is a strip"
            for op in model.ops:
                assert not isinstance(op, RowCopyOp), f"{case.name}: '{op.name}' is a row copy"


def test_every_catalogue_case_still_plans_byte_for_byte() -> None:
    """The digest itself. See this module's docstring before touching
    `_RATIFIED_DIGEST`."""
    dump = catalogue_dump()
    assert len(dump) == _RATIFIED_CASE_COUNT, (
        f"the digest was ratified over {_RATIFIED_CASE_COUNT} cases but the catalogue now "
        f"has {len(dump)}. Adding a case is fine -- re-ratify both constants together, in a "
        "commit that says so."
    )
    assert catalogue_digest() == _RATIFIED_DIGEST, (
        "the untiled catalogue no longer plans byte for byte. Run "
        "`print_catalogue_digest()` in this module on both checkouts and diff the two "
        "`catalogue_dump.json` files to find which case and which buffer moved; see this "
        "module's docstring."
    )
