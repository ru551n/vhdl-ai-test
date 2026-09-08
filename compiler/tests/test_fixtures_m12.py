"""M12 fixtures (doc/tosa_compiler_plan.md ~line 723, "compiler: per-channel
rescale"): `per_channel.mlir` (16 output channels, `per_channel = true`,
distinct multipliers, shifts 10/12/14 on six channels so `legalize_rescale`
rewrites those elements to `shift_min = 15`) and `per_channel_oc8.mlir`
(the 8-channel sibling the RTL conv_core testbench can stream, see
`modules/cnn_accel/module_cnn_accel.py::_COMPILER_VECTORS_FIXTURES`).

Acceptance (plan M12):

- e2e byte-exact x3 seeds: `run_program` (golden model reading the scale
  table out of DDR at `scale_addr`) == `gir.interp` == IREE;
- the decoded descriptor has `PER_CHANNEL_EN` and a `scale_addr` inside the
  constants region (a const SCALE_TABLE buffer, byte-identical to
  `cnn_accel_model.pack_scale_table_for_hw`);
- an ISA v1.1 target (no `scale_addr`/`PER_CHANNEL_EN`) -> `CapabilityError`;
- planner: the table is `align`-aligned and overlaps no other buffer.

The golden dumps under `tests/golden/per_channel*.{gir,fused.gir,hir,
program}.txt` pin the legalized per-element multipliers/shifts, the
`%<y>.scale` buffer and the W14 `scale_addr` field.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnnc.backend.cnn_accel_v1 import decode_program, print_program, run_program
from cnnc.backend.cnn_accel_v1.vectors import write_conv_core_vectors
from cnnc.driver import CompileResult, compile_tosa
from cnnc.errors import CapabilityError
from cnnc.gir import interp
from cnnc.gir.printer import print_gir
from cnnc.hir.printer import print_hir
from cnnc.lower.layout import pack_scale_table
from cnnc.testing.iree_oracle import iree_available, run_iree
from conftest import v11_target

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = Path(__file__).parent / "golden"

# name -> (input shape, out_channels)
FIXTURES = {
    "per_channel": ((1, 8, 8, 8), 16),
    "per_channel_oc8": ((1, 8, 8, 8), 8),
}
OUT_ID = "10"


def _compile(name: str, target, tmp_path) -> CompileResult:
    return compile_tosa(FIXTURES_DIR / f"{name}.mlir", target, out_dir=tmp_path / "out")


def _seed_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=shape).astype(np.int8)


def _flag_names(flags: int, target) -> set[str]:
    return {name for name, bit in target.isa.flags.items() if (flags >> bit) & 1}


def _scale_buffer(result: CompileResult):
    tables = [b for b in result.hir_planned.buffers.values() if b.layout == "SCALE_TABLE"]
    assert len(tables) == 1
    return tables[0]


# ---------------------------------------------------------------------------
# Golden dumps.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_matches_golden_dumps(name, target, tmp_path):
    result = _compile(name, target, tmp_path)

    assert print_gir(result.imported_graph) == (GOLDEN_DIR / f"{name}.gir.txt").read_text()
    assert print_gir(result.fused_graph) == (GOLDEN_DIR / f"{name}.fused.gir.txt").read_text()
    assert print_hir(result.hir_mapped) == (GOLDEN_DIR / f"{name}.hir.txt").read_text()

    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    descriptors = decode_program(result.program.program_bytes, target, program_addr=program_addr)
    text = print_program(descriptors, target, program_addr=program_addr)
    assert text == (GOLDEN_DIR / f"{name}.program.txt").read_text()
    assert f"scale_addr=0x{descriptors[0].scale_addr:08x}" in text and "PER_CHANNEL_EN" in text


# ---------------------------------------------------------------------------
# Per-element legalization actually happened, and only where needed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_per_element_legalization(name, target, tmp_path):
    result = _compile(name, target, tmp_path)
    caps = target.units[0].epilogue.rescale
    _shape, out_c = FIXTURES[name]

    imported = next(op for op in result.imported_graph.ops if op.kind == "rescale").attrs
    fused = next(op for op in result.fused_graph.ops if op.kind == "fused_conv").attrs.rescale
    assert imported.per_channel and fused.per_channel
    assert len(imported.multiplier) == len(fused.multiplier) == out_c
    assert len(set(fused.multiplier)) == out_c, "multipliers must stay distinct"

    below = [i for i, s in enumerate(imported.shift) if s < caps.shift_min]
    assert below, "fixture must contain shifts below shift_min to exercise legalization"
    for i in range(out_c):
        if i in below:
            k = caps.shift_min - imported.shift[i]
            assert fused.shift[i] == caps.shift_min
            assert fused.multiplier[i] == imported.multiplier[i] << k
        else:
            assert (fused.multiplier[i], fused.shift[i]) == (imported.multiplier[i], imported.shift[i])
    assert all(caps.shift_min <= s <= caps.shift_max for s in fused.shift)
    assert all(0 <= m < 2 ** (caps.multiplier_bits - 1) for m in fused.multiplier)


# ---------------------------------------------------------------------------
# Descriptor: PER_CHANNEL_EN + scale_addr -> const SCALE_TABLE buffer whose
# bytes are the model's own image of the legalized (mult, shift-15) pairs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_descriptor_has_per_channel_flag_and_table_in_constants_region(name, target, tmp_path):
    result = _compile(name, target, tmp_path)
    _shape, out_c = FIXTURES[name]
    unit = target.units[0]
    caps = unit.epilogue.rescale
    pe_rows = unit.internal_tiling.cout

    desc = result.program.descriptors[0]
    assert _flag_names(desc.flags, target) == {"PER_CHANNEL_EN", "CLAMP_EN", "BIAS_EN", "REQUANT_EN", "PAD_EN"}
    assert (desc.requant_scale, desc.requant_shift) == (0, 0)

    table = _scale_buffer(result)
    assert table.role == "const"
    assert desc.scale_addr == table.addr
    assert table.size_bytes == -(-out_c // pe_rows) * pe_rows * 8

    # Inside the constants region: [min const addr, max const end).
    consts = [b for b in result.hir_planned.buffers.values() if b.role == "const"]
    lo = min(b.addr for b in consts)
    hi = max(b.end for b in consts)
    assert lo <= desc.scale_addr and desc.scale_addr + table.size_bytes <= hi
    # ... and it is in the emitted constants blob at its manifest offset.
    entry = next(e for e in result.program.manifest["buffers"] if e["id"] == table.id)
    off = entry["constants_offset"]
    assert result.program.constants_bytes[off : off + table.size_bytes] == table.data
    assert entry["layout"] == "SCALE_TABLE" and entry["addr"] == desc.scale_addr

    # Bytes == pack of the *legalized* pairs with the implicit shift removed.
    fused = next(op for op in result.fused_graph.ops if op.kind == "fused_conv").attrs.rescale
    expected = pack_scale_table(
        tuple(fused.multiplier), tuple(s - caps.implicit_shift for s in fused.shift), pe_rows
    )
    assert table.data == expected

    # The decoded program (bytes -> Descriptor) round-trips the field.
    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    decoded = decode_program(result.program.program_bytes, target, program_addr=program_addr)
    assert decoded[0].scale_addr == table.addr
    assert "PER_CHANNEL_EN" in _flag_names(decoded[0].flags, target)


# ---------------------------------------------------------------------------
# Planner: table aligned, non-overlapping with every other buffer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_scale_table_planned_aligned_and_non_overlapping(name, target, tmp_path):
    result = _compile(name, target, tmp_path)
    space = next(iter(target.memory.spaces.values()))
    table = _scale_buffer(result)

    assert table.addr is not None and table.addr % space.align == 0
    assert table.align == space.align
    for other in result.hir_planned.buffers.values():
        if other.id == table.id:
            continue
        assert other.end <= table.addr or table.end <= other.addr, (
            f"{table.id} [{table.addr:#x},{table.end:#x}) overlaps {other.id} [{other.addr:#x},{other.end:#x})"
        )
    assert table.end <= result.hir_planned.memory_size


# ---------------------------------------------------------------------------
# End-to-end: run_program (model reads the table from DDR) == interp, x3
# seeds, and the per-channel scaling is visible (channels differ in range).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e2e_run_program_matches_interp(name, seed, target, tmp_path):
    result = _compile(name, target, tmp_path)
    shape, out_c = FIXTURES[name]
    x = _seed_input(shape, seed)

    interp_out = interp.run(result.fused_graph, {"arg0": x})[OUT_ID]
    run_out = run_program(result.program, {"arg0": x})[OUT_ID]

    assert run_out.dtype == np.int8 and run_out.shape == interp_out.shape == (1, 8, 8, out_c)
    np.testing.assert_array_equal(run_out, interp_out)

    # Also equals the un-legalized imported graph (legalization is exact).
    np.testing.assert_array_equal(interp.run(result.imported_graph, {"arg0": x})[OUT_ID], interp_out)

    assert len(np.unique(interp_out)) >= 16
    saturated = np.count_nonzero((interp_out == -128) | (interp_out == 127))
    assert saturated < interp_out.size // 2
    # Every channel carries real signal (not a constant lane).
    assert all(len(np.unique(interp_out[..., c])) >= 8 for c in range(out_c))


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_per_channel_table_matters(name, target, tmp_path):
    # Sanity: broadcasting channel 0's pair to every lane (what a wrongly
    # ignored table would do) gives a different output.
    import dataclasses

    result = _compile(name, target, tmp_path)
    shape, _ = FIXTURES[name]
    x = _seed_input(shape, 0)
    fused = next(op for op in result.fused_graph.ops if op.kind == "fused_conv")
    r = fused.attrs.rescale
    broadcast = dataclasses.replace(
        r, per_channel=False, multiplier=(r.multiplier[0],), shift=(r.shift[0],)
    )
    graph_b = result.fused_graph.replace(
        ops=tuple(
            dataclasses.replace(op, attrs=dataclasses.replace(op.attrs, rescale=broadcast)) if op is fused else op
            for op in result.fused_graph.ops
        )
    )
    a = interp.run(result.fused_graph, {"arg0": x})[OUT_ID]
    b = interp.run(graph_b, {"arg0": x})[OUT_ID]
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# External oracle: IREE agrees with interp and run_program on per-channel.
# ---------------------------------------------------------------------------


@pytest.mark.iree
@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e2e_run_program_matches_interp_and_iree(name, seed, target, tmp_path):
    if not iree_available():
        pytest.skip("iree-compile/iree-run-module not found next to the interpreter")

    result = _compile(name, target, tmp_path)
    shape, _ = FIXTURES[name]
    x = _seed_input(shape, seed)

    interp_out = interp.run(result.fused_graph, {"arg0": x})[OUT_ID]
    run_out = run_program(result.program, {"arg0": x})[OUT_ID]
    iree_out = run_iree(FIXTURES_DIR / f"{name}.mlir", [x], tmp_path / f"iree_{seed}")[0]

    np.testing.assert_array_equal(run_out, interp_out)
    np.testing.assert_array_equal(iree_out, interp_out)


# ---------------------------------------------------------------------------
# Rejection: ISA v1.1 target (pre-H2) has no scale table.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_per_channel_rejected_on_isa_v11_target(name, target, tmp_path):
    with pytest.raises(CapabilityError) as exc_info:
        _compile(name, v11_target(target), tmp_path)
    # `per_channel: false` keeps FusePass from fusing, so the standalone
    # conv2d is what to_hir rejects -- loudly either way.
    assert exc_info.value.stage == "to_hir"


# ---------------------------------------------------------------------------
# RTL vectors hook: the 8-channel case writes scale_table_packed.txt with
# the padded pe_rows entries; the 16-channel case is skipped (one tile only)
# under the `max_out_channels=pe_rows` limit module_cnn_accel.py passes.
# ---------------------------------------------------------------------------


def test_conv_core_vectors_write_scale_table_for_oc8(target, tmp_path):
    result = _compile("per_channel_oc8", target, tmp_path)
    unit = target.units[0]
    pe_rows = unit.internal_tiling.cout
    caps = unit.epilogue.rescale
    inputs = {"arg0": _seed_input((1, 8, 8, 8), 0)}
    out_dir = tmp_path / "vectors"
    vectors = write_conv_core_vectors(result.program, inputs, out_dir, target=target, graph=result.fused_graph)
    assert vectors.written == ["op0"] and vectors.skipped == []

    desc_fields = {}
    for line in (out_dir / "op0" / "desc.txt").read_text().splitlines():
        key, _, value = line.partition(" ")
        desc_fields[key] = int(value)
    assert (desc_fields["flags"] >> target.isa.flags["PER_CHANNEL_EN"]) & 1 == 1

    flat = [int(v) for v in (out_dir / "op0" / "scale_table_packed.txt").read_text().split()]
    assert len(flat) == 2 * pe_rows
    fused = next(op for op in result.fused_graph.ops if op.kind == "fused_conv").attrs.rescale
    expected = [v for m, s in zip(fused.multiplier, fused.shift) for v in (m, s - caps.implicit_shift)]
    assert flat == expected


def test_conv_core_vectors_skip_16_channel_case(target, tmp_path):
    result = _compile("per_channel", target, tmp_path)
    pe_rows = target.units[0].internal_tiling.cout
    inputs = {"arg0": _seed_input((1, 8, 8, 8), 0)}
    vectors = write_conv_core_vectors(
        result.program, inputs, tmp_path / "vectors", target=target, graph=result.fused_graph,
        max_out_channels=pe_rows,
    )
    assert vectors.written == [] and [name for name, _ in vectors.skipped] == ["op0"]
