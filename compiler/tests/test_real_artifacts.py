"""Parse-survival tests against *genuine* exported TOSA, not the
hand-written fixtures in `tests/fixtures/`.

The hand-written fixtures are what `iree-opt` prints for IR this compiler
already handles, so they can only ever confirm that the parser still
accepts what it accepted yesterday. The artifacts here come out of a real
torch-mlir export of YOLOv8n and contain everything a hand-written fixture
never does: TOSA-1.0 `!tosa.shape<N>` operands, `dense_resource` constants
plus the trailing `{-# dialect_resources ... #-}` block, `dense<"0x...">`
hex blobs, `tensor<4xindex>` shape constants, fp32 dtypes, and 477 ops of
detection head that this accelerator is not meant to run.

The bar these tests set is deliberately split in two:

* **Parsing must succeed.** `mlir_generic` is a syntax layer; a real
  producer's output is not "unsupported", it is the input.
* **Importing must fail with a typed, actionable diagnostic** naming the
  offending op and its `line:col` -- never a raw traceback out of
  `struct`, `int()` or a tuple unpack.

The artifacts live outside the repository (they are large, and generated
by a separate toolchain), so every test here skips when they are absent.
Point `CNNC_TOSA_ARTIFACTS` at the directory to run them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cnnc.errors import TosaImportError
from cnnc.frontend.mlir_generic import (
    DenseElementsAttr,
    DenseResourceAttr,
    ShapeType,
    parse_file,
)
from cnnc.frontend.tosa_import import import_tosa

_DEFAULT_ARTIFACT_DIR = Path.home() / "yolo-gap" / "artifacts"

#: Artifact file name -> (op count, whether it carries a dialect_resources
#: block). The op counts are asserted so a silently-truncated parse (an
#: early `func.return`, a dropped region) fails loudly rather than passing
#: as "well, it parsed".
_ARTIFACTS = {
    "yolov8n_fp32_full_patched_tosa.generic.mlir": (477, True),
    "yolov8n_fp32_backbone_tosa.generic.mlir": (197, True),
    "yolov8n_fp32_firstconv_tosa.generic.mlir": (10, True),
    "yolov8n_fp32_firstconv_tosa.generic_inline.mlir": (10, True),
    "yolov8n_int8_refrep_firstconv_tosa.generic.mlir": (84, True),
}


def _artifact_dir() -> Path:
    return Path(os.environ.get("CNNC_TOSA_ARTIFACTS", _DEFAULT_ARTIFACT_DIR))


def _artifact(name: str) -> Path:
    path = _artifact_dir() / name
    if not path.is_file():
        pytest.skip(f"real TOSA artifact {name} not available (set CNNC_TOSA_ARTIFACTS)")
    return path


@pytest.mark.parametrize(("name", "op_count", "has_resources"), [
    (name, ops, res) for name, (ops, res) in _ARTIFACTS.items()
])
def test_real_artifact_parses(name: str, op_count: int, has_resources: bool):
    module = parse_file(_artifact(name))
    assert len(module.funcs) == 1
    assert len(module.funcs[0].ops) == op_count
    assert bool(module.resources) == has_resources


def test_full_yolov8n_carries_the_constructs_that_used_to_crash():
    """The point of the full artifact: it is the one file that contains
    every construct at once. Asserted explicitly so a future parser change
    that quietly stops producing one of them is caught here rather than by
    an op count that happens to still match."""
    module = parse_file(_artifact("yolov8n_fp32_full_patched_tosa.generic.mlir"))
    ops = module.funcs[0].ops

    shape_typed = [op for op in ops if op.results and isinstance(op.results[0][1], ShapeType)]
    assert len(shape_typed) == 28, "!tosa.shape<N> results (tosa.const_shape)"

    attrs = [op.attrs.get("values") for op in ops if op.name == "tosa.const"]
    assert any(isinstance(a, DenseResourceAttr) for a in attrs), "dense_resource constants"
    assert any(isinstance(a, DenseElementsAttr) and a.blob is not None for a in attrs), 'dense<"0x..."> blobs'
    assert any(
        isinstance(a, DenseElementsAttr) and isinstance(a.splat, float) for a in attrs
    ), "float splat constants"
    assert len(module.resources) == 65, "dialect_resources blobs"


def test_dense_resource_blobs_decode_against_their_declared_type():
    """Every resource-backed constant's blob must be exactly the size its
    tensor type implies -- the check that would catch a wrong assumption
    about MLIR's blob header or element width."""
    module = parse_file(_artifact("yolov8n_fp32_full_patched_tosa.generic.mlir"))
    decoded = 0
    for op in module.funcs[0].ops:
        attr = op.attrs.get("values")
        if isinstance(attr, DenseResourceAttr):
            assert len(attr.decode(module.resources)) == attr.tensor_type.numel
            decoded += 1
    assert decoded > 0


@pytest.mark.parametrize("name", sorted(_ARTIFACTS))
def test_real_artifact_import_fails_cleanly(name: str):
    """fp32 is genuinely unsupported -- this compiler is integer-only --
    but that has to arrive as a `TosaImportError`, not a traceback."""
    module = parse_file(_artifact(name))
    with pytest.raises(TosaImportError) as exc:
        import_tosa(module)
    message = str(exc.value)
    assert "[import]" in message
    assert "f32" in message


def test_host_side_head_ops_are_rejected_as_host_side():
    """A detection-head op must not be reported as "unsupported op" -- the
    graph needs splitting, and the diagnostic has to say so."""
    from cnnc.frontend.mlir_generic import parse_module

    text = (
        '"builtin.module"() ({\n'
        '  "func.func"() <{function_type = (tensor<1x4xi8>) -> tensor<1x4xi8>, sym_name = "main"}> ({\n'
        "  ^bb0(%arg0: tensor<1x4xi8>):\n"
        '    %0 = "tosa.sigmoid"(%arg0) : (tensor<1x4xi8>) -> tensor<1x4xi8>\n'
        '    "func.return"(%0) : (tensor<1x4xi8>) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )
    with pytest.raises(TosaImportError) as exc:
        import_tosa(parse_module(text))
    message = str(exc.value)
    assert "host-side detection head" in message
    assert "tosa.sigmoid at 4:10" in message


# --------------------------------------------------------------------------
# `dense_resource` as real data, not just as something that parses.
#
# The artifacts above all stop at the fp32 gate, so on their own they only
# prove the parser survives. This builds the same situation the parser must
# eventually serve -- an int8 model whose weights live in the trailing
# `dialect_resources` block, which is how torch-mlir exports anything
# larger than its inlining threshold -- and compiles it all the way to a
# program, byte-identical to the inline-constant version.
# --------------------------------------------------------------------------


def _to_dense_resource(text: str, const_line_marker: str, key: str, blob: bytes) -> str:
    """Rewrite one `tosa.const`'s inline `dense<...>` into a
    `dense_resource<key>` and append the resource block carrying `blob`."""
    import re

    line = next(ln for ln in text.splitlines() if const_line_marker in ln)
    rewritten = re.sub(r"dense<[^>]*>", f"dense_resource<{key}>", line, count=1)
    header = (4).to_bytes(4, "little")  # MLIR blob alignment header
    return text.replace(line, rewritten) + (
        "\n{-#\n  dialect_resources: {\n    builtin: {\n"
        f'      {key}: "0x{(header + blob).hex().upper()}"\n'
        "    }\n  }\n#-}\n"
    )


def test_resource_backed_weights_compile_identically_to_inline_ones(target, tmp_path):
    from cnnc.driver import compile_tosa

    inline_path = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"
    inline_text = inline_path.read_text()

    # The fixture's weight constant is a splat of 1 over 8x3x3x4 int8.
    weights = bytes([1]) * (8 * 3 * 3 * 4)
    resource_text = _to_dense_resource(inline_text, "tensor<8x3x3x4xi8>", "conv_weights", weights)
    resource_path = tmp_path / "resource.mlir"
    resource_path.write_text(resource_text)

    inline = compile_tosa(inline_path, target, out_dir=tmp_path / "inline")
    resourced = compile_tosa(resource_path, target, out_dir=tmp_path / "resourced")

    assert resourced.program.program_bytes == inline.program.program_bytes
    assert resourced.program.constants_bytes == inline.program.constants_bytes


def test_undecodable_resource_is_a_typed_diagnostic_not_a_traceback(target, tmp_path):
    """A blob whose length disagrees with its tensor type must name the op
    and the key, not surface `struct.error`."""
    inline_text = (Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir").read_text()
    truncated = _to_dense_resource(inline_text, "tensor<8x3x3x4xi8>", "conv_weights", b"\x01\x02\x03")

    from cnnc.frontend.mlir_generic import parse_module

    with pytest.raises(TosaImportError) as exc:
        import_tosa(parse_module(truncated))
    message = str(exc.value)
    assert "conv_weights" in message
    assert "3 bytes" in message
