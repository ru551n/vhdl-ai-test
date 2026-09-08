"""Custom `hdl-registers` code generator for the parts of `cnn_accel`'s ISA
that `hdl-registers`' native register/constant model cannot express as one
coherent, structurally-checked artifact: the 64-byte instruction word's
byte-offset table (with its documented reserved gaps), the opcode values,
and the flag bit indices. See `cnn_accel_constants.py` for the single
Python source of truth this reads, and `doc/cnn_accel_arch.md`'s
"Instruction Set (v1)" table / `doc/cnn_accel_pkg_req.md` for the spec.

Why this needs a custom `RegisterCodeGenerator` rather than more
`RegisterList.add_constant()` calls in `registers_hook()`:

- Every individual *value* here (an opcode byte, a flag bit index, a byte
  offset) is, in isolation, perfectly representable as a native
  `hdl-registers` constant (`UnsignedVectorConstant` even preserves the
  `std_ulogic_vector(7 downto 0)` type opcodes need). That is not the
  limitation.
- What is not representable is the *table itself* as a structured unit:
  `hdl-registers`' `Constant` classes carry a name, a value, and a
  description -- nothing else. There is no way to attach "this is a 4-byte
  field, treated as signed" to a constant as queryable metadata (as
  opposed to just one already-rendered VHDL line), so a future consumer
  (e.g. a generated decode/slice function) has no structured table to walk.
  There is also no "reserved gap" concept and no facility to assert
  self-consistency (no overlaps, nothing crossing the 64-byte word) as
  part of generation -- that structural guarantee only exists because this
  generator and `cnn_accel_constants.py`'s own self-consistency test (see
  `test_cnn_accel_model.py`) both walk the *same* `ISA_LAYOUT` table.
- Mixing these into the same `RegisterList` as the CSR registers would
  also dump unrelated concepts into one `cnn_accel_regs_pkg.vhd`: CSR
  registers are real AXI4-Lite-addressable state (`doc/cnn_accel_csr_req.md`),
  while the ISA table describes the DMA-fetched instruction *stream*
  format -- never touched through the register file. Keeping them in
  separate generated packages keeps that distinction structural, not just
  a comment.

Hooked in from `module_cnn_accel.py`'s `create_register_synthesis_files()`
override (calls `super()` first for the native artifacts, then this
generator), per that method's own docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from hdl_registers.generator.register_code_generator import RegisterCodeGenerator

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import cnn_accel_constants  # noqa: E402


class CnnAccelIsaPackageGenerator(RegisterCodeGenerator):
    """
    Generates `cnn_accel_isa_pkg.vhd`: opcode constants, flag bit-index
    constants, and the instruction word's per-field byte-offset constants,
    all derived from `cnn_accel_constants.py`.
    """

    __version__ = "1.0.0"

    SHORT_DESCRIPTION = "cnn_accel ISA VHDL package"

    COMMENT_START = "--"

    @property
    def output_file(self) -> Path:
        return self.output_folder / "cnn_accel_isa_pkg.vhd"

    def _create_artifact(self, output_file: Path, **kwargs: Any) -> Path:  # noqa: ANN401
        """
        Write the package only when its body actually changed, so an
        unchanged run does not bump the file's mtime.

        **This is load-bearing, not an optimization.** GHDL records the
        source timestamp of every analysed file in its work library and
        refuses to elaborate against a stale one ("file ... has changed and
        must be reanalysed"). tsfpga's netlist flow *creates* every build
        project (which regenerates register artifacts, then analyses) before
        it *builds* any of them (elaboration), so an unconditional rewrite
        here means a later project's create invalidates the work library an
        earlier project already analysed. That is exactly what made
        `cnn_accel_pe_array` and `cnn_accel_conv_core` -- the last two of the
        six Yosys netlist builds -- fail in CI while all six passed when run
        one at a time.

        `RegisterCodeGenerator.create()` always rewrites, and
        `create_if_needed()` cannot fix it: its staleness check hashes the
        `RegisterList`, which does not contain this file's source of truth
        (`cnn_accel_constants.py`'s `ISA_LAYOUT`), so it would never
        regenerate after the first time and would silently ship a stale ISA
        package when the layout changes. Comparing the generated body is the
        check that is actually correct for this generator.

        The comparison ignores the generated header, which carries a
        timestamp and a Git commit and therefore always differs.
        """
        # Must match `RegisterCodeGenerator._create_artifact()` byte for byte:
        # it writes f"{self.header}\n{code}", i.e. there is a blank line
        # between the header and the code. Comparing against the bare
        # `get_code()` output silently never matches, which makes this whole
        # override a no-op -- do not "simplify" this.
        new_body = f"\n{self.get_code(**kwargs)}"

        if output_file.exists():
            separator_line = self.get_separator_line(indent=0)
            # The header is 'separator + comment block + separator', so the
            # body is everything after the second separator. maxsplit=2 keeps
            # any later separator-looking line as part of the body.
            existing = output_file.read_text().split(separator_line, 2)
            if len(existing) == 3 and existing[2] == new_body:
                print(f"{self.SHORT_DESCRIPTION} unchanged, not rewritten: {output_file.name}")
                return output_file

        return super()._create_artifact(output_file=output_file, **kwargs)

    def get_code(
        self,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> str:
        assert cnn_accel_constants.isa_total_bytes() == cnn_accel_constants.INSTR_WORD_BYTES, (
            "cnn_accel_constants.ISA_LAYOUT does not add up to INSTR_WORD_BYTES -- "
            "fix the layout table before generating."
        )

        vhdl = """\
library ieee;
use ieee.std_logic_1164.all;

-- Generated ISA constants for modules/cnn_accel/: opcodes, flag bit
-- indices, and instruction-word byte offsets. Single Python source of
-- truth is cnn_accel_constants.py; the golden model (cnn_accel_model.py)
-- derives its own OPCODE_*/FLAG_*/OFF_* names from the same table, so this
-- package and the golden model's encoder agree byte-for-byte by
-- construction. See doc/cnn_accel_arch.md "Instruction Set (v1)" and
-- doc/cnn_accel_pkg_req.md for the specification.
package cnn_accel_isa_pkg is

"""
        vhdl += self._opcodes()
        vhdl += "\n"
        vhdl += self._flags()
        vhdl += "\n"
        vhdl += self._spaces()
        vhdl += "\n"
        vhdl += self._instruction_layout()
        vhdl += "\nend package cnn_accel_isa_pkg;\n"

        return vhdl

    def _opcodes(self) -> str:
        result = self.comment_block(
            text=["Opcodes (instruction word W0 bits [7:0])."],
            indent=2,
        )
        for name, value in cnn_accel_constants.OPCODES.items():
            result += (
                f'  constant OPCODE_{name} : std_ulogic_vector(7 downto 0) := x"{value:02x}";\n'
            )
        return result

    def _flags(self) -> str:
        result = self.comment_block(
            text=["Flag bits (instruction word W0 bits [15:8]), indices into the flags byte."],
            indent=2,
        )
        for name, bit in cnn_accel_constants.FLAGS.items():
            result += f"  constant FLAG_{name} : natural := {bit};\n"
        return result

    def _spaces(self) -> str:
        result = self.comment_block(
            text=[
                "Storage-space tag values (ISA v2.0), 2 bits per operand.",
                "DDR is 0 so a v1.2 word's reserved-zero spaces byte means "
                "'every operand in DDR', i.e. exact v1.2 semantics.",
            ],
            indent=2,
        )
        for name, value in cnn_accel_constants.SPACES.items():
            result += (
                f"  constant SPACE_{name} : std_ulogic_vector("
                f"{cnn_accel_constants.SPACE_TAG_BITS - 1} downto 0) := "
                f'"{value:0{cnn_accel_constants.SPACE_TAG_BITS}b}";\n'
            )

        result += "\n"
        result += self.comment_block(
            text=[
                "Bit index of each operand's space tag inside the 'spaces' byte "
                "(instruction word W0 bits [23:16]), same idiom as FLAG_* above.",
            ],
            indent=2,
        )
        for name, bit in cnn_accel_constants.SPACE_FIELDS.items():
            result += f"  constant SPACE_SHIFT_{name} : natural := {bit};\n"

        return result

    def _instruction_layout(self) -> str:
        result = self.comment_block(
            text=[
                "Instruction word layout: size and per-field byte offsets.",
                "cnn_accel_model.py's encoder is generated from the same table "
                "(cnn_accel_constants.ISA_LAYOUT) and so agrees byte-for-byte by "
                "construction.",
            ],
            indent=2,
        )
        result += (
            f"  constant c_instr_word_bytes : positive := "
            f"{cnn_accel_constants.INSTR_WORD_BYTES};\n\n"
        )

        offset = 0
        for field in cnn_accel_constants.ISA_LAYOUT:
            if field.name == cnn_accel_constants.RESERVED:
                first_word = offset // 4
                last_word = (offset + field.width_bytes - 1) // 4
                word_label = (
                    f"W{first_word}" if first_word == last_word else f"W{first_word}-W{last_word}"
                )
                byte_label = (
                    f"{offset}"
                    if field.width_bytes == 1
                    else f"{offset}-{offset + field.width_bytes - 1}"
                )
                byte_noun = "byte" if field.width_bytes == 1 else "bytes"
                result += self.comment(
                    f"{word_label} {byte_noun} {byte_label}: reserved, must be 0.", indent=2
                )
            else:
                result += f"  constant c_off_{field.name} : natural := {offset};\n"

            offset += field.width_bytes

        return result
