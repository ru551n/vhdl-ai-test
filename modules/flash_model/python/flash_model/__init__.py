"""Python device model for a generic JEDEC QSPI NOR flash.

The VHDL verification component (`sim/flash_model.vhd`) owns pins and
clocks and nothing else: every byte of device behaviour -- the opcode
table, the array, protection, timing, SFDP -- lives here, because a
protocol state machine is far cheaper to write, read and unit-test in
Python than in VHDL, and because the same model can then be exercised by
pytest with no simulator in the loop.

Layering, outermost first:

* `flash_model_bridge` (one level up, NOT part of this package) is the only
  module VHDL ever calls. It unpacks arguments, looks up an instance id and
  delegates. It holds no device logic.
* `device`: the protocol state machine -- `cs_assert` / `xfer` /
  `cs_deassert`, status registers, the page-program latch.
* `commands`: the JEDEC opcode table as *data*. Adding an opcode is a table
  entry, never a new branch.
* `array`: the storage. NOR program/erase semantics (program is AND-only,
  erase sets 0xFF) live here and nowhere else.
* `protection`, `timing`, `mode`, `sfdp`, `profiles`, `directive`: one
  concern each, all data-driven so that a new part is a dict entry.

The FFI wire format is defined by `doc/flash_model_ffi_contract.md`;
`directive.LAYOUT_VERSION` is the run-time guard against the two sides
drifting apart.
"""

from __future__ import annotations

from .directive import LAYOUT_VERSION

__all__ = ["LAYOUT_VERSION"]
