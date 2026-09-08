"""Python foundation layer for the `cnn_accel_top` rev-2 (ISA v2.0)
command-driven top level.

See `modules/cnn_accel/doc/cnn_accel_top_v2_arch.md` (sections 3, 5, 6, 7)
for the authoritative spec this package implements:

* `isa`: ISA v2.0 descriptor constants, the `DescV2` dataclass and its
  64-byte little-endian codec (`encode_desc`/`decode_desc`).
* `memimage`: sparse 64-bit-word external memory image plus the section-7
  CSV memory-image reader/writer.
* `ddrmap`: section-6 DDR region layout and a bump allocator per region.
"""

from __future__ import annotations
