"""Section-6 external DDR memory map and a per-region bump allocator.

Authoritative source: `modules/cnn_accel/doc/cnn_accel_top_v2_arch.md`
section 6. Regions are **soft** -- defined by the Python layout below,
not by hardware decode -- so this module is the single source of truth
the program generator, weight/bias/scale packers and testbench-CSV
export all key off of.
"""

from __future__ import annotations


class DdrMap:
    """Bump allocator over the section-6 DDR region layout.

    Each region has its own independent cursor, starting at the
    region's base offset and monotonically increasing; `alloc` never
    reuses or reorders a previous allocation within a run (call
    `reset()` to start over). Regions are laid out back-to-back in
    ascending address order, so a region's usable end is the base
    offset of the next region (or `LIMIT` for the last one, `OUTPUTS`).
    """

    #: never accessed by hardware; guards against null-address bugs.
    NULL_GUARD = 0x0000_0000
    #: chained 64-byte descriptors, `HALT` last.
    PROGRAM = 0x0000_1000
    #: `pack_weights_for_hw` images, per layer.
    WEIGHTS = 0x0001_0000
    #: `pack_bias_for_hw` images.
    BIAS = 0x0002_0000
    #: `pack_scale_table_for_hw` images.
    SCALE = 0x0003_0000
    #: 256-byte int8->int8 activation tables.
    LUT = 0x0004_0000
    #: initial activation tensors (PLANES layout).
    INPUTS = 0x0008_0000
    #: destinations of `STORE`-as-spill.
    SPILL = 0x000C_0000
    #: final results; the region exported to CSV.
    OUTPUTS = 0x0010_0000

    #: default simulation DDR limit (2 MiB), bounds validation.
    LIMIT = 0x0020_0000

    #: allocatable regions, ascending by address (`NULL_GUARD` is never
    #: allocated into -- it exists only to catch null-address bugs).
    _REGION_ORDER: tuple[int, ...] = (
        PROGRAM,
        WEIGHTS,
        BIAS,
        SCALE,
        LUT,
        INPUTS,
        SPILL,
        OUTPUTS,
    )

    _REGION_NAMES: dict[int, str] = {
        NULL_GUARD: "NULL_GUARD",
        PROGRAM: "PROGRAM",
        WEIGHTS: "WEIGHTS",
        BIAS: "BIAS",
        SCALE: "SCALE",
        LUT: "LUT",
        INPUTS: "INPUTS",
        SPILL: "SPILL",
        OUTPUTS: "OUTPUTS",
    }

    def __init__(self) -> None:
        self._next: dict[int, int] = {}
        self.reset()

    def reset(self) -> None:
        """Reset every region's bump cursor back to its base offset."""
        self._next = {base: base for base in self._REGION_ORDER}

    def _region_limit(self, region: int) -> int:
        idx = self._REGION_ORDER.index(region)
        if idx + 1 < len(self._REGION_ORDER):
            return self._REGION_ORDER[idx + 1]
        return self.LIMIT

    def alloc(self, region: int, nbytes: int, align: int = 8) -> int:
        """Allocate `nbytes` (rounded up to `align`-aligned start) from
        `region` (one of the region-base class attributes, e.g.
        `DdrMap.WEIGHTS`) and return the allocated start address.

        Raises `ValueError` if `region` is not an allocatable region, or
        if the allocation would overflow into the next region (or past
        `LIMIT` for `OUTPUTS`, the last region)."""
        if region not in self._next:
            name = self._REGION_NAMES.get(region, f"0x{region:08x}")
            raise ValueError(f"{name} is not an allocatable DDR region")
        if nbytes < 0:
            raise ValueError(f"negative allocation size {nbytes}")
        if align <= 0:
            raise ValueError(f"non-positive alignment {align}")

        cursor = self._next[region]
        rem = cursor % align
        if rem:
            cursor += align - rem
        end = cursor + nbytes

        limit = self._region_limit(region)
        if end > limit:
            name = self._REGION_NAMES[region]
            raise ValueError(
                f"{name} region overflow: allocating {nbytes} bytes at "
                f"0x{cursor:08x} would end at 0x{end:08x}, past region "
                f"limit 0x{limit:08x}"
            )

        self._next[region] = end
        return cursor


__all__ = ["DdrMap"]
