"""Tests for `accel_v2.ddrmap`: section-6 region offsets and the
per-region bump allocator."""

from __future__ import annotations

import pytest

from accel_v2.ddrmap import DdrMap


def test_region_offsets_match_section_6() -> None:
    assert DdrMap.NULL_GUARD == 0x0000_0000
    assert DdrMap.PROGRAM == 0x0000_1000
    assert DdrMap.WEIGHTS == 0x0001_0000
    assert DdrMap.BIAS == 0x0002_0000
    assert DdrMap.SCALE == 0x0003_0000
    assert DdrMap.LUT == 0x0004_0000
    assert DdrMap.INPUTS == 0x0008_0000
    assert DdrMap.SPILL == 0x000C_0000
    assert DdrMap.OUTPUTS == 0x0010_0000
    assert DdrMap.LIMIT == 0x0020_0000


def test_alloc_starts_at_region_base() -> None:
    m = DdrMap()
    assert m.alloc(DdrMap.WEIGHTS, 16) == DdrMap.WEIGHTS
    assert m.alloc(DdrMap.WEIGHTS, 16) == DdrMap.WEIGHTS + 16


def test_alloc_is_independent_per_region() -> None:
    m = DdrMap()
    m.alloc(DdrMap.WEIGHTS, 100)
    assert m.alloc(DdrMap.BIAS, 8) == DdrMap.BIAS


def test_alloc_respects_alignment() -> None:
    m = DdrMap()
    m.alloc(DdrMap.WEIGHTS, 3)  # cursor now at WEIGHTS + 3
    addr = m.alloc(DdrMap.WEIGHTS, 8, align=8)
    assert addr % 8 == 0
    assert addr == DdrMap.WEIGHTS + 8


def test_reset_rewinds_all_regions() -> None:
    m = DdrMap()
    m.alloc(DdrMap.PROGRAM, 64)
    m.alloc(DdrMap.OUTPUTS, 1024)
    m.reset()
    assert m.alloc(DdrMap.PROGRAM, 64) == DdrMap.PROGRAM
    assert m.alloc(DdrMap.OUTPUTS, 1024) == DdrMap.OUTPUTS


def test_alloc_overflow_into_next_region_raises() -> None:
    m = DdrMap()
    room = DdrMap.WEIGHTS_LIMIT if hasattr(DdrMap, "WEIGHTS_LIMIT") else DdrMap.BIAS - DdrMap.WEIGHTS
    with pytest.raises(ValueError, match="WEIGHTS"):
        m.alloc(DdrMap.WEIGHTS, room + 1)


def test_alloc_overflow_past_limit_on_last_region_raises() -> None:
    m = DdrMap()
    room = DdrMap.LIMIT - DdrMap.OUTPUTS
    with pytest.raises(ValueError, match="OUTPUTS"):
        m.alloc(DdrMap.OUTPUTS, room + 1)
    # Exactly filling the region is fine.
    m.reset()
    m.alloc(DdrMap.OUTPUTS, room)


def test_alloc_unknown_region_raises() -> None:
    m = DdrMap()
    with pytest.raises(ValueError):
        m.alloc(DdrMap.NULL_GUARD, 8)
    with pytest.raises(ValueError):
        m.alloc(0x1234, 8)
