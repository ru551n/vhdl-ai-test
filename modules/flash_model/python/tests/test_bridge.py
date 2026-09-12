"""The FFI surface itself: return types, argument shapes, instance
handling.

These tests exist because the bridge's callers are VHDL, which cannot say
"that was a list, I wanted an int32 array" -- it just misbehaves. Every
contract-typed return value is asserted for dtype here."""

from __future__ import annotations

import flash_model_bridge as bridge
import numpy as np
import pytest
from flash_model.directive import Action, unpack


@pytest.fixture
def fid() -> int:
    """A fresh instance with busy times off, the common testbench setup."""
    instance = bridge.flash_create()
    bridge.set_timing_enable(id=instance, enable=0)
    return instance


def test_layout_version_matches_the_contract() -> None:
    assert bridge.layout_version() == 1


def test_instances_are_independent_and_ids_are_never_zero() -> None:
    first = bridge.flash_create()
    second = bridge.flash_create(profile="w25q32jv")
    assert first != second and 0 not in (first, second)
    bridge.preload([0xAA], id=first, addr=0)
    assert list(bridge.read_back(id=second, addr=0, num_bytes=1)) == [0xFF]
    assert bridge.get_stat(id=second, name="sr1") == 0


def test_an_unknown_instance_id_raises_helpfully() -> None:
    with pytest.raises(KeyError, match="flash_create"):
        bridge.cs_assert(id=9999, now_s=0.0)


def test_geometry_overrides() -> None:
    instance = bridge.flash_create(
        profile="generic_16mib", size_bytes=1 << 20, page_bytes=128, addr_bytes=4
    )
    assert bridge.get_stat(id=instance, name="addr_bytes") == 4
    bridge.preload_fill(id=instance, addr=0, num_bytes=1 << 20, value=0x00)
    with pytest.raises(ValueError):
        bridge.preload_fill(id=instance, addr=0, num_bytes=(1 << 20) + 1, value=0)


def test_bad_override_name_and_geometry_are_rejected() -> None:
    with pytest.raises(KeyError):
        bridge.flash_create(profile="no-such-part")
    with pytest.raises(ValueError):
        bridge.flash_create(size_bytes=3 * 1024)


@pytest.mark.parametrize(
    "call",
    [
        lambda i: bridge.flash_reset(id=i),
        lambda i: bridge.preload([1], id=i, addr=0),
        lambda i: bridge.preload_fill(id=i, addr=0, num_bytes=16, value=0),
        lambda i: bridge.check_content([0xFF] * 16, id=i, addr=0),
        lambda i: bridge.check_content_fill(id=i, addr=0, num_bytes=16, value=0xFF),
        lambda i: bridge.set_timing_enable(id=i, enable=1),
        lambda i: bridge.set_timing(id=i, name="tPP", seconds=1e-6),
        lambda i: bridge.set_protection(id=i, addr=0, num_bytes=16, locked=1),
    ],
)
def test_side_effecting_calls_return_zero(fid: int, call) -> None:
    # The bridge has no "takes arguments, returns nothing" form, so every
    # setter returns 0 for a throwaway VHDL variable.
    assert call(fid) == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda i: bridge.get_timing_limits(id=i),
        lambda i: bridge.read_back(id=i, addr=0, num_bytes=4),
        lambda i: bridge.written_regions(id=i),
    ],
)
def test_array_returns_are_int32_numpy(fid: int, call) -> None:
    value = call(fid)
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.int32


def test_timing_limits_order_and_length(fid: int) -> None:
    limits = bridge.get_timing_limits(id=fid)
    assert len(limits) == 10
    assert limits[0] == 7519  # t_sck_min_ps, 133 MHz
    assert limits[8] == limits[9] == 6000  # the two VC-side delays


def test_preload_and_read_back_round_trip(fid: int) -> None:
    bridge.preload(np.array([1, 2, 3], dtype=np.int32), id=fid, addr=0x100)
    assert list(bridge.read_back(id=fid, addr=0xFF, num_bytes=5)) == [
        0xFF,
        1,
        2,
        3,
        0xFF,
    ]


def test_preload_rejects_non_byte_values(fid: int) -> None:
    with pytest.raises(ValueError, match="element 1"):
        bridge.preload([0x00, 0x100], id=fid, addr=0)
    with pytest.raises(ValueError):
        bridge.preload([-1], id=fid, addr=0)


def test_check_content_raises_at_the_first_bad_byte(fid: int) -> None:
    bridge.preload([0xDE, 0xAD], id=fid, addr=0x10)
    assert bridge.check_content([0xDE, 0xAD], id=fid, addr=0x10) == 0
    with pytest.raises(AssertionError, match="0x00000011"):
        bridge.check_content([0xDE, 0xBE], id=fid, addr=0x10)


def test_check_content_fill_does_not_build_the_expectation(fid: int) -> None:
    assert bridge.check_content_fill(id=fid, addr=0, num_bytes=1 << 20, value=0xFF) == 0
    bridge.preload([0x00], id=fid, addr=0x8_0000)
    with pytest.raises(AssertionError, match="0x00080000"):
        bridge.check_content_fill(id=fid, addr=0, num_bytes=1 << 20, value=0xFF)


def test_written_regions_is_a_flat_addr_len_array(fid: int) -> None:
    assert list(bridge.written_regions(id=fid)) == []
    transaction(fid, [0x06])
    transaction(fid, [0x02, 0x00, 0x10, 0x00, 0x00, 0x00])
    assert list(bridge.written_regions(id=fid)) == [0x1000, 2]


def test_load_image(fid: int, tmp_path) -> None:
    path = tmp_path / "image.bin"
    path.write_bytes(bytes(range(4)))
    assert bridge.load_image(id=fid, path=str(path), fmt="bin", base=0x200) == 0
    assert list(bridge.read_back(id=fid, addr=0x200, num_bytes=4)) == [0, 1, 2, 3]


def test_set_protection_through_the_bridge(fid: int) -> None:
    bridge.set_protection(id=fid, addr=0x1000, num_bytes=0x1000, locked=1)
    transaction(fid, [0x06])
    transaction(fid, [0x02, 0x00, 0x10, 0x00, 0x00])
    assert list(bridge.read_back(id=fid, addr=0x1000, num_bytes=1)) == [0xFF]
    assert bridge.get_stat(id=fid, name="protect_reject_count") == 1
    bridge.set_protection(id=fid, addr=0x1000, num_bytes=0x1000, locked=0)
    transaction(fid, [0x06])
    transaction(fid, [0x02, 0x00, 0x10, 0x00, 0x00])
    assert list(bridge.read_back(id=fid, addr=0x1000, num_bytes=1)) == [0x00]


def test_flash_reset_keeps_the_array_but_drops_the_mode(fid: int) -> None:
    bridge.preload([0x5A], id=fid, addr=0)
    transaction(fid, [0xB7])  # EN4B
    assert bridge.get_stat(id=fid, name="addr_bytes") == 4
    assert bridge.flash_reset(id=fid) == 0
    assert bridge.get_stat(id=fid, name="addr_bytes") == 3
    assert list(bridge.read_back(id=fid, addr=0, num_bytes=1)) == [0x5A]


def test_unknown_stat_raises(fid: int) -> None:
    with pytest.raises(KeyError):
        bridge.get_stat(id=fid, name="not_a_stat")


# -- the wire, through the bridge exactly as the VC drives it ----------------


def transaction(instance: int, send: list[int], read: int = 0, now: float = 0.0):
    """cs_assert -> one xfer per byte -> cs_deassert, following the
    directives, which is all the VC ever does."""
    out: list[int] = []
    directive = unpack(bridge.cs_assert(id=instance, now_s=now))
    index = 0
    while True:
        if directive.action is Action.IGNORE_REST:
            break
        if directive.action is Action.RECEIVE:
            if index >= len(send):
                break
            byte = send[index]
            index += 1
        else:
            if len(out) >= read:
                break
            out.append(directive.byte_out)
            byte = -1
        kwargs = {"now_s": now} if directive.volatile else {}
        directive = unpack(bridge.xfer(id=instance, byte_in=byte, **kwargs))
    busy = bridge.cs_deassert(id=instance, trailing_bits=0, now_s=now)
    return out, busy


def test_a_full_read_transaction(fid: int) -> None:
    bridge.preload([0x11, 0x22], id=fid, addr=0x1234)
    out, busy = transaction(fid, [0x03, 0x00, 0x12, 0x34], read=2)
    assert out == [0x11, 0x22]
    assert busy == 0.0


def test_busy_comes_back_as_a_float_in_seconds() -> None:
    instance = bridge.flash_create()
    transaction(instance, [0x06])
    _, busy = transaction(instance, [0x02, 0, 0, 0, 0x00])
    assert isinstance(busy, float)
    assert busy == pytest.approx(700e-6)
    # WIP is derived from the last time the model was told about, so right
    # after the CS rise at t=0 the device is (correctly) still busy.
    assert bridge.get_stat(id=instance, name="wip") == 1
    assert bridge.get_stat(id=instance, name="busy_deadline_ps") == 700_000_000
    # Telling it a later time is all it takes; nothing has to clear a flag.
    bridge.cs_assert(id=instance, now_s=1.0)
    bridge.cs_deassert(id=instance, trailing_bits=0, now_s=1.0)
    assert bridge.get_stat(id=instance, name="wip") == 0


def test_xfer_accepts_an_omitted_now_s(fid: int) -> None:
    bridge.cs_assert(id=fid, now_s=1.0)
    packed = bridge.xfer(id=fid, byte_in=0x9F)
    assert unpack(packed).action is Action.TRANSMIT
    assert bridge.cs_deassert(id=fid, trailing_bits=0, now_s=1.0) == 0.0
