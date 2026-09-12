"""Profiles: adding a part must stay a data entry, and every entry must be
usable without further code."""

from __future__ import annotations

import pytest
from flash_model import profiles
from flash_model.device import FlashDevice
from flash_model.timing import BUSY_KEYS, LIMIT_KEYS


@pytest.mark.parametrize("name", sorted(profiles.PROFILES))
def test_every_profile_builds_a_working_device(name: str) -> None:
    device = FlashDevice(profiles.build(name))
    assert device.size_bytes == profiles.PROFILES[name]["size_bytes"]
    assert device.read_back(device.size_bytes - 1, 1) == b"\xff"
    assert len(device.jedec_bytes) == 3
    assert device.sfdp_image[:4] == b"SFDP"


@pytest.mark.parametrize("name", sorted(profiles.PROFILES))
def test_every_profile_carries_a_complete_timing_table(name: str) -> None:
    profile = profiles.PROFILES[name]
    assert set(profile["timing"]) == set(BUSY_KEYS)
    assert set(profile["limits"]) == set(LIMIT_KEYS)


def test_get_returns_a_copy_so_instances_cannot_poison_the_table() -> None:
    first = profiles.get()
    first["timing"]["tPP"] = 99.0
    assert profiles.get()["timing"]["tPP"] == pytest.approx(700e-6)


def test_the_default_profile_is_the_generic_16mib_baseline() -> None:
    assert profiles.DEFAULT_PROFILE == "generic_16mib"
    profile = profiles.build()
    assert profile["size_bytes"] == 16 * 1024 * 1024
    assert profile["page_bytes"] == 256
    assert profile["sector_bytes"] == 4096
    assert profile["block_bytes"] == 65536
    assert profile["addr_bytes"] == 3


def test_overrides_are_validated_not_trusted() -> None:
    assert profiles.build(size_bytes=1 << 20)["size_bytes"] == 1 << 20
    with pytest.raises(ValueError, match="power of two"):
        profiles.build(size_bytes=1_000_000)
    with pytest.raises(ValueError, match="addr_bytes"):
        profiles.build(addr_bytes=2)
    with pytest.raises(KeyError, match="overridable"):
        profiles.build(name="nope")


def test_none_overrides_are_ignored_so_vhdl_can_omit_them() -> None:
    assert profiles.build(None, size_bytes=None, jedec_id=None) == profiles.build()


def test_electronic_id_follows_the_capacity_code() -> None:
    assert profiles.electronic_id(profiles.build()) == 0x17  # 0x18 - 1
    assert profiles.electronic_id(profiles.build(jedec_id=0xEF4016)) == 0x15
