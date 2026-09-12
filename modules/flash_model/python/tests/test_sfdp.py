"""SFDP: structurally valid, and -- the part that actually matters --
agreeing with the profile and the opcode table it was built from."""

from __future__ import annotations

import pytest
from flash_model import profiles, sfdp
from flash_model.commands import COMMANDS


def dwords(profile: dict) -> list[int]:
    return sfdp.basic_parameter_table(profile)


def test_signature_and_header() -> None:
    image = sfdp.build(profiles.build())
    assert image[0:4] == b"SFDP"
    assert image[4] == sfdp.SFDP_MINOR
    assert image[5] == sfdp.SFDP_MAJOR
    assert image[6] == 0x00  # one parameter header (NPH is "minus one")


def test_parameter_header_points_at_the_basic_table() -> None:
    image = sfdp.build(profiles.build())
    header = image[sfdp.PARAM_HEADER_OFFSET : sfdp.PARAM_HEADER_OFFSET + 8]
    assert header[0] == 0x00 and header[7] == 0xFF  # JEDEC basic, ID 0xFF00
    assert header[3] == sfdp.BASIC_TABLE_DWORDS
    pointer = int.from_bytes(header[4:7], "little")
    assert pointer == sfdp.PARAM_TABLE_OFFSET
    assert len(image) == pointer + 4 * sfdp.BASIC_TABLE_DWORDS


@pytest.mark.parametrize("name", ["generic_16mib", "w25q32jv", "mt25ql256"])
def test_density_agrees_with_the_profile(name: str) -> None:
    profile = profiles.build(name)
    density = dwords(profile)[1]
    assert density >> 31 == 0, "these parts are all under 2 Gbit"
    assert density + 1 == profile["size_bytes"] * 8


def test_erase_types_agree_with_the_command_table() -> None:
    profile = profiles.build()
    table = dwords(profile)
    types = [
        (table[7] & 0xFF, (table[7] >> 8) & 0xFF),
        ((table[7] >> 16) & 0xFF, (table[7] >> 24) & 0xFF),
        (table[8] & 0xFF, (table[8] >> 8) & 0xFF),
    ]
    assert types[0] == (12, 0x20)  # 4 KiB
    assert types[1] == (15, 0x52)  # 32 KiB
    assert types[2] == (16, 0xD8)  # 64 KiB
    for exponent, opcode in types:
        assert COMMANDS[opcode].erase_bytes == 1 << exponent
    # The unused fourth slot is the "no such erase type" encoding.
    assert ((table[8] >> 16) & 0xFF, (table[8] >> 24) & 0xFF) == (0, 0xFF)


def test_dword1_advertises_what_the_model_implements() -> None:
    table = dwords(profiles.build())
    dword1 = table[0]
    assert dword1 & 0b11 == 0b01  # uniform 4 KiB erase
    assert (dword1 >> 8) & 0xFF == 0x20  # and its opcode
    assert (dword1 >> 16) & 1  # 1-1-2 (0x3B)
    assert (dword1 >> 20) & 1  # 1-2-2 (0xBB)
    assert (dword1 >> 21) & 1  # 1-4-4 (0xEB)
    assert (dword1 >> 22) & 1  # 1-1-4 (0x6B)
    assert (dword1 >> 17) & 0b11 == 0b01  # both 3- and 4-byte addressing


def test_fast_read_dummy_cycles_match_the_command_table() -> None:
    table = dwords(profiles.build())
    assert (table[2] & 0x1F, (table[2] >> 8) & 0xFF) == (COMMANDS[0xEB].dummy_cycles, 0xEB)
    assert ((table[2] >> 16) & 0x1F, (table[2] >> 24) & 0xFF) == (
        COMMANDS[0x6B].dummy_cycles,
        0x6B,
    )
    assert (table[3] & 0x1F, (table[3] >> 8) & 0xFF) == (COMMANDS[0x3B].dummy_cycles, 0x3B)
    assert ((table[3] >> 16) & 0x1F, (table[3] >> 24) & 0xFF) == (
        COMMANDS[0xBB].dummy_cycles,
        0xBB,
    )


def test_qpi_support_is_advertised() -> None:
    table = dwords(profiles.build())
    assert (table[4] >> 4) & 1  # supports 4-4-4
    assert not table[4] & 1  # does not claim 2-2-2, which the model lacks


def test_reads_past_the_table_return_0xff() -> None:
    image = sfdp.build(profiles.build())
    assert sfdp.read(image, len(image), 4) == b"\xff" * 4
    tail = sfdp.read(image, len(image) - 2, 4)
    assert tail[2:] == b"\xff\xff"
    assert sfdp.read(image, 0, 4) == b"SFDP"


def test_a_profile_whose_sector_size_has_no_opcode_is_rejected() -> None:
    profile = profiles.build()
    profile["sector_bytes"] = 1024
    with pytest.raises(ValueError, match="erase opcode"):
        sfdp.basic_parameter_table(profile)
