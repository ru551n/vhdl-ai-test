"""Tests for `accel_v2.memimage`: sparse image reads/writes and the
section-7 CSV round-trip."""

from __future__ import annotations

import os
import random
import tempfile

import pytest

from accel_v2.memimage import WORD_BYTES, MemoryImage


def test_write_bytes_aligned_read_back() -> None:
    img = MemoryImage()
    img.write_bytes(0x100, bytes(range(16)))
    assert img.read_bytes(0x100, 16) == bytes(range(16))
    assert img.words() == [
        (0x100, int.from_bytes(bytes(range(8)), "little")),
        (0x108, int.from_bytes(bytes(range(8, 16)), "little")),
    ]


def test_write_bytes_rejects_misaligned_address() -> None:
    img = MemoryImage()
    with pytest.raises(ValueError):
        img.write_bytes(0x101, b"\x00" * 8)


def test_write_bytes_pads_tail_with_zeros() -> None:
    img = MemoryImage()
    img.write_bytes(0x0, b"\x01\x02\x03")
    (addr, value), = img.words()
    assert addr == 0
    assert value == 0x0000000000030201
    assert img.read_bytes(0, 8) == b"\x01\x02\x03\x00\x00\x00\x00\x00"


def test_write_bytes_partial_tail_does_not_merge_with_prior_write() -> None:
    """A short second write to an address that already holds a full word
    zero-pads its own tail rather than preserving the old word's
    trailing bytes -- the spec calls this "pads the tail with zeros to a
    whole word", not "merges with existing contents"."""
    img = MemoryImage()
    img.write_bytes(0x0, bytes([0xFF] * 8))
    img.write_bytes(0x0, bytes([0xAA, 0xBB]))
    assert img.read_bytes(0x0, 8) == bytes([0xAA, 0xBB, 0, 0, 0, 0, 0, 0])


def test_write_words_and_read_bytes_unaligned_span() -> None:
    img = MemoryImage()
    img.write_words(0x0, [0x1122334455667788, 0xAABBCCDDEEFF0011])
    # Read a span that straddles both words, starting mid-word.
    data = img.read_bytes(0x4, 8)
    expected = (0x1122334455667788).to_bytes(8, "little")[4:] + (
        0xAABBCCDDEEFF0011
    ).to_bytes(8, "little")[:4]
    assert data == expected


def test_read_bytes_unwritten_reads_zero() -> None:
    img = MemoryImage()
    assert img.read_bytes(0x1000, 8) == b"\x00" * 8


def test_size_bytes_and_regions() -> None:
    img = MemoryImage()
    assert img.size_bytes() == 0
    assert img.regions() == []
    img.write_words(0x0, [1, 2, 3])  # 0x0, 0x8, 0x10
    img.write_words(0x100, [4])
    assert img.size_bytes() == 0x108
    assert img.regions() == [(0x0, 0x18), (0x100, 0x8)]


def _random_image(seed: int) -> MemoryImage:
    rng = random.Random(seed)
    img = MemoryImage()
    for _ in range(64):
        addr = rng.randrange(0, 0x10000, WORD_BYTES)
        img.write_words(addr, [rng.randrange(0, 2**64)])
    return img


def test_csv_round_trip() -> None:
    img = _random_image(seed=42)
    path = os.path.join(tempfile.mkdtemp(), "image.csv")
    img.write_csv(path, comment_lines=["provenance: test_memimage.py"])
    reloaded = MemoryImage.read_csv(path)
    assert reloaded.words() == img.words()


def test_csv_format_matches_section_7_schema() -> None:
    img = MemoryImage()
    img.write_words(0x1000, [1])
    img.write_words(0x1008, [0x1000])
    path = os.path.join(tempfile.mkdtemp(), "image.csv")
    img.write_csv(path)
    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0] == "# cnn_accel memory image v1"
    assert lines[1] == "address,data"
    assert lines[2] == "00001000,0000000000000001"
    assert lines[3] == "00001008,0000000000001000"


def test_read_csv_rejects_missing_header() -> None:
    path = os.path.join(tempfile.mkdtemp(), "bad.csv")
    with open(path, "w") as f:
        f.write("# cnn_accel memory image v1\n00001000,0000000000000001\n")
    with pytest.raises(ValueError, match="header"):
        MemoryImage.read_csv(path)


def test_read_csv_rejects_malformed_hex() -> None:
    path = os.path.join(tempfile.mkdtemp(), "bad.csv")
    with open(path, "w") as f:
        f.write("address,data\nZZZZZZZZ,0000000000000001\n")
    with pytest.raises(ValueError, match="line 2"):
        MemoryImage.read_csv(path)


def test_read_csv_rejects_unaligned_address() -> None:
    path = os.path.join(tempfile.mkdtemp(), "bad.csv")
    with open(path, "w") as f:
        f.write("address,data\n00001001,0000000000000001\n")
    with pytest.raises(ValueError, match="line 2"):
        MemoryImage.read_csv(path)


def test_read_csv_rejects_duplicate_address() -> None:
    path = os.path.join(tempfile.mkdtemp(), "bad.csv")
    with open(path, "w") as f:
        f.write(
            "address,data\n"
            "00001000,0000000000000001\n"
            "00001000,0000000000000002\n"
        )
    with pytest.raises(ValueError, match="line 3"):
        MemoryImage.read_csv(path)
