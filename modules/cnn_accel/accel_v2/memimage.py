"""Sparse 64-bit-word external memory image and the section-7 CSV
memory-image format.

Authoritative source: `modules/cnn_accel/doc/cnn_accel_top_v2_arch.md`
section 7. One CSV schema serves program, weights, bias, scale, LUT,
inputs and outputs; the same `MemoryImage` also backs the model of the
`memory_t` the VUnit AXI slave BFM is bound to (section 11).
"""

from __future__ import annotations

WORD_BYTES = 8
_HEADER = "address,data"


class MemoryImage:
    """Sparse model of external DDR as a `{word_address: 64-bit value}`
    map. Only 8-byte-aligned whole words are ever stored; unwritten words
    are undefined (not present in `words()`/the CSV export) rather than
    implicitly zero, matching section 7's "sparse and order-independent"
    rule.
    """

    def __init__(self) -> None:
        self._words: dict[int, int] = {}

    # -- writes ---------------------------------------------------------

    def write_bytes(self, addr: int, data: bytes) -> None:
        """Write `data` starting at the 8-byte-aligned byte address
        `addr`. If `len(data)` is not a multiple of `WORD_BYTES`, the
        final word's trailing bytes (beyond `len(data)`) are zero-padded
        -- they are not merged with any previously-written value at that
        word."""
        if addr % WORD_BYTES != 0:
            raise ValueError(f"address 0x{addr:x} is not {WORD_BYTES}-byte aligned")
        if not data:
            return
        pad = (-len(data)) % WORD_BYTES
        padded = bytes(data) + b"\x00" * pad
        for i in range(0, len(padded), WORD_BYTES):
            word = int.from_bytes(padded[i : i + WORD_BYTES], "little")
            self._words[addr + i] = word

    def write_words(self, addr: int, words) -> None:
        """Write consecutive 64-bit `words` (an iterable of `int`,
        masked to 64 bits) starting at the 8-byte-aligned `addr`, one
        word every `WORD_BYTES` bytes."""
        if addr % WORD_BYTES != 0:
            raise ValueError(f"address 0x{addr:x} is not {WORD_BYTES}-byte aligned")
        for i, word in enumerate(words):
            self._words[addr + i * WORD_BYTES] = word & 0xFFFFFFFFFFFFFFFF

    # -- inspection -------------------------------------------------------

    def words(self) -> dict[int, int]:
        """Copy of the sparse `byte address -> 64-bit word` contents.

        Two images comparing equal here hold exactly the same bytes at
        exactly the same addresses (absent words are zero by contract, so
        an explicitly-written zero and an absent word are NOT equal -- a
        deliberate strictness, since a program that writes a zero word is
        a different program from one that never touches it)."""
        return dict(self._words)

    def words_in_range(self, lo: int, hi: int) -> dict[int, int]:
        """Words whose address falls in `[lo, hi)` -- e.g. splitting one
        `DdrMap` region out of a bigger image."""
        return {addr: value for addr, value in self._words.items() if lo <= addr < hi}

    def without_range(self, lo: int, hi: int) -> "MemoryImage":
        """Copy of this image with every word in `[lo, hi)` removed --
        the complement of `words_in_range`, e.g. for writing "everything
        except this one region" to a file while that region is seeded
        some other way."""
        copy = MemoryImage()
        copy._words = {addr: value for addr, value in self._words.items() if not (lo <= addr < hi)}
        return copy

    # -- reads ------------------------------------------------------------

    def read_bytes(self, addr: int, length: int) -> bytes:
        """Read `length` bytes starting at (not necessarily aligned)
        `addr`. Words not present in the image read back as zero."""
        if length < 0:
            raise ValueError(f"negative length {length}")
        if length == 0:
            return b""
        first_word = (addr // WORD_BYTES) * WORD_BYTES
        last_word = ((addr + length - 1) // WORD_BYTES) * WORD_BYTES
        out = bytearray()
        for word_addr in range(first_word, last_word + WORD_BYTES, WORD_BYTES):
            value = self._words.get(word_addr, 0)
            out += value.to_bytes(WORD_BYTES, "little")
        start = addr - first_word
        return bytes(out[start : start + length])

    # -- inspection -------------------------------------------------------

    def words(self) -> list[tuple[int, int]]:
        """`[(address, 64-bit value), ...]`, ascending by address."""
        return sorted(self._words.items())

    def size_bytes(self) -> int:
        """Byte address span of the image: `0` if nothing has been
        written, else `highest_written_word_address + WORD_BYTES` (the
        image is sparse, so this is a diagnostic upper bound, not a
        count of written bytes)."""
        if not self._words:
            return 0
        return max(self._words) + WORD_BYTES

    def regions(self) -> list[tuple[int, int]]:
        """`[(start_addr, length_bytes), ...]` contiguous runs of
        written words, ascending by address, merging adjacent words
        (`addr` and `addr + WORD_BYTES` both present) into one run. For
        diagnostics (e.g. summarizing a memory image in a log)."""
        addrs = sorted(self._words)
        if not addrs:
            return []
        out: list[tuple[int, int]] = []
        run_start = addrs[0]
        run_end = addrs[0] + WORD_BYTES  # exclusive
        for a in addrs[1:]:
            if a == run_end:
                run_end = a + WORD_BYTES
            else:
                out.append((run_start, run_end - run_start))
                run_start = a
                run_end = a + WORD_BYTES
        out.append((run_start, run_end - run_start))
        return out

    # -- CSV (section 7) --------------------------------------------------

    def write_csv(self, path: str, comment_lines=()) -> None:
        """Write this image in the section-7 CSV format: a mandatory
        `# cnn_accel memory image v1` line, then any `comment_lines`
        (each emitted as its own `#`-prefixed line), then the
        `address,data` header, then one `%08x,%016x` record per word in
        ascending address order."""
        lines = ["# cnn_accel memory image v1"]
        for comment in comment_lines:
            lines.append(comment if comment.startswith("#") else f"# {comment}")
        lines.append(_HEADER)
        for addr, value in self.words():
            lines.append(f"{addr:08x},{value:016x}")
        with open(path, "w", encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")

    @classmethod
    def read_csv(cls, path: str) -> "MemoryImage":
        """Parse the section-7 CSV format. Strict: raises `ValueError`
        (naming the 1-based line number) on a missing `address,data`
        header, malformed hex fields, a misaligned address, or a
        duplicate address."""
        image = cls()
        seen_header = False
        with open(path, "r", encoding="ascii") as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if not seen_header:
                    if line != _HEADER:
                        raise ValueError(
                            f"line {lineno}: expected header {_HEADER!r}, got {line!r}"
                        )
                    seen_header = True
                    continue
                parts = line.split(",")
                if len(parts) != 2:
                    raise ValueError(f"line {lineno}: expected 'address,data', got {line!r}")
                addr_str, data_str = parts
                if len(addr_str) != 8 or not _is_hex(addr_str):
                    raise ValueError(f"line {lineno}: malformed address {addr_str!r}")
                if len(data_str) != 16 or not _is_hex(data_str):
                    raise ValueError(f"line {lineno}: malformed data {data_str!r}")
                addr = int(addr_str, 16)
                value = int(data_str, 16)
                if addr % WORD_BYTES != 0:
                    raise ValueError(
                        f"line {lineno}: address 0x{addr:x} is not {WORD_BYTES}-byte aligned"
                    )
                if addr in image._words:
                    raise ValueError(f"line {lineno}: duplicate address 0x{addr:08x}")
                image._words[addr] = value
        if not seen_header:
            raise ValueError(f"missing required {_HEADER!r} header")
        return image


def _is_hex(s: str) -> bool:
    if not s:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in s)


__all__ = ["WORD_BYTES", "MemoryImage"]
