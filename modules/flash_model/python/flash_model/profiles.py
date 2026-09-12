"""Device profiles: plain dicts, one per part.

Adding a part must be a data entry, never code. Everything that differs
between two QSPI NOR flashes -- geometry, JEDEC ID, power-up addressing
mode, busy times, pin-level AC limits, reset defaults of the status
registers -- is a key here, and every other module reads it from the
profile rather than hard-coding a number. The default is a deliberately
generic 16 MiB JEDEC baseline so that a testbench which does not care about
a specific part does not have to name one.

`build()` layers explicit `flash_create` keyword arguments on top of a
profile, so a test can say "the generic part but 1 MiB and 4-byte
addressing" without inventing a profile for it.
"""

from __future__ import annotations

import copy

KIB = 1024
MIB = 1024 * 1024

# Generic 133 MHz QSPI NOR AC characteristics, picoseconds. `0` would mean
# "not specified, do not check" -- nothing here is 0, because a VC that
# checks nothing is a VC that finds nothing.
_BASE_LIMITS: dict[str, int] = {
    "t_sck_min_ps": 7_519,  # 133 MHz
    "t_sck_high_min_ps": 3_000,
    "t_sck_low_min_ps": 3_000,
    "t_slch_ps": 5_000,
    "t_chsh_ps": 5_000,
    "t_shsl_ps": 30_000,
    "t_dvch_ps": 2_000,
    "t_chdx_ps": 3_000,
    "t_clqv_ps": 6_000,  # VC output delay, not a check
    "t_shqz_ps": 6_000,  # VC output delay, not a check
}

# Typical, not worst-case: a testbench that wants worst-case overrides the
# one number it cares about with set_timing().
_BASE_TIMING: dict[str, float] = {
    "tPP": 700e-6,
    "tSE": 45e-3,
    "tBE32": 120e-3,
    "tBE64": 150e-3,
    "tCE": 20.0,
    "tW": 10e-3,
    "tRST": 30e-6,
    "tRES1": 3e-6,
    "tRES2": 1.8e-6,
}

# Status-register reset values. QE (SR2 bit 1) defaults set, because the
# model refuses quad commands without it and a testbench exercising 0xEB
# should not have to write SR2 first -- but it is a profile key precisely
# so that a test CAN clear it and prove the refusal.
_BASE_STATUS: dict[str, int] = {"sr1_default": 0x00, "sr2_default": 0x02, "sr3_default": 0x00}


def _profile(
    name: str,
    *,
    size_bytes: int,
    jedec_id: int,
    addr_bytes: int = 3,
    page_bytes: int = 256,
    sector_bytes: int = 4 * KIB,
    block32_bytes: int = 32 * KIB,
    block_bytes: int = 64 * KIB,
    **extra: object,
) -> dict:
    profile: dict = {
        "name": name,
        "size_bytes": size_bytes,
        "page_bytes": page_bytes,
        "sector_bytes": sector_bytes,
        "block32_bytes": block32_bytes,
        "block_bytes": block_bytes,
        "addr_bytes": addr_bytes,
        "jedec_id": jedec_id,
        "timing": dict(_BASE_TIMING),
        "limits": dict(_BASE_LIMITS),
    }
    profile.update(_BASE_STATUS)
    profile.update(extra)
    return profile


PROFILES: dict[str, dict] = {
    # The default. A 16 MiB / 128 Mbit part with 4 KiB sectors, 32 KiB and
    # 64 KiB blocks, 256-byte pages, powering up in 3-byte addressing.
    "generic_16mib": _profile("generic_16mib", size_bytes=16 * MIB, jedec_id=0xEF4018),
    "w25q128jv": _profile("w25q128jv", size_bytes=16 * MIB, jedec_id=0xEF4018),
    "w25q32jv": _profile("w25q32jv", size_bytes=4 * MIB, jedec_id=0xEF4016),
    # 32 MiB: large enough to need 4-byte addressing, powering up in 3-byte
    # mode like the real part, so EN4B/EX4B actually matter.
    "mt25ql256": _profile("mt25ql256", size_bytes=32 * MIB, jedec_id=0x20BA19),
    # Same geometry but powering up in 4-byte mode, for the other half of
    # the addressing matrix.
    "generic_32mib_4b": _profile(
        "generic_32mib_4b", size_bytes=32 * MIB, jedec_id=0xEF4019, addr_bytes=4
    ),
}

DEFAULT_PROFILE = "generic_16mib"

#: `flash_create` keyword arguments that override a profile field directly.
OVERRIDABLE = (
    "size_bytes",
    "page_bytes",
    "sector_bytes",
    "block32_bytes",
    "block_bytes",
    "addr_bytes",
    "jedec_id",
)


def get(name: str | None = None) -> dict:
    """A deep copy of one profile -- callers mutate their copy (overrides,
    per-instance timing) and must not disturb the table."""
    key = name or DEFAULT_PROFILE
    if key not in PROFILES:
        raise KeyError(f"unknown flash profile {key!r}; known: {sorted(PROFILES)}")
    return copy.deepcopy(PROFILES[key])


def electronic_id(profile: dict) -> int:
    """The legacy one-byte device ID returned by 0xAB, derived from the
    capacity code so it cannot contradict the JEDEC ID."""
    if "electronic_id" in profile:
        return int(profile["electronic_id"]) & 0xFF
    return (int(profile["jedec_id"]) & 0xFF) - 1 & 0xFF


def jedec_id_bytes(profile: dict) -> bytes:
    """Manufacturer, memory type, capacity -- the three bytes 0x9F clocks
    out, most significant first."""
    return int(profile["jedec_id"]).to_bytes(3, "big")


def build(profile: str | None = None, **overrides: object) -> dict:
    """A profile with `flash_create` overrides applied and validated."""
    result = get(profile)
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in OVERRIDABLE:
            raise KeyError(f"{key!r} is not an overridable profile field")
        result[key] = int(value)  # type: ignore[arg-type]
    _validate(result)
    return result


def _validate(profile: dict) -> None:
    size = int(profile["size_bytes"])
    page = int(profile["page_bytes"])
    if size <= 0 or size & (size - 1):
        raise ValueError(f"size_bytes={size} must be a positive power of two")
    if page <= 0 or page & (page - 1):
        raise ValueError(f"page_bytes={page} must be a positive power of two")
    if size % page:
        raise ValueError(f"size_bytes={size} is not a multiple of page_bytes={page}")
    if int(profile["addr_bytes"]) not in (3, 4):
        raise ValueError(f"addr_bytes={profile['addr_bytes']} must be 3 or 4")
    for key in ("sector_bytes", "block32_bytes", "block_bytes"):
        value = profile.get(key)
        if value and (value & (value - 1) or size % value):
            raise ValueError(f"{key}={value} must be a power of two dividing the device")
