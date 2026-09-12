"""Two unrelated kinds of time, deliberately kept in one module because
both are "numbers a profile supplies and a testbench overrides".

**Busy time** (tPP, tSE, ...) is how long the device holds WIP after a
program or erase. It is modelled as a *deadline*, not a flag: `cs_deassert`
computes `deadline = now_s + duration` and every later query derives
`WIP = now_s < deadline`. There is no busy flag anywhere in the model.

Why a deadline and not a flag: the Python model has no clock. It only ever
learns the time when VHDL tells it, and VHDL only calls at CS edges and
byte boundaries. A flag would have to be cleared by *someone*, and there is
no one -- the only honest representation of "busy until t" is t. It also
makes the model immune to the testbench polling at arbitrary times, and
makes `set_timing_enable(False)` a one-line change of semantics (every
duration becomes 0, so every deadline is already in the past) instead of a
special case threaded through the state machine.

**Pin-level AC limits** (`get_timing_limits`) are a fixed-order array of
picosecond values the VC checks against the DUT's pins. The order is part
of the FFI contract; VHDL indexes it with named constants. Indices 0..7 are
checks on DUT-driven signals; 8..9 are delays the VC applies to its own
outputs (asserting on those would make the VC fail on itself).
"""

from __future__ import annotations

# Busy-time keys, in the order a human thinks about them. Every key must
# exist in a profile's `timing` dict; `set_timing` rejects anything else,
# because silently accepting "tPp" would leave the override with no effect
# and the test quietly passing for the wrong reason.
BUSY_KEYS: tuple[str, ...] = (
    "tPP",  # page program
    "tSE",  # sector erase (4 KiB)
    "tBE32",  # block erase (32 KiB)
    "tBE64",  # block erase (64 KiB)
    "tCE",  # chip erase
    "tW",  # write status register
    "tRST",  # software reset recovery
    "tRES1",  # release from deep power-down to standby
    "tRES2",  # release from deep power-down with electronic ID read
)

#: Fixed order of `get_timing_limits`. Index meaning is contract, not
#: convention -- see doc/flash_model_ffi_contract.md.
LIMIT_KEYS: tuple[str, ...] = (
    "t_sck_min_ps",  # 0  minimum SCK period
    "t_sck_high_min_ps",  # 1
    "t_sck_low_min_ps",  # 2
    "t_slch_ps",  # 3  CS low to first SCK edge
    "t_chsh_ps",  # 4  last SCK edge to CS high
    "t_shsl_ps",  # 5  CS deselect between commands
    "t_dvch_ps",  # 6  data in setup
    "t_chdx_ps",  # 7  data in hold
    "t_clqv_ps",  # 8  clock to output valid (VC delay, not a check)
    "t_shqz_ps",  # 9  CS high to output Hi-Z (VC delay, not a check)
)


class Timing:
    """Busy-time table and the WIP deadline for one device instance."""

    def __init__(
        self, busy_seconds: dict[str, float], limits_ps: dict[str, int]
    ) -> None:
        missing = set(BUSY_KEYS) - set(busy_seconds)
        if missing:
            raise ValueError(f"profile is missing busy times: {sorted(missing)}")
        self._busy = {k: float(busy_seconds[k]) for k in BUSY_KEYS}
        self._limits = {k: int(limits_ps.get(k, 0)) for k in LIMIT_KEYS}
        self.enabled = True
        self._deadline = 0.0

    # -- table -------------------------------------------------------------

    def set_busy(self, name: str, seconds: float) -> None:
        """Override one busy time. Unknown names raise: a typo'd override
        that silently did nothing would be indistinguishable from a model
        bug."""
        if name not in self._busy:
            raise KeyError(f"unknown timing name {name!r}; known: {list(BUSY_KEYS)}")
        if seconds < 0:
            raise ValueError(f"timing {name} must not be negative (got {seconds})")
        self._busy[name] = float(seconds)

    def set_enable(self, enable: bool) -> None:
        """`False` collapses every busy time to zero. For the common test
        that cares about protocol, not milliseconds -- and it must collapse
        *all* of them, so no test can accidentally depend on one op still
        being slow."""
        self.enabled = bool(enable)

    def busy_seconds(self, name: str | None) -> float:
        if name is None or not self.enabled:
            return 0.0
        return self._busy[name]

    def limits_ps(self) -> list[int]:
        """The contract-ordered picosecond array. `0` means "not specified,
        do not check"."""
        return [self._limits[k] for k in LIMIT_KEYS]

    # -- the deadline ------------------------------------------------------

    def start_busy(self, now_s: float, name: str | None) -> float:
        """Arm the WIP deadline and return the duration the VC should
        expect. A zero duration leaves the deadline in the past, so WIP is
        never observed -- no special case needed."""
        duration = self.busy_seconds(name)
        self._deadline = now_s + duration
        return duration

    def is_busy(self, now_s: float) -> bool:
        """WIP, derived rather than stored."""
        return now_s < self._deadline

    def deadline(self) -> float:
        return self._deadline

    def clear_busy(self) -> None:
        """Used by a reset, which aborts whatever was in progress."""
        self._deadline = 0.0
