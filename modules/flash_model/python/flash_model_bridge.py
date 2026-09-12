"""The only module the VHDL verification component calls.

`sim/flash_model.vhd` loads this file once per component with `exec_file`,
then reaches the device model exclusively through the functions below. Every
function here does three things and nothing else: unpack arguments, look up an
instance, delegate. All device behaviour lives in the `flash_model` package --
if a change ever seems to belong in this file, it belongs in
`flash_model/device.py` instead.

Conventions, from `doc/flash_model_ffi_contract.md` (the VHDL side uses VUnit's
upstream `python_pkg` API):

* the VHDL side calls `call(identifier, arg(...), kwarg(...))`, so every
  function takes its instance `id` as a keyword argument, and array data as its
  one positional argument;
* side-effecting functions return `None`: the VHDL side uses the procedure
  form of `call`, which discards the result;
* functions the contract types as `int32[]` return
  `np.array(..., dtype=np.int32)`, never a Python list. Results are converted
  strictly on the VHDL side, so an `int` must never come back where the
  contract says `real`;
* errors are raised, never encoded in a return value. The bridge turns an
  uncaught exception into a failure on the `vunit_lib:python` logger carrying
  the full traceback, which localizes a bug far better than a status code the
  VHDL side has to check.

The instance handle is spelled `id` throughout, shadowing the builtin,
because that is the keyword name the contract and the VHDL side use; a
bridge whose keyword names differ from the contract is a bridge nobody can
call.

Module-level state is per simulator process (one per VUnit test config), so
nothing leaks between tests. The instance registry deliberately lives in
`flash_model.registry` rather than here: every VC loads this file with
`exec_file`, which RE-RUNS it, so a registry defined at this module's level
would be wiped by the second component's load and both would be handed the
same id. See that module's docstring.
"""

from __future__ import annotations

import numpy as np
from flash_model import profiles, registry
from flash_model.device import FlashDevice
from flash_model.directive import LAYOUT_VERSION


def _device(id: int) -> FlashDevice:
    return registry.get(id)


def _int32(values) -> np.ndarray:
    """The array form every `integer_array_t` return value takes."""
    return np.array(list(values), dtype=np.int32)


def _bytes(values) -> bytes:
    """One `integer_array_t` of byte values from VHDL, checked. VUnit hands
    these over as a sequence of Python ints."""
    out = bytearray()
    for index, value in enumerate(values):
        byte = int(value)
        if not 0 <= byte <= 0xFF:
            raise ValueError(f"element {index} = {byte} is not a byte value")
        out.append(byte)
    return bytes(out)


# -- contract guard --------------------------------------------------------


def layout_version() -> int:
    """Bumped whenever the packed directive or the timing-limit order
    changes. The VC asserts this at instance creation, so a drift between
    the two hand-written halves of the contract fails at time 0."""
    return LAYOUT_VERSION


# -- lifecycle ---------------------------------------------------------------


def flash_create(
    profile: str | None = None,
    size_bytes: int | None = None,
    page_bytes: int | None = None,
    sector_bytes: int | None = None,
    block_bytes: int | None = None,
    addr_bytes: int | None = None,
    jedec_id: int | None = None,
) -> int:
    """Create a device and return its instance id. Every geometry argument
    is optional and overrides the named profile."""
    built = profiles.build(
        profile,
        size_bytes=size_bytes,
        page_bytes=page_bytes,
        sector_bytes=sector_bytes,
        block_bytes=block_bytes,
        addr_bytes=addr_bytes,
        jedec_id=jedec_id,
    )
    return registry.add(FlashDevice(built))


def flash_reset(id: int) -> None:
    """Power-on reset of the volatile state. The array is untouched -- a
    reset is not an erase."""
    _device(id).reset_state()


# -- the wire ----------------------------------------------------------------


def cs_assert(id: int, now_s: float) -> int:
    return _device(id).cs_assert(float(now_s))


def xfer(id: int, byte_in: int, now_s: float | None = None) -> int:
    """One byte on the wire. `byte_in` is -1 when the VC clocked a byte
    out rather than in."""
    return _device(id).xfer(int(byte_in), None if now_s is None else float(now_s))


def cs_deassert(id: int, trailing_bits: int, now_s: float) -> float:
    """Returns the busy time in seconds; 0.0 when nothing went busy."""
    return _device(id).cs_deassert(int(trailing_bits), float(now_s))


def get_timing_limits(id: int) -> np.ndarray:
    """Pin-level AC limits in picoseconds, in the contract's fixed order."""
    return _int32(_device(id).timing_limits_ps())


# -- control plane -------------------------------------------------------------


def preload(data, id: int, addr: int) -> None:
    _device(id).preload(int(addr), _bytes(data))


def preload_fill(id: int, addr: int, num_bytes: int, value: int) -> None:
    _device(id).preload_fill(int(addr), int(num_bytes), int(value))


def load_image(id: int, path: str, fmt: str | None = None, base: int = 0) -> None:
    _device(id).load_image(str(path), fmt, int(base))


def read_back(id: int, addr: int, num_bytes: int) -> np.ndarray:
    return _int32(_device(id).read_back(int(addr), int(num_bytes)))


def check_content(expected, id: int, addr: int) -> None:
    _device(id).check_content(int(addr), _bytes(expected))


def check_content_fill(id: int, addr: int, num_bytes: int, value: int) -> None:
    _device(id).check_content_fill(int(addr), int(num_bytes), int(value))


def written_regions(id: int) -> np.ndarray:
    """Flat `[addr, len, addr, len, ...]` of everything the device
    programmed or erased, coalesced."""
    flat: list[int] = []
    for addr, length in _device(id).written_regions():
        flat += [addr, length]
    return _int32(flat)


def set_timing_enable(id: int, enable: bool | int) -> None:
    """`enable = 0` collapses every busy time to zero, for the majority of
    tests that care about protocol rather than milliseconds."""
    _device(id).set_timing_enable(bool(enable))


def set_timing(id: int, name: str, seconds: float) -> None:
    _device(id).set_timing(str(name), float(seconds))


def set_protection(id: int, addr: int, num_bytes: int, locked: bool | int) -> None:
    """Lock or unlock a region. A program or erase touching a locked region
    is silently ignored, exactly as on silicon."""
    _device(id).set_protection(int(addr), int(num_bytes), bool(locked))


def get_stat(id: int, name: str) -> int:
    return _device(id).get_stat(str(name))
