"""The only module the VHDL verification component calls.

`sim/flash_model.vhd` loads this file once per test process with
`python_execute`, then reaches the device model exclusively through the
functions below. Every function here does three things and nothing else:
unpack arguments, look up an instance, delegate. All device behaviour lives
in the `flash_model` package -- if a change ever seems to belong in this
file, it belongs in `flash_model/device.py` instead.

Conventions, from `shared/Vunit.md` section 7 and
`doc/flash_model_ffi_contract.md`:

* one `python_call` carries at most one positional scalar (or any number of
  `integer_array_t`) and returns exactly one value, so every function takes
  its instance `id` through `kw()` and pure side-effecting calls return `0`;
* functions the contract types as `int32[]` return
  `np.array(..., dtype=np.int32)`, never a Python list;
* errors are raised, never encoded in a return value. `python_call` turns an
  uncaught exception into a VUnit FAILURE carrying the full traceback, which
  localizes a bug far better than a status code the VHDL side has to check.

The instance handle is spelled `id` throughout, shadowing the builtin,
because that is the keyword name the contract and the VHDL side use; a
bridge whose keyword names differ from the contract is a bridge nobody can
call.

Module-level state is per simulator process (one per VUnit test config), so
the instance registry never leaks between tests.
"""

from __future__ import annotations

import numpy as np
from flash_model import profiles
from flash_model.device import FlashDevice
from flash_model.directive import LAYOUT_VERSION

#: instance id -> device. Ids start at 1 so that 0 is never a valid handle.
_INSTANCES: dict[int, FlashDevice] = {}
_NEXT_ID = 1


def _device(id: int) -> FlashDevice:
    key = int(id)
    device = _INSTANCES.get(key)
    if device is None:
        raise KeyError(
            f"no flash model instance {key}; flash_create() returns the id to use "
            f"(live ids: {sorted(_INSTANCES)})"
        )
    return device


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
    global _NEXT_ID
    built = profiles.build(
        profile,
        size_bytes=size_bytes,
        page_bytes=page_bytes,
        sector_bytes=sector_bytes,
        block_bytes=block_bytes,
        addr_bytes=addr_bytes,
        jedec_id=jedec_id,
    )
    instance_id = _NEXT_ID
    _NEXT_ID += 1
    _INSTANCES[instance_id] = FlashDevice(built)
    return instance_id


def flash_reset(id: int) -> int:
    """Power-on reset of the volatile state. The array is untouched -- a
    reset is not an erase."""
    _device(id).reset_state()
    return 0


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


def preload(data, id: int, addr: int) -> int:
    _device(id).preload(int(addr), _bytes(data))
    return 0


def preload_fill(id: int, addr: int, num_bytes: int, value: int) -> int:
    _device(id).preload_fill(int(addr), int(num_bytes), int(value))
    return 0


def load_image(id: int, path: str, fmt: str | None = None, base: int = 0) -> int:
    _device(id).load_image(str(path), fmt, int(base))
    return 0


def read_back(id: int, addr: int, num_bytes: int) -> np.ndarray:
    return _int32(_device(id).read_back(int(addr), int(num_bytes)))


def check_content(expected, id: int, addr: int) -> int:
    _device(id).check_content(int(addr), _bytes(expected))
    return 0


def check_content_fill(id: int, addr: int, num_bytes: int, value: int) -> int:
    _device(id).check_content_fill(int(addr), int(num_bytes), int(value))
    return 0


def written_regions(id: int) -> np.ndarray:
    """Flat `[addr, len, addr, len, ...]` of everything the device
    programmed or erased, coalesced."""
    flat: list[int] = []
    for addr, length in _device(id).written_regions():
        flat += [addr, length]
    return _int32(flat)


def set_timing_enable(id: int, enable: bool | int) -> int:
    """`enable = 0` collapses every busy time to zero, for the majority of
    tests that care about protocol rather than milliseconds."""
    _device(id).set_timing_enable(bool(enable))
    return 0


def set_timing(id: int, name: str, seconds: float) -> int:
    _device(id).set_timing(str(name), float(seconds))
    return 0


def set_protection(id: int, addr: int, num_bytes: int, locked: bool | int) -> int:
    """Lock or unlock a region. A program or erase touching a locked region
    is silently ignored, exactly as on silicon."""
    _device(id).set_protection(int(addr), int(num_bytes), bool(locked))
    return 0


def get_stat(id: int, name: str) -> int:
    return _device(id).get_stat(str(name))
