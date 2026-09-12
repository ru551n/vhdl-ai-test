"""Live device instances, keyed by the id handed back to VHDL.

This deliberately does NOT live in `flash_model_bridge.py`, and the reason is
subtle enough to be worth spelling out.

Every flash_model verification component loads the bridge itself, by calling
`python_execute` from its own init process, so that a testbench never has to
know the model is Python. `python_execute` *runs the file's module-level code*
— so with two VC instances in one testbench the bridge file is executed twice,
and a registry defined at the bridge's module level would be reset to empty by
the second execution. Both components would then be handed id 1 and would
address the same device: instance A would silently read instance B's array.

Imported modules are different. The bridge does `from flash_model import
registry`, and an import resolves through `sys.modules`, which the simulator's
single embedded interpreter keeps for the whole simulation. Re-executing the
bridge re-imports this module rather than re-running it, so the registry
survives.

Ids start at 1 so that 0 is never a valid handle, and the VHDL side's "no
instance yet" sentinel (-1) can never collide with a real one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flash_model.device import FlashDevice

_INSTANCES: dict[int, FlashDevice] = {}
_NEXT_ID = 1


def add(device: FlashDevice) -> int:
    """Register `device` and return the id VHDL will address it by."""
    global _NEXT_ID
    instance_id = _NEXT_ID
    _NEXT_ID += 1
    _INSTANCES[instance_id] = device
    return instance_id


def get(instance_id: int) -> FlashDevice:
    """The device for `instance_id`, or a KeyError naming the live ids."""
    key = int(instance_id)
    device = _INSTANCES.get(key)
    if device is None:
        raise KeyError(
            f"no flash model instance {key}; flash_create() returns the id to use "
            f"(live ids: {sorted(_INSTANCES)})"
        )
    return device


def live_ids() -> list[int]:
    return sorted(_INSTANCES)


def clear() -> None:
    """Drop every instance. For tests only — a simulation never needs it."""
    global _NEXT_ID
    _INSTANCES.clear()
    _NEXT_ID = 1
