"""`cnnc.target`: the compiler-facing description of what a backend can
execute (`Target`), derived from the accelerator's own source of truth
rather than hard-coded -- see `discover.py`.
"""

from .contract import CapabilityError, Target, TargetError
from .load import load_target

__all__ = ["CapabilityError", "Target", "TargetError", "load_target"]
