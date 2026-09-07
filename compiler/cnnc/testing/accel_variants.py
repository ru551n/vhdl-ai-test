"""Test-only `cnn_accel` variants that must never touch `modules/cnn_accel/`
(the real HW source of truth, still round-half-even until HW milestone
H0 lands).

`half_up_accel_root` copies `cnn_accel_constants.py`/`cnn_accel_model.py`
into a scratch directory with `round_shift_right_signed`'s `convergent`
default flipped to `False` (round-half-up), so `discover.discover_cnn_accel`
probes a half-up target without editing the checked-in golden model. The
substring match is asserted to occur exactly once so an unrelated future
change to that signature fails loudly here instead of silently no-op'ing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from cnnc.target.contract import Target
from cnnc.target.discover import discover_cnn_accel

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPO_ROOT / "modules" / "cnn_accel"

_PATTERN = "convergent: bool = True"
_REPLACEMENT = "convergent: bool = False"


def half_up_accel_root(tmp_dir: Path) -> Path:
    """Copy `cnn_accel_constants.py`/`cnn_accel_model.py` into `tmp_dir`
    with `round_shift_right_signed`'s default rounding flipped to
    half-up. Returns `tmp_dir`."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_SOURCE_ROOT / "cnn_accel_constants.py", tmp_dir / "cnn_accel_constants.py")

    model_src = (_SOURCE_ROOT / "cnn_accel_model.py").read_text()
    count = model_src.count(_PATTERN)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one occurrence of {_PATTERN!r} in cnn_accel_model.py, found {count}; "
            "round_shift_right_signed's signature changed upstream, update this test double"
        )
    (tmp_dir / "cnn_accel_model.py").write_text(model_src.replace(_PATTERN, _REPLACEMENT))
    return tmp_dir


def load_half_up_target(tmp_dir: Path) -> Target:
    """`discover_cnn_accel` against a half-up `half_up_accel_root(tmp_dir)`."""
    return discover_cnn_accel(half_up_accel_root(tmp_dir))
