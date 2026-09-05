from __future__ import annotations

from tsfpga.module import BaseModule


class Module(BaseModule):
    """No `setup_vunit`/`get_build_projects` yet: `modules/cnn_accel/src/`
    currently holds only `cnn_accel_pkg.vhd` (a package, not an entity) and
    has no testbenches. `tsfpga.module.get_modules()` still needs this file
    to register `modules/cnn_accel/` as the `cnn_accel` VUnit library (its
    package compiles as part of every other module's/testbench's
    compilation via `run.py`'s dependency resolution), even with an empty
    `Module` body. Add `setup_vunit`/`get_build_projects` here as entities
    and testbenches are added, following `module_canny.py`'s pattern."""
