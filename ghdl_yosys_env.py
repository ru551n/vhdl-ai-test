"""Shared local-machine GHDL/Yosys environment resolution for this
project's ``module_*.py`` ``get_build_projects()`` methods
(``modules/cnn_accel/module_cnn_accel.py``).

Deliberately project-scoped, not MCP-specific: ``build_fpga.py`` and every
``get_build_projects()`` that calls this must behave identically whether
invoked directly from a terminal, from CI, or via any MCP wrapper around
``build_fpga.py`` -- an env var literally named after one particular MCP
server (the previous ``TSFPGA_MCP_GHDL_PLUGIN``/``TSFPGA_MCP_GHDL_PREFIX``
names) would violate that; a human running ``python3 build_fpga.py`` in a
plain terminal has no reason to know or set an "MCP" variable, and tsfpga
itself (the library ``YosysNetlistBuild`` comes from) has no concept of
MCP at all.

``GHDL_YOSYS_PLUGIN``/``GHDL_YOSYS_PREFIX`` are this project's own
override-only env var names. If unset, ``resolve_ghdl_plugin_path()``
falls back to this machine's system-wide ``ghdl-yosys-plugin`` install
path, so a plain ``python3 build_fpga.py`` needs no environment setup at
all on a machine where that system install exists.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Installed system-wide (see ~sebbe memory `tsfpga_ghdl_plugin_env.md`).
_SYSTEM_GHDL_PLUGIN_PATH = Path("/usr/local/share/yosys/plugins/ghdl.so")


def resolve_ghdl_plugin_path() -> Path | None:
    """Path to pass as ``YosysNetlistBuild(ghdl_plugin_path=...)``, or
    ``None`` if ``ghdl`` must already be loadable into Yosys some other
    way (e.g. a build of Yosys with the plugin compiled in)."""
    env_value = os.environ.get("GHDL_YOSYS_PLUGIN")
    if env_value:
        return Path(env_value)
    if _SYSTEM_GHDL_PLUGIN_PATH.exists():
        return _SYSTEM_GHDL_PLUGIN_PATH
    return None


def resolve_ghdl_prefix() -> Path | None:
    """Path to pass as ``YosysNetlistBuild(ghdl_prefix=...)`` (GHDL's own
    ``--std`` support-file prefix), or ``None`` to let GHDL use its own
    default. Override-only -- no system-wide fallback needed so far."""
    env_value = os.environ.get("GHDL_YOSYS_PREFIX")
    return Path(env_value) if env_value else None


def resolve_vivado_path() -> Path | None:
    """Path to the ``vivado`` executable to pass as
    ``VivadoProject(vivado_path=...)``, or ``None`` if this machine has no
    Vivado at all.

    ``None`` is the normal, expected result in CI: the ``ru551n/hdl-docker``
    image the ``synthesize`` job runs in ships GHDL/Yosys and no Vivado, and
    ``build_fpga.py --netlist-builds`` there builds *every* registered
    project with no filter. Any ``get_build_projects()`` that returns a
    Vivado project must therefore register it only when this returns a real
    path, or it breaks CI on a machine that was never expected to have the
    tool. Yosys builds stay the unconditional, CI-gating ones; Vivado builds
    are an opt-in local extra for vendor-accurate numbers on the large
    composition entities, which are slow under Yosys.

    ``VIVADO_PATH`` is this project's own override-only env var name, chosen
    for the same reason as ``GHDL_YOSYS_PLUGIN`` above: a human running
    ``python3 build_fpga.py`` in a plain terminal must not need to know
    about any MCP server. The fallbacks cover the two ways Vivado is
    normally reachable -- on ``PATH`` after sourcing ``settings64.sh``, or
    (as on this machine) installed under ``/opt/xilinx`` and not on ``PATH``
    at all, in which case the newest install wins.
    """
    env_value = os.environ.get("VIVADO_PATH")
    if env_value:
        return Path(env_value)

    on_path = shutil.which("vivado")
    if on_path:
        return Path(on_path)

    candidates = sorted(Path("/opt/xilinx").glob("*/Vivado/bin/vivado"))
    return candidates[-1] if candidates else None
