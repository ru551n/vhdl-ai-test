#!/usr/bin/env python3
"""VUnit run script for the vhdl-ai-test counter design."""

from pathlib import Path

from vunit import VUnit

ROOT = Path(__file__).resolve().parent

vu = VUnit.from_argv()
vu.add_vhdl_builtins()

lib = vu.add_library("lib")
lib.add_source_files(ROOT / "src" / "*.vhd")
lib.add_source_files(ROOT / "src" / "test" / "*.vhd")

vu.main()
