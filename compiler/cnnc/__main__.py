"""`python -m cnnc ...`: see `cnnc.cli` for the `compile`/`run` subcommands."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
