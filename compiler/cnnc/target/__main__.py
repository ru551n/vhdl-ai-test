"""`python -m cnnc.target dump <name>`: print a target's JSON to stdout."""

from __future__ import annotations

import sys

from .contract import TargetError
from .load import load_target


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "dump":
        print("usage: python -m cnnc.target dump <target-name>", file=sys.stderr)
        return 2
    try:
        target = load_target(argv[1])
    except TargetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(target.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
