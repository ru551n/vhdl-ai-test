"""pytest configuration: make `cnnc` importable from the repo checkout."""

import sys
from pathlib import Path

COMPILER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPILER_ROOT.parent

if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))
