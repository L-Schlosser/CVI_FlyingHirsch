from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src() -> None:
    """
    Ensure `src/` is on sys.path so `python -m scripts.*` works without editable installs.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

