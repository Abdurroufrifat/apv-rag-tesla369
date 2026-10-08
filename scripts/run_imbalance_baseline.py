#!/usr/bin/env python3
"""Run the frozen Phase 2D imbalance-aware evidence baseline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.imbalance_baseline import run_phase2d  # noqa: E402


def main() -> int:
    try:
        run_dir = run_phase2d(PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        print(f"Phase 2D failed: {exc}")
        return 1
    print("Phase 2D imbalance-aware baseline passed.")
    print(f"Output folder: {run_dir}")
    print("Official dev records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
