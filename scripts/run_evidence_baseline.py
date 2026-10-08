#!/usr/bin/env python3
"""Run the frozen Phase 2C evidence baseline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.evidence_baseline import run_baseline  # noqa: E402


def main() -> int:
    try:
        run_dir = run_baseline(PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        print(f"Phase 2C evidence baseline failed: {exc}")
        return 1
    print("Phase 2C evidence baseline passed.")
    print(f"Output folder: {run_dir}")
    print("Official dev records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
