#!/usr/bin/env python3
"""Validate a saved Phase 2C evidence-baseline run."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.evidence_baseline import validate_run  # noqa: E402


def main() -> int:
    output_root = PROJECT_ROOT / "artifacts" / "phase2c_evidence_baseline"
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1]).resolve()
    else:
        runs = sorted(path for path in output_root.glob("run_*") if path.is_dir())
        if not runs:
            print("Phase 2C validation failed: no run directory found.")
            return 1
        run_dir = runs[-1]
    errors = validate_run(run_dir)
    if errors:
        print("Phase 2C validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Phase 2C evidence baseline validation passed.")
    print(f"Run folder: {run_dir}")
    print("Official dev records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
