"""Validate the newest or specified Phase 2I run."""

import argparse
from pathlib import Path

from apv_rag.ocr_stress import validate_phase2i_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted((root / "artifacts/phase2i_ocr_stress").glob("run_*"))
        if not candidates:
            print("Phase 2I validation failed: no run folder found")
            return 1
        run_dir = candidates[-1]
    errors = validate_phase2i_run(run_dir)
    if errors:
        print("Phase 2I validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Phase 2I OCR-stress validation passed.")
    print(f"Run folder: {run_dir.resolve()}")
    print("Official development records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
