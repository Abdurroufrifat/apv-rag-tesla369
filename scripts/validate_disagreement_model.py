"""Validate the newest or specified Phase 2E run."""

import argparse
from pathlib import Path

from apv_rag.disagreement_model import validate_phase2e_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted((root / "artifacts/phase2e_disagreement_model").glob("run_*"))
        if not candidates:
            print("Phase 2E validation failed: no run folder found")
            return 1
        run_dir = candidates[-1]
    errors = validate_phase2e_run(run_dir)
    if errors:
        print("Phase 2E validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Phase 2E evidence-disagreement validation passed.")
    print(f"Run folder: {run_dir.resolve()}")
    print("Official development records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
