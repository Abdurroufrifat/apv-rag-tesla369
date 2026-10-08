"""Validate the newest or specified Phase 2G run."""

import argparse
from pathlib import Path

from apv_rag.abstention_model import validate_phase2g_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted((root / "artifacts/phase2g_abstention_model").glob("run_*"))
        if not candidates:
            print("Phase 2G validation failed: no run folder found")
            return 1
        run_dir = candidates[-1]
    errors = validate_phase2g_run(run_dir)
    if errors:
        print("Phase 2G validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Phase 2G calibrated-abstention validation passed.")
    print(f"Run folder: {run_dir.resolve()}")
    print("Official development records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
