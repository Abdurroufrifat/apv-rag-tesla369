#!/usr/bin/env python3
"""Validate a SciAttr-369 claim CSV from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag import validate_claim_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to a claim-level CSV")
    args = parser.parse_args()

    records, errors = validate_claim_csv(args.csv_path)
    if errors:
        print(f"Validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    canonical_count = len({record.canonical_claim_id for record in records})
    print(
        f"Validation passed: {len(records)} record(s), "
        f"{canonical_count} canonical claim(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

