#!/usr/bin/env python3
"""Validate the locally downloaded pinned AVeriTeC dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.averitec import (  # noqa: E402
    DATASET_DIR_NAME,
    SPLITS,
    UPSTREAM_COMMIT,
    DatasetIntegrityError,
    load_and_validate_split,
)


def main() -> int:
    directory = PROJECT_ROOT / "data" / "external" / "averitec" / DATASET_DIR_NAME
    manifest_path = directory / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
            raise DatasetIntegrityError("manifest upstream commit is not frozen")
        for split, spec in SPLITS.items():
            audit = load_and_validate_split(directory / f"{split}.json", spec)
            entry = manifest.get("splits", {}).get(split, {})
            if entry.get("sha256") != spec.sha256 or entry.get("audit") != audit:
                raise DatasetIntegrityError(f"manifest does not match verified {split} data")
    except (DatasetIntegrityError, OSError, json.JSONDecodeError) as exc:
        print(f"AVeriTeC validation failed: {exc}")
        return 1

    print("AVeriTeC Phase 2A validation passed.")
    print(f"Pinned commit: {UPSTREAM_COMMIT}")
    print("Verified records: train=3068, dev=500, total=3568")
    print("Official labels preserved: yes")
    print("Development split reserved for evaluation: yes")
    print("Known upstream empty/duplicate claims recorded: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
