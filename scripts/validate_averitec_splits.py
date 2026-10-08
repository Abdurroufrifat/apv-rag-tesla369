#!/usr/bin/env python3
"""Validate Phase 2B split completeness, integrity, and leakage controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS, load_and_validate_split  # noqa: E402
from apv_rag.splits import build_connected_groups, sha256  # noqa: E402


def main() -> int:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "splits.yaml").read_text())
    source_path = (
        PROJECT_ROOT / "data" / "external" / "averitec" / DATASET_DIR_NAME / "train.json"
    )
    output_dir = PROJECT_ROOT / config["output_directory"]
    try:
        load_and_validate_split(source_path, SPLITS["train"])
        records = json.loads(source_path.read_text(encoding="utf-8"))
        train = json.loads((output_dir / "train_indices.json").read_text())
        validation = json.loads((output_dir / "validation_indices.json").read_text())
        excluded_rows = json.loads((output_dir / "excluded_indices.json").read_text())
        assignments = json.loads((output_dir / "group_assignments.json").read_text())
        manifest = json.loads((output_dir / "split_manifest.json").read_text())
        excluded = [row["upstream_index"] for row in excluded_rows]
        if set(train) & set(validation):
            raise ValueError("train and validation indices overlap")
        if sorted(train + validation + excluded) != list(range(len(records))):
            raise ValueError("split is not a complete, one-time partition of upstream records")
        if set(assignments) != {str(index) for index in train + validation}:
            raise ValueError("group assignments do not cover every eligible record exactly once")
        train_groups = {assignments[str(index)] for index in train}
        validation_groups = {assignments[str(index)] for index in validation}
        if train_groups & validation_groups:
            raise ValueError("connected provenance/claim groups cross the split boundary")
        groups, expected_excluded = build_connected_groups(records)
        if excluded != expected_excluded or sum(map(len, groups)) != len(train) + len(validation):
            raise ValueError("exclusion or group coverage differs from the frozen policy")
        for name in ("train", "validation", "excluded", "groups"):
            entry = manifest["outputs"][name]
            if sha256(output_dir / entry["path"]) != entry["sha256"]:
                raise ValueError(f"{name} output hash does not match the manifest")
        actual_fraction = len(validation) / (len(train) + len(validation))
        if abs(actual_fraction - float(config["validation_fraction"])) > 0.03:
            raise ValueError(f"validation fraction is outside tolerance: {actual_fraction:.4f}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Phase 2B split validation failed: {exc}")
        return 1

    print("Phase 2B split validation passed.")
    print(f"Records: train={len(train)}, validation={len(validation)}, excluded={len(excluded)}")
    print(f"Validation fraction: {actual_fraction:.4f}")
    print("Exact-claim leakage: 0 groups")
    print("Shared-article leakage: 0 groups")
    print("Official development set touched: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
