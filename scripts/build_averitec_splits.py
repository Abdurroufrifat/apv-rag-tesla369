#!/usr/bin/env python3
"""Build the frozen Phase 2B AVeriTeC internal split."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS, load_and_validate_split  # noqa: E402
from apv_rag.splits import make_grouped_split, sha256, write_json_atomic  # noqa: E402


def label_counts(records, indices):
    return dict(sorted(Counter(records[index]["label"] for index in indices).items()))


def main() -> int:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "splits.yaml").read_text())
    source_dir = PROJECT_ROOT / "data" / "external" / "averitec" / DATASET_DIR_NAME
    source_path = source_dir / "train.json"
    try:
        source_audit = load_and_validate_split(source_path, SPLITS["train"])
        records = json.loads(source_path.read_text(encoding="utf-8"))
        result = make_grouped_split(
            records,
            validation_fraction=float(config["validation_fraction"]),
            seed=int(config["random_seed"]),
            trials=int(config["random_search_trials"]),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Phase 2B split creation failed: {exc}")
        return 1

    output_dir = PROJECT_ROOT / config["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train_indices.json",
        "validation": output_dir / "validation_indices.json",
        "excluded": output_dir / "excluded_indices.json",
        "groups": output_dir / "group_assignments.json",
    }
    write_json_atomic(paths["train"], result.train_indices)
    write_json_atomic(paths["validation"], result.validation_indices)
    write_json_atomic(
        paths["excluded"],
        [{"upstream_index": index, "reason": "empty_claim"} for index in result.excluded_indices],
    )
    write_json_atomic(
        paths["groups"],
        {str(index): group for index, group in sorted(result.group_by_index.items())},
    )
    train_groups = {result.group_by_index[index] for index in result.train_indices}
    validation_groups = {result.group_by_index[index] for index in result.validation_indices}
    manifest = {
        "protocol_version": config["protocol_version"],
        "source": {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "sha256": SPLITS["train"].sha256,
            "audit": source_audit,
        },
        "policy": {
            "random_seed": config["random_seed"],
            "validation_fraction_target": config["validation_fraction"],
            "random_search_trials": config["random_search_trials"],
            "grouping_keys": config["grouping_keys"],
            "official_dev_policy": config["official_dev_policy"],
        },
        "counts": {
            "train": len(result.train_indices),
            "validation": len(result.validation_indices),
            "excluded": len(result.excluded_indices),
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
        },
        "labels": {
            "train": label_counts(records, result.train_indices),
            "validation": label_counts(records, result.validation_indices),
        },
        "outputs": {
            name: {"path": path.name, "sha256": sha256(path)} for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print("Phase 2B leakage-safe split creation passed.")
    print(f"Output folder: {output_dir}")
    print(
        f"Records: train={len(result.train_indices)}, "
        f"validation={len(result.validation_indices)}, excluded={len(result.excluded_indices)}"
    )
    print(f"Groups: train={len(train_groups)}, validation={len(validation_groups)}")
    print("Official dev records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
