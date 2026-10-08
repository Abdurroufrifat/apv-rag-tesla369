#!/usr/bin/env python3
"""Download the pinned official AVeriTeC train and development files."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.averitec import (  # noqa: E402
    DATASET_DIR_NAME,
    SPLITS,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    DatasetIntegrityError,
    file_sha256,
    load_and_validate_split,
)

TARGET_DIR = PROJECT_ROOT / "data" / "external" / "averitec" / DATASET_DIR_NAME


def download_atomic(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "APV-RAG-research/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        os.replace(partial, target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    manifest_splits = {}
    try:
        for split, spec in SPLITS.items():
            target = TARGET_DIR / f"{split}.json"
            if target.exists():
                if file_sha256(target) != spec.sha256:
                    raise DatasetIntegrityError(
                        f"existing {target} has the wrong hash; move it aside and rerun"
                    )
                print(f"Reusing verified {target.relative_to(PROJECT_ROOT)}")
            else:
                print(f"Downloading pinned AVeriTeC {split} split...")
                download_atomic(spec.url, target)
            audit = load_and_validate_split(target, spec)
            manifest_splits[split] = {
                "path": target.name,
                "records": spec.count,
                "sha256": spec.sha256,
                "audit": audit,
                "source_url": spec.url,
            }
    except (DatasetIntegrityError, OSError, TimeoutError) as exc:
        print(f"AVeriTeC download/validation failed: {exc}")
        return 1

    manifest = {
        "dataset": "AVeriTeC",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "license": "CC BY-NC 4.0",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "splits": manifest_splits,
    }
    manifest_path = TARGET_DIR / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("AVeriTeC Phase 2A download and integrity validation passed.")
    print(f"Dataset folder: {TARGET_DIR}")
    print("Records: train=3068, dev=500, total=3568")
    print("Human review required: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
