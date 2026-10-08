"""Pinned AVeriTeC dataset specifications and integrity validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/MichSchli/AVeriTeC"
UPSTREAM_COMMIT = "7c62d1ec8df3fb560d6efe2b85fa191135636f81"
DATASET_DIR_NAME = "official_7c62d1e"
ALLOWED_LABELS = frozenset(
    {"Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"}
)
REQUIRED_FIELDS = frozenset(
    {"claim", "label", "justification", "claim_date", "questions"}
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    count: int
    sha256: str
    expected_empty_claims: int
    expected_duplicate_claim_extras: int

    @property
    def url(self) -> str:
        return (
            "https://raw.githubusercontent.com/MichSchli/AVeriTeC/"
            f"{UPSTREAM_COMMIT}/data/{self.name}.json"
        )


SPLITS = {
    "train": SplitSpec(
        "train",
        3068,
        "ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c",
        1,
        69,
    ),
    "dev": SplitSpec(
        "dev",
        500,
        "499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300",
        0,
        9,
    ),
}


class DatasetIntegrityError(ValueError):
    """Raised when downloaded data violate the frozen specification."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_split(path: Path, spec: SplitSpec) -> dict[str, Any]:
    """Validate hash, JSON shape, count, fields, labels, and unique claims."""

    actual_hash = file_sha256(path)
    if actual_hash != spec.sha256:
        raise DatasetIntegrityError(
            f"{path.name} SHA-256 mismatch: expected {spec.sha256}, got {actual_hash}"
        )
    try:
        records: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetIntegrityError(f"{path.name} is invalid JSON: {exc}") from exc
    if not isinstance(records, list) or len(records) != spec.count:
        raise DatasetIntegrityError(
            f"{path.name} must contain {spec.count} records, got "
            f"{len(records) if isinstance(records, list) else 'non-list'}"
        )

    labels: Counter[str] = Counter()
    claims: list[str] = []
    empty_claims = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DatasetIntegrityError(f"{path.name}[{index}] is not an object")
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise DatasetIntegrityError(f"{path.name}[{index}] missing fields: {sorted(missing)}")
        label = record["label"]
        if label not in ALLOWED_LABELS:
            raise DatasetIntegrityError(f"{path.name}[{index}] has unknown label: {label!r}")
        claim = record["claim"]
        if not isinstance(claim, str) or not claim.strip():
            empty_claims += 1
        else:
            claims.append(claim.strip())
        labels[label] += 1
    duplicate_extras = sum(count - 1 for count in Counter(claims).values() if count > 1)
    if empty_claims != spec.expected_empty_claims:
        raise DatasetIntegrityError(
            f"{path.name} empty-claim count changed: expected "
            f"{spec.expected_empty_claims}, got {empty_claims}"
        )
    if duplicate_extras != spec.expected_duplicate_claim_extras:
        raise DatasetIntegrityError(
            f"{path.name} duplicate-claim count changed: expected "
            f"{spec.expected_duplicate_claim_extras}, got {duplicate_extras}"
        )
    return {
        "labels": dict(sorted(labels.items())),
        "empty_claims": empty_claims,
        "duplicate_claim_extras": duplicate_extras,
    }
