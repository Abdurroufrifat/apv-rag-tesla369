import hashlib
import json
from pathlib import Path

import pytest

from apv_rag.averitec import DatasetIntegrityError, SplitSpec, load_and_validate_split


def write_fixture(path: Path, records: list[dict]) -> SplitSpec:
    path.write_text(json.dumps(records), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    claims = [
        record.get("claim", "").strip() for record in records if record.get("claim", "").strip()
    ]
    duplicate_extras = len(claims) - len(set(claims))
    empty_claims = len(records) - len(claims)
    return SplitSpec(path.stem, len(records), digest, empty_claims, duplicate_extras)


def valid_record(claim: str = "A test claim") -> dict:
    return {
        "claim": claim,
        "label": "Supported",
        "justification": "Test-only evidence supports the claim.",
        "claim_date": "2023-01-01",
        "questions": [],
    }


def test_valid_split_fixture(tmp_path: Path):
    path = tmp_path / "fixture.json"
    spec = write_fixture(path, [valid_record()])
    assert load_and_validate_split(path, spec) == {
        "labels": {"Supported": 1},
        "empty_claims": 0,
        "duplicate_claim_extras": 0,
    }


def test_hash_mismatch_is_rejected(tmp_path: Path):
    path = tmp_path / "fixture.json"
    path.write_text("[]", encoding="utf-8")
    spec = SplitSpec("fixture", 0, "0" * 64, 0, 0)
    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        load_and_validate_split(path, spec)


def test_unknown_label_is_rejected(tmp_path: Path):
    path = tmp_path / "fixture.json"
    record = valid_record()
    record["label"] = "machine_candidate"
    spec = write_fixture(path, [record])
    with pytest.raises(DatasetIntegrityError, match="unknown label"):
        load_and_validate_split(path, spec)


def test_duplicate_claim_is_audited(tmp_path: Path):
    path = tmp_path / "fixture.json"
    spec = write_fixture(path, [valid_record(), valid_record()])
    assert load_and_validate_split(path, spec)["duplicate_claim_extras"] == 1
