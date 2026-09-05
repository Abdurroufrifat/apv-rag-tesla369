from pathlib import Path

import pytest

from apv_rag.records import ClaimRecord, validate_claim_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def valid_row() -> dict[str, str]:
    return {
        "claim_id": "TEST-001",
        "canonical_claim_id": "TEST-C001",
        "claim_text": "A sufficiently long test attribution claim.",
        "claimed_person": "Test Person",
        "claim_type": "quotation",
        "language": "en",
        "verdict": "insufficient",
        "verdict_confidence": "0.5",
        "sufficiency": "insufficient",
        "source_urls": "https://example.org/source",
        "provenance_family_ids": "PF-001",
        "review_status": "pilot",
        "annotator_id": "ANN-01",
        "notes": "Test record only.",
    }


def test_starter_template_is_valid() -> None:
    records, errors = validate_claim_csv(PROJECT_ROOT / "data/templates/claims_template.csv")
    assert errors == []
    assert len(records) == 2
    assert len({record.canonical_claim_id for record in records}) == 2


def test_definitive_verdict_requires_sufficient_evidence() -> None:
    row = valid_row()
    row["verdict"] = "authenticated"
    with pytest.raises(ValueError, match="definitive verdict requires sufficient evidence"):
        ClaimRecord.from_mapping(row)


def test_invalid_url_is_rejected() -> None:
    row = valid_row()
    row["source_urls"] = "not-a-url"
    with pytest.raises(ValueError, match="invalid source URL"):
        ClaimRecord.from_mapping(row)


def test_confidence_must_be_bounded() -> None:
    row = valid_row()
    row["verdict_confidence"] = "1.4"
    with pytest.raises(ValueError, match="between 0 and 1"):
        ClaimRecord.from_mapping(row)
