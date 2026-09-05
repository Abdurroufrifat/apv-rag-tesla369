from pathlib import Path

from apv_rag.evidence import validate_evidence_csv, validate_search_plan_csv
from apv_rag.records import validate_claim_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT = PROJECT_ROOT / "data" / "pilot"


def test_phase1a_has_twenty_distinct_claims() -> None:
    records, errors = validate_claim_csv(PILOT / "tesla_claims_pilot_v0_1.csv")
    assert errors == []
    assert len(records) == 20
    assert len({record.canonical_claim_id for record in records}) == 20


def test_evidence_seed_is_valid_and_cross_referenced() -> None:
    claims, _ = validate_claim_csv(PILOT / "tesla_claims_pilot_v0_1.csv")
    evidence, errors = validate_evidence_csv(PILOT / "tesla_evidence_seed_v0_1.csv")
    assert errors == []
    claim_ids = {record.claim_id for record in claims}
    evidence_ids = {record.evidence_id for record in evidence}
    assert len(evidence) >= 10
    assert all(record.claim_id in claim_ids for record in evidence)
    assert all(
        not record.parent_evidence_id or record.parent_evidence_id in evidence_ids
        for record in evidence
    )


def test_every_claim_has_a_frozen_search_plan() -> None:
    claims, _ = validate_claim_csv(PILOT / "tesla_claims_pilot_v0_1.csv")
    searches, errors = validate_search_plan_csv(PILOT / "tesla_search_plan_v0_1.csv")
    assert errors == []
    assert {record["claim_id"] for record in searches} == {
        record.claim_id for record in claims
    }

