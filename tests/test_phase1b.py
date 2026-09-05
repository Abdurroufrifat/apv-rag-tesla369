import json
from collections import defaultdict
from pathlib import Path

from apv_rag.evidence import validate_evidence_csv
from apv_rag.phase1b import (
    split_ids,
    validate_provenance_edge_csv,
    validate_provisional_verdict_csv,
    validate_search_execution_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT = PROJECT_ROOT / "data" / "pilot"


def test_phase1b_tables_are_valid_and_have_expected_scope() -> None:
    evidence, evidence_errors = validate_evidence_csv(
        PILOT / "tesla_phase1b_evidence_v0_1.csv"
    )
    searches, search_errors = validate_search_execution_csv(
        PILOT / "tesla_phase1b_search_log_v0_1.csv"
    )
    verdicts, verdict_errors = validate_provisional_verdict_csv(
        PILOT / "tesla_phase1b_verdicts_v0_1.csv"
    )
    edges, edge_errors = validate_provenance_edge_csv(
        PILOT / "tesla_phase1b_provenance_edges_v0_1.csv"
    )
    assert evidence_errors == []
    assert search_errors == []
    assert verdict_errors == []
    assert edge_errors == []
    assert len(evidence) == 20
    assert len(searches) == 35
    assert len(verdicts) == 5
    assert len(edges) == 11


def test_every_pilot_claim_has_required_search_coverage() -> None:
    searches, _ = validate_search_execution_csv(PILOT / "tesla_phase1b_search_log_v0_1.csv")
    by_claim: dict[str, list] = defaultdict(list)
    for record in searches:
        by_claim[record.claim_id].append(record)
    assert set(by_claim) == {f"T369-{number:03d}" for number in range(1, 6)}
    for records in by_claim.values():
        assert len({row.repository_or_index for row in records if row.source_tier == "A"}) >= 2
        assert {"exact", "fragment"} <= {row.query_variant for row in records}
        assert any(row.source_tier == "B" for row in records)
        assert any(row.stage == "upstream_trace" for row in records)


def test_verdicts_remain_provisional_and_require_human_review() -> None:
    verdicts, _ = validate_provisional_verdict_csv(
        PILOT / "tesla_phase1b_verdicts_v0_1.csv"
    )
    assert all(record.label_status == "machine_assisted_provisional" for record in verdicts)
    assert all(record.requires_human_review for record in verdicts)
    assert sum(record.provisional_verdict == "authenticated" for record in verdicts) == 1
    assert sum(record.provisional_verdict == "insufficient" for record in verdicts) == 4


def test_authenticated_claim_has_primary_support_and_translation_caveat() -> None:
    evidence, _ = validate_evidence_csv(PILOT / "tesla_phase1b_evidence_v0_1.csv")
    verdicts, _ = validate_provisional_verdict_csv(
        PILOT / "tesla_phase1b_verdicts_v0_1.csv"
    )
    evidence_by_id = {record.evidence_id: record for record in evidence}
    verdict = next(record for record in verdicts if record.claim_id == "T369-004")
    decisive = [evidence_by_id[item] for item in split_ids(verdict.decisive_evidence_ids)]
    assert any(item.is_primary and item.source_rank == 1 for item in decisive)
    assert any(item.stance == "supports" for item in decisive)
    assert "translation" in verdict.rationale.lower()
    assert "bilingual" in verdict.next_human_action.lower()


def test_provenance_edges_are_closed_and_capture_citation_mismatches() -> None:
    evidence, _ = validate_evidence_csv(PILOT / "tesla_phase1b_evidence_v0_1.csv")
    edges, _ = validate_provenance_edge_csv(
        PILOT / "tesla_phase1b_provenance_edges_v0_1.csv"
    )
    evidence_ids = {record.evidence_id for record in evidence}
    assert all(edge.child_evidence_id in evidence_ids for edge in edges)
    assert all(edge.parent_evidence_id in evidence_ids for edge in edges)
    assert sum(edge.relation == "citation_mismatch" for edge in edges) == 2


def test_phase1b_json_schemas_are_parseable() -> None:
    schema_names = (
        "search_execution.schema.json",
        "provisional_verdict.schema.json",
        "provenance_edge.schema.json",
    )
    for name in schema_names:
        with (PROJECT_ROOT / "schemas" / name).open(encoding="utf-8") as handle:
            schema = json.load(handle)
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["type"] == "object"
