#!/usr/bin/env python3
"""Validate Phase 1B evidence reconnaissance and provisional pilot labels."""

from __future__ import annotations

import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.evidence import validate_evidence_csv, validate_search_plan_csv  # noqa: E402
from apv_rag.phase1b import (  # noqa: E402
    split_ids,
    validate_provenance_edge_csv,
    validate_provisional_verdict_csv,
    validate_search_execution_csv,
)
from apv_rag.records import validate_claim_csv  # noqa: E402

TARGET_CLAIMS = {f"T369-{number:03d}" for number in range(1, 6)}
NEGATIVE_STATEMENT = "No authenticating primary source was located under protocol version 0.1"


def collect_phase1b_errors() -> tuple[dict[str, int], list[str]]:
    """Run file-level and cross-file checks used by both the CLI and tests."""

    pilot = PROJECT_ROOT / "data" / "pilot"
    claims, claim_errors = validate_claim_csv(pilot / "tesla_claims_pilot_v0_1.csv")
    evidence, evidence_errors = validate_evidence_csv(
        pilot / "tesla_phase1b_evidence_v0_1.csv"
    )
    search_plan, plan_errors = validate_search_plan_csv(
        pilot / "tesla_search_plan_v0_1.csv"
    )
    executions, execution_errors = validate_search_execution_csv(
        pilot / "tesla_phase1b_search_log_v0_1.csv"
    )
    verdicts, verdict_errors = validate_provisional_verdict_csv(
        pilot / "tesla_phase1b_verdicts_v0_1.csv"
    )
    edges, edge_errors = validate_provenance_edge_csv(
        pilot / "tesla_phase1b_provenance_edges_v0_1.csv"
    )
    errors = [
        *claim_errors,
        *evidence_errors,
        *plan_errors,
        *execution_errors,
        *verdict_errors,
        *edge_errors,
    ]

    master_claim_ids = {claim.claim_id for claim in claims}
    if not TARGET_CLAIMS <= master_claim_ids:
        errors.append("the Phase 1A claim table is missing one or more Phase 1B target claims")

    evidence_ids = {record.evidence_id for record in evidence}
    evidence_by_id = {record.evidence_id: record for record in evidence}
    if {record.claim_id for record in evidence} != TARGET_CLAIMS:
        errors.append("Phase 1B evidence must cover exactly T369-001 through T369-005")
    for record in evidence:
        if record.parent_evidence_id and record.parent_evidence_id not in evidence_ids:
            errors.append(
                f"{record.evidence_id} references unknown parent {record.parent_evidence_id}"
            )

    plan_map = {record["search_id"]: record["claim_id"] for record in search_plan}
    execution_claims: dict[str, list] = defaultdict(list)
    for record in executions:
        execution_claims[record.claim_id].append(record)
        if record.search_id not in plan_map:
            errors.append(f"{record.execution_id} references unknown plan {record.search_id}")
        elif plan_map[record.search_id] != record.claim_id:
            errors.append(f"{record.execution_id} claim does not match its frozen search plan")
        for evidence_id in split_ids(record.evidence_ids):
            if evidence_id not in evidence_ids:
                errors.append(f"{record.execution_id} references unknown evidence {evidence_id}")

    if set(execution_claims) != TARGET_CLAIMS:
        errors.append("executed-search log must cover exactly T369-001 through T369-005")
    for claim_id in sorted(TARGET_CLAIMS):
        rows = execution_claims.get(claim_id, [])
        tier_a_repositories = {
            row.repository_or_index for row in rows if row.source_tier == "A"
        }
        if len(tier_a_repositories) < 2:
            errors.append(f"{claim_id} needs searches targeting two distinct Tier-A sources")
        variants = {row.query_variant for row in rows}
        if not {"exact", "fragment"} <= variants:
            errors.append(f"{claim_id} needs both exact and fragment query variants")
        if not any(row.source_tier == "B" for row in rows):
            errors.append(f"{claim_id} needs at least one scholarly-index search")
        if not any(row.stage == "upstream_trace" for row in rows):
            errors.append(f"{claim_id} needs an upstream-reference trace")

    verdict_by_claim = {record.claim_id: record for record in verdicts}
    if set(verdict_by_claim) != TARGET_CLAIMS:
        errors.append("provisional verdicts must cover exactly T369-001 through T369-005")
    for claim_id, verdict in verdict_by_claim.items():
        decisive_ids = split_ids(verdict.decisive_evidence_ids)
        if not decisive_ids:
            errors.append(f"{claim_id} must identify decisive evidence")
        for evidence_id in decisive_ids:
            if evidence_id not in evidence_by_id:
                errors.append(f"{claim_id} references unknown decisive evidence {evidence_id}")
            elif evidence_by_id[evidence_id].claim_id != claim_id:
                errors.append(
                    f"{claim_id} decisive evidence {evidence_id} belongs to another claim"
                )
        if verdict.provisional_verdict == "insufficient":
            if NEGATIVE_STATEMENT not in verdict.rationale:
                errors.append(f"{claim_id} must use the protocol's bounded negative statement")
        else:
            decisive_records = [
                evidence_by_id[item] for item in decisive_ids if item in evidence_by_id
            ]
            if not any(
                item.stance == "supports" and item.source_rank <= 2
                for item in decisive_records
            ):
                errors.append(
                    f"{claim_id} definitive verdict lacks supporting rank-1/2 evidence"
                )

    for edge in edges:
        if edge.child_evidence_id not in evidence_ids:
            errors.append(f"{edge.edge_id} has unknown child {edge.child_evidence_id}")
        if edge.parent_evidence_id not in evidence_ids:
            errors.append(f"{edge.edge_id} has unknown parent {edge.parent_evidence_id}")

    manifest_path = pilot / "phase1b_manifest_v0_1.sha256"
    expected_manifest_paths = {
        "data/pilot/tesla_phase1b_evidence_v0_1.csv",
        "data/pilot/tesla_phase1b_search_log_v0_1.csv",
        "data/pilot/tesla_phase1b_verdicts_v0_1.csv",
        "data/pilot/tesla_phase1b_provenance_edges_v0_1.csv",
    }
    manifest_entries: dict[str, str] = {}
    if not manifest_path.is_file():
        errors.append(f"file not found: {manifest_path}")
    else:
        for line_number, raw_line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                errors.append(f"manifest line {line_number} is malformed")
                continue
            manifest_entries[parts[1]] = parts[0]
        if set(manifest_entries) != expected_manifest_paths:
            errors.append("Phase 1B manifest paths do not match the four pilot tables")
        for relative_path, expected_hash in manifest_entries.items():
            target = PROJECT_ROOT / relative_path
            if not target.is_file():
                errors.append(f"manifest target is missing: {relative_path}")
                continue
            actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                errors.append(f"manifest hash mismatch: {relative_path}")

    counts = {
        "claims": len(TARGET_CLAIMS),
        "evidence": len(evidence),
        "families": len({record.provenance_family_id for record in evidence}),
        "searches": len(executions),
        "edges": len(edges),
        "authenticated": sum(
            record.provisional_verdict == "authenticated" for record in verdicts
        ),
        "insufficient": sum(
            record.provisional_verdict == "insufficient" for record in verdicts
        ),
    }
    return counts, errors


def main() -> int:
    counts, errors = collect_phase1b_errors()
    if errors:
        print(f"Phase 1B validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Phase 1B validation passed.")
    print(f"Pilot claims: {counts['claims']}")
    print(
        f"Evidence records: {counts['evidence']} across {counts['families']} provenance families"
    )
    print(f"Executed searches: {counts['searches']}")
    print(f"Provenance edges: {counts['edges']}")
    print(
        "Provisional outcomes: "
        f"{counts['authenticated']} authenticated; {counts['insufficient']} insufficient"
    )
    print("Scientific status: machine-assisted pre-annotations only; no gold labels assigned.")
    print("Human review required for every claim; bilingual review required for T369-004.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
