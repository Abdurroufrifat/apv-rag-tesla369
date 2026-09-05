#!/usr/bin/env python3
"""Validate all Phase 1A Tesla pilot files and their cross-references."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.evidence import validate_evidence_csv, validate_search_plan_csv  # noqa: E402
from apv_rag.records import validate_claim_csv  # noqa: E402


def main() -> int:
    pilot = PROJECT_ROOT / "data" / "pilot"
    claim_path = pilot / "tesla_claims_pilot_v0_1.csv"
    evidence_path = pilot / "tesla_evidence_seed_v0_1.csv"
    search_path = pilot / "tesla_search_plan_v0_1.csv"

    claims, claim_errors = validate_claim_csv(claim_path)
    evidence, evidence_errors = validate_evidence_csv(evidence_path)
    searches, search_errors = validate_search_plan_csv(search_path)
    errors = [*claim_errors, *evidence_errors, *search_errors]

    claim_ids = {claim.claim_id for claim in claims}
    canonical_ids = {claim.canonical_claim_id for claim in claims}
    if len(claims) != 20:
        errors.append(f"pilot must contain exactly 20 claims; found {len(claims)}")
    if len(canonical_ids) != 20:
        errors.append("each pilot row must represent a distinct canonical claim")
    if any(claim.claimed_person != "Nikola Tesla" for claim in claims):
        errors.append("Phase 1A pilot claims must all target Nikola Tesla")
    if any(claim.review_status != "candidate" for claim in claims):
        errors.append("unreviewed pilot claims must retain candidate status")

    for item in evidence:
        if item.claim_id not in claim_ids:
            errors.append(f"{item.evidence_id} references unknown claim {item.claim_id}")
    evidence_ids = {item.evidence_id for item in evidence}
    for item in evidence:
        if item.parent_evidence_id and item.parent_evidence_id not in evidence_ids:
            errors.append(
                f"{item.evidence_id} references unknown parent evidence {item.parent_evidence_id}"
            )

    planned_claim_ids = {item["claim_id"] for item in searches}
    missing_plans = sorted(claim_ids - planned_claim_ids)
    unknown_plans = sorted(planned_claim_ids - claim_ids)
    if missing_plans:
        errors.append(f"claims without search plans: {', '.join(missing_plans)}")
    if unknown_plans:
        errors.append(f"search plans reference unknown claims: {', '.join(unknown_plans)}")

    if errors:
        print(f"Phase 1A validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    family_count = len({item.provenance_family_id for item in evidence})
    print("Phase 1A validation passed.")
    print(f"Claims: {len(claims)} distinct canonical candidates")
    print(f"Evidence seeds: {len(evidence)} across {family_count} provenance families")
    print(f"Frozen search plans: {len(searches)}")
    print("Scientific status: candidate records only; no gold verdicts assigned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

