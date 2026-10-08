#!/usr/bin/env python3
"""Validate the Phase 1C blinded bilingual-review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.phase1c import HumanReview, validate_source_capture_csv  # noqa: E402

PILOT = PROJECT_ROOT / "data" / "pilot"
TEMPLATES = PROJECT_ROOT / "data" / "templates"
SOURCE_REGISTER = PILOT / "tesla_phase1c_source_register_v0_1.csv"
REVIEW_TEMPLATE = TEMPLATES / "t369004_bilingual_review_template_v0_1.csv"
MANIFEST = PILOT / "phase1c_manifest_v0_1.sha256"
EXPECTED_STATIC_PATHS = {
    "README.md",
    "data/README.md",
    "data/pilot/tesla_phase1c_source_register_v0_1.csv",
    "data/templates/t369004_bilingual_review_template_v0_1.csv",
    "docs/DECISION_LOG.md",
    "docs/PHASE_1C_COORDINATOR_GUIDE.md",
    "docs/PHASE_1C_INSTALL.md",
    "docs/PHASE_1C_REVIEWER_GUIDE.md",
    "schemas/human_review.schema.json",
    "scripts/create_phase1c_packets.py",
    "scripts/validate_phase1c.py",
    "src/apv_rag/phase1c.py",
    "tests/test_phase1c.py",
}
BLIND_REQUIRED_BLANKS = {
    "reviewer_code",
    "bilingual_qualification",
    "independent_review_confirmed",
    "review_date",
    "serbian_transcription",
    "literal_translation",
    "normalized_english",
    "translation_equivalence",
    "verdict",
    "sufficiency",
    "confidence",
    "rationale",
    "missing_evidence",
    "limitations",
    "frozen_at_utc",
}
FORBIDDEN_BLIND_MARKERS = (
    "machine_assisted_provisional",
    "tesla_phase1b_verdicts",
    "provisional_verdict",
    "verdict_confidence",
    "0.90",
)


def _validate_template() -> list[str]:
    errors: list[str] = []
    if not REVIEW_TEMPLATE.is_file():
        return [f"file not found: {REVIEW_TEMPLATE}"]
    with REVIEW_TEMPLATE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(HumanReview.REQUIRED_FIELDS):
            errors.append("Phase 1C template header does not match HumanReview fields")
        rows = list(reader)
    if len(rows) != 1:
        errors.append("Phase 1C template must contain exactly one assignment row")
        return errors
    row = {key: (value or "").strip() for key, value in rows[0].items()}
    if row["claim_id"] != "T369-004":
        errors.append("blinded template must target T369-004")
    if row["printed_page"] != "2" or row["column"] != "1":
        errors.append("blinded template must preserve printed page 2, column 1")
    if row["decisive_evidence_ids"] != "EVID-112":
        errors.append("blinded template must identify EVID-112")
    if row["provenance_family_ids"] != "PF-POLITIKA-1927":
        errors.append("blinded template has an unexpected provenance family")
    for field in sorted(BLIND_REQUIRED_BLANKS):
        if row.get(field):
            errors.append(f"blinded template field must remain blank: {field}")
    if row.get("frozen", "").lower() != "false":
        errors.append("blinded template must start with frozen=false")
    if row.get("protocol_version") != "0.2":
        errors.append("blinded template must use protocol version 0.2")
    return errors


def _validate_blinding() -> list[str]:
    errors: list[str] = []
    paths = [
        REVIEW_TEMPLATE,
        PROJECT_ROOT / "docs" / "PHASE_1C_REVIEWER_GUIDE.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())
    lowered = combined.lower()
    for marker in FORBIDDEN_BLIND_MARKERS:
        if marker.lower() in lowered:
            errors.append(f"blinded reviewer material leaks forbidden marker: {marker}")
    return errors


def _validate_manifest() -> list[str]:
    errors: list[str] = []
    entries: dict[str, str] = {}
    if not MANIFEST.is_file():
        return [f"file not found: {MANIFEST}"]
    for line_number, raw_line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            errors.append(f"manifest line {line_number} is malformed")
            continue
        entries[parts[1]] = parts[0]
    if set(entries) != EXPECTED_STATIC_PATHS:
        missing = sorted(EXPECTED_STATIC_PATHS - set(entries))
        extra = sorted(set(entries) - EXPECTED_STATIC_PATHS)
        if missing:
            errors.append(f"Phase 1C manifest is missing: {', '.join(missing)}")
        if extra:
            errors.append(f"Phase 1C manifest has unexpected paths: {', '.join(extra)}")
    for relative_path, expected_hash in entries.items():
        target = PROJECT_ROOT / relative_path
        if not target.is_file():
            errors.append(f"manifest target is missing: {relative_path}")
            continue
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"manifest hash mismatch: {relative_path}")
    return errors


def collect_phase1c_errors(
    require_private_capture: bool = False,
) -> tuple[dict[str, object], list[str]]:
    """Run static, blinding, and optional private-capture integrity checks."""

    captures, errors = validate_source_capture_csv(SOURCE_REGISTER)
    errors.extend(_validate_template())
    errors.extend(_validate_blinding())
    errors.extend(_validate_manifest())

    schema_path = PROJECT_ROOT / "schemas" / "human_review.schema.json"
    if not schema_path.is_file():
        errors.append(f"file not found: {schema_path}")
    else:
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            errors.append("human-review schema must be a closed object schema")

    gitignore_path = PROJECT_ROOT / ".gitignore"
    gitignore = gitignore_path.read_text(encoding="utf-8") if gitignore_path.is_file() else ""
    if "data/raw/*" not in gitignore:
        errors.append(".gitignore must protect private raw evidence")
    if "artifacts/" not in gitignore:
        errors.append(".gitignore must protect generated reviewer packets")

    private_status = "not_present_optional"
    for capture in captures:
        target = PROJECT_ROOT / Path(capture.private_relative_path)
        if not target.is_file():
            if require_private_capture:
                errors.append(f"private capture is missing: {capture.private_relative_path}")
            continue
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != capture.sha256:
            errors.append(f"private capture hash mismatch: {capture.private_relative_path}")
            private_status = "hash_mismatch"
        else:
            private_status = "verified"

    counts: dict[str, object] = {
        "captures": len(captures),
        "claim_id": captures[0].claim_id if captures else "unknown",
        "private_status": private_status,
    }
    return counts, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-private-capture",
        action="store_true",
        help="fail unless the registered private screenshot exists and matches its hash",
    )
    args = parser.parse_args()
    counts, errors = collect_phase1c_errors(args.require_private_capture)
    if errors:
        print(f"Phase 1C validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Phase 1C package validation passed.")
    print(f"Source capture records: {counts['captures']} ({counts['claim_id']})")
    print(f"Private capture integrity: {str(counts['private_status']).replace('_', ' ')}")
    print("Blinded reviewer-template leakage check: passed")
    print("Scientific status: no human label or gold verdict has been assigned.")
    print("Workflow status: Phase 1C is archived and human review is not required.")
    print("Next requirement: validate the Phase 2 machine-only benchmark protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
