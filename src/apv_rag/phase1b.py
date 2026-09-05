"""Typed records and validation for the Phase 1B archival-verification pilot."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import ClassVar


def _required_columns(fieldnames: list[str] | None, required: tuple[str, ...]) -> list[str]:
    if fieldnames is None:
        return list(required)
    return [field for field in required if field not in fieldnames]


def _valid_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _valid_iso_datetime(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def split_ids(value: str) -> list[str]:
    """Split the project's semicolon-delimited identifier fields."""

    return [item.strip() for item in value.split(";") if item.strip()]


@dataclass(frozen=True)
class SearchExecution:
    execution_id: str
    search_id: str
    claim_id: str
    stage: str
    source_tier: str
    repository_or_index: str
    search_interface: str
    query_variant: str
    query_text: str
    executed_at_utc: str
    result_status: str
    authenticating_result: bool
    evidence_ids: str
    earliest_located_date: str
    limitations: str
    executor_id: str

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "execution_id",
        "search_id",
        "claim_id",
        "stage",
        "source_tier",
        "repository_or_index",
        "search_interface",
        "query_variant",
        "query_text",
        "executed_at_utc",
        "result_status",
        "authenticating_result",
        "evidence_ids",
        "earliest_located_date",
        "limitations",
        "executor_id",
    )
    STAGES: ClassVar[frozenset[str]] = frozenset(
        {
            "tier_a_primary",
            "tier_b_scholarly",
            "tier_c_curated",
            "tier_d_propagation",
            "upstream_trace",
        }
    )
    INTERFACES: ClassVar[frozenset[str]] = frozenset(
        {
            "archive_ui",
            "direct_document_search",
            "web_domain_query",
            "scholarly_index_query",
            "backward_citation_trace",
            "scan_viewer",
        }
    )
    QUERY_VARIANTS: ClassVar[frozenset[str]] = frozenset(
        {"exact", "fragment", "concept", "translation", "source_specific", "upstream"}
    )
    RESULT_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {
            "authenticating_source_located",
            "occurrence_only",
            "contextual_result",
            "no_authenticating_result",
            "blocked",
        }
    )

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> SearchExecution:
        values = {field: (row.get(field) or "").strip() for field in cls.REQUIRED_FIELDS}
        if not re.fullmatch(r"RUN-[0-9]{3,}", values["execution_id"]):
            raise ValueError("execution_id must resemble RUN-001")
        if not re.fullmatch(r"SEARCH-[0-9]{3,}", values["search_id"]):
            raise ValueError("search_id must resemble SEARCH-001")
        if not re.fullmatch(r"T369-[0-9]{3,}", values["claim_id"]):
            raise ValueError("claim_id must resemble T369-001")
        if values["stage"] not in cls.STAGES:
            raise ValueError(f"invalid search stage: {values['stage']!r}")
        if values["source_tier"] not in {"A", "B", "C", "D"}:
            raise ValueError("source_tier must be A, B, C, or D")
        if not values["repository_or_index"]:
            raise ValueError("repository_or_index cannot be empty")
        if values["search_interface"] not in cls.INTERFACES:
            raise ValueError(f"invalid search_interface: {values['search_interface']!r}")
        if values["query_variant"] not in cls.QUERY_VARIANTS:
            raise ValueError(f"invalid query_variant: {values['query_variant']!r}")
        if not values["query_text"]:
            raise ValueError("query_text cannot be empty")
        if not _valid_iso_datetime(values["executed_at_utc"]):
            raise ValueError("executed_at_utc must be a timezone-aware ISO datetime")
        if values["result_status"] not in cls.RESULT_STATUSES:
            raise ValueError(f"invalid result_status: {values['result_status']!r}")
        bool_text = values["authenticating_result"].lower()
        if bool_text not in {"true", "false"}:
            raise ValueError("authenticating_result must be true or false")
        authenticating_result = bool_text == "true"
        expected = values["result_status"] == "authenticating_source_located"
        if authenticating_result != expected:
            raise ValueError("authenticating_result conflicts with result_status")
        for evidence_id in split_ids(values["evidence_ids"]):
            if not re.fullmatch(r"EVID-[0-9]{3,}", evidence_id):
                raise ValueError(f"invalid evidence ID: {evidence_id!r}")
        if values["earliest_located_date"] and not re.fullmatch(
            r"[0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?", values["earliest_located_date"]
        ):
            raise ValueError("earliest_located_date must use YYYY, YYYY-MM, or YYYY-MM-DD")
        if not values["limitations"]:
            raise ValueError("limitations cannot be empty")
        if not values["executor_id"]:
            raise ValueError("executor_id cannot be empty")
        parsed_values = dict(values)
        parsed_values["authenticating_result"] = authenticating_result
        return cls(**parsed_values)


@dataclass(frozen=True)
class ProvisionalVerdict:
    claim_id: str
    provisional_verdict: str
    verdict_confidence: float
    sufficiency: str
    label_status: str
    requires_human_review: bool
    decisive_evidence_ids: str
    provenance_family_ids: str
    rationale: str
    missing_evidence: str
    next_human_action: str
    annotator_id: str
    annotation_date: str
    protocol_version: str

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "claim_id",
        "provisional_verdict",
        "verdict_confidence",
        "sufficiency",
        "label_status",
        "requires_human_review",
        "decisive_evidence_ids",
        "provenance_family_ids",
        "rationale",
        "missing_evidence",
        "next_human_action",
        "annotator_id",
        "annotation_date",
        "protocol_version",
    )
    VERDICTS: ClassVar[frozenset[str]] = frozenset(
        {"authenticated", "misattributed", "contradicted", "insufficient"}
    )

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> ProvisionalVerdict:
        values = {field: (row.get(field) or "").strip() for field in cls.REQUIRED_FIELDS}
        if not re.fullmatch(r"T369-[0-9]{3,}", values["claim_id"]):
            raise ValueError("claim_id must resemble T369-001")
        if values["provisional_verdict"] not in cls.VERDICTS:
            raise ValueError(f"invalid provisional_verdict: {values['provisional_verdict']!r}")
        try:
            confidence = float(values["verdict_confidence"])
        except ValueError as exc:
            raise ValueError("verdict_confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("verdict_confidence must be between 0 and 1")
        if values["sufficiency"] not in {"sufficient", "insufficient"}:
            raise ValueError("sufficiency must be sufficient or insufficient")
        if values["provisional_verdict"] == "insufficient":
            if values["sufficiency"] != "insufficient":
                raise ValueError("an insufficient verdict must have insufficient evidence")
        elif values["sufficiency"] != "sufficient":
            raise ValueError("a definitive verdict requires sufficient evidence")
        if values["label_status"] != "machine_assisted_provisional":
            raise ValueError("Phase 1B labels must remain machine_assisted_provisional")
        bool_text = values["requires_human_review"].lower()
        if bool_text != "true":
            raise ValueError("every Phase 1B verdict must require human review")
        for evidence_id in split_ids(values["decisive_evidence_ids"]):
            if not re.fullmatch(r"EVID-[0-9]{3,}", evidence_id):
                raise ValueError(f"invalid evidence ID: {evidence_id!r}")
        if not values["provenance_family_ids"]:
            raise ValueError("provenance_family_ids cannot be empty")
        if len(values["rationale"]) < 40:
            raise ValueError("rationale must contain at least 40 characters")
        if not values["next_human_action"]:
            raise ValueError("next_human_action cannot be empty")
        if not values["annotator_id"]:
            raise ValueError("annotator_id cannot be empty")
        if not _valid_iso_date(values["annotation_date"]):
            raise ValueError("annotation_date must use YYYY-MM-DD")
        if not re.fullmatch(r"[0-9]+\.[0-9]+", values["protocol_version"]):
            raise ValueError("protocol_version must resemble 0.1")
        parsed_values = dict(values)
        parsed_values["verdict_confidence"] = confidence
        parsed_values["requires_human_review"] = True
        return cls(**parsed_values)


@dataclass(frozen=True)
class ProvenanceEdge:
    edge_id: str
    child_evidence_id: str
    parent_evidence_id: str
    relation: str
    confidence: str
    dependency_signals: str
    rationale: str

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "edge_id",
        "child_evidence_id",
        "parent_evidence_id",
        "relation",
        "confidence",
        "dependency_signals",
        "rationale",
    )
    RELATIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "quotes",
            "cites",
            "paraphrases",
            "mirrors",
            "syndicates",
            "unknown",
            "citation_mismatch",
        }
    )
    SIGNALS: ClassVar[frozenset[str]] = frozenset(
        {
            "explicit_citation",
            "shared_wording",
            "chronology",
            "shared_error",
            "archive_metadata",
            "near_duplicate",
            "same_url_target",
            "content_comparison",
        }
    )

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> ProvenanceEdge:
        values = {field: (row.get(field) or "").strip() for field in cls.REQUIRED_FIELDS}
        if not re.fullmatch(r"EDGE-[0-9]{3,}", values["edge_id"]):
            raise ValueError("edge_id must resemble EDGE-001")
        for field in ("child_evidence_id", "parent_evidence_id"):
            if not re.fullmatch(r"EVID-[0-9]{3,}", values[field]):
                raise ValueError(f"{field} must resemble EVID-001")
        if values["child_evidence_id"] == values["parent_evidence_id"]:
            raise ValueError("a provenance edge cannot point to itself")
        if values["relation"] not in cls.RELATIONS:
            raise ValueError(f"invalid provenance relation: {values['relation']!r}")
        if values["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium, or low")
        signals = split_ids(values["dependency_signals"])
        if len(set(signals)) < 2:
            raise ValueError("at least two distinct dependency signals are required")
        unknown_signals = sorted(set(signals) - cls.SIGNALS)
        if unknown_signals:
            raise ValueError(f"unknown dependency signals: {', '.join(unknown_signals)}")
        if len(values["rationale"]) < 20:
            raise ValueError("rationale must contain at least 20 characters")
        return cls(**values)


def _validate_csv(path: str | Path, record_type: type) -> tuple[list, list[str]]:
    csv_path = Path(path)
    records: list = []
    errors: list[str] = []
    if not csv_path.is_file():
        return records, [f"file not found: {csv_path}"]
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _required_columns(reader.fieldnames, record_type.REQUIRED_FIELDS)
        if missing:
            return records, [f"missing CSV columns: {', '.join(missing)}"]
        seen_ids: set[str] = set()
        identity_field = record_type.REQUIRED_FIELDS[0]
        for line_number, row in enumerate(reader, start=2):
            try:
                record = record_type.from_mapping(row)
                identity = getattr(record, identity_field)
                if identity in seen_ids:
                    raise ValueError(f"duplicate {identity_field}: {identity}")
                seen_ids.add(identity)
                records.append(record)
            except ValueError as exc:
                errors.append(f"line {line_number}: {exc}")
    if not records and not errors:
        errors.append("CSV contains no data rows")
    return records, errors


def validate_search_execution_csv(
    path: str | Path,
) -> tuple[list[SearchExecution], list[str]]:
    return _validate_csv(path, SearchExecution)


def validate_provisional_verdict_csv(
    path: str | Path,
) -> tuple[list[ProvisionalVerdict], list[str]]:
    return _validate_csv(path, ProvisionalVerdict)


def validate_provenance_edge_csv(path: str | Path) -> tuple[list[ProvenanceEdge], list[str]]:
    return _validate_csv(path, ProvenanceEdge)
