"""Validation helpers for Phase 1A evidence and archival search records."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse


def _required_columns(fieldnames: list[str] | None, required: tuple[str, ...]) -> list[str]:
    if fieldnames is None:
        return list(required)
    return [field for field in required if field not in fieldnames]


def _valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _valid_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    claim_id: str
    title: str
    creator: str
    publisher_or_archive: str
    publication_date: str
    source_url: str
    access_date: str
    source_type: str
    source_rank: int
    is_primary: bool
    location_pointer: str
    stance: str
    evidence_summary: str
    provenance_family_id: str
    parent_evidence_id: str
    provenance_relation: str
    provenance_confidence: str
    ocr_status: str
    verification_status: str
    annotator_id: str
    notes: str

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "evidence_id",
        "claim_id",
        "title",
        "creator",
        "publisher_or_archive",
        "publication_date",
        "source_url",
        "access_date",
        "source_type",
        "source_rank",
        "is_primary",
        "location_pointer",
        "stance",
        "evidence_summary",
        "provenance_family_id",
        "parent_evidence_id",
        "provenance_relation",
        "provenance_confidence",
        "ocr_status",
        "verification_status",
        "annotator_id",
        "notes",
    )
    STANCES: ClassVar[frozenset[str]] = frozenset(
        {"supports", "refutes", "mentions_only", "unrelated"}
    )
    RELATIONS: ClassVar[frozenset[str]] = frozenset(
        {"original", "quotes", "cites", "paraphrases", "mirrors", "syndicates", "unknown"}
    )
    PROVENANCE_CONFIDENCE: ClassVar[frozenset[str]] = frozenset({"high", "medium", "low"})
    OCR_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {
            "not_applicable",
            "scan_available",
            "ocr_plus_scan",
            "digital_text",
            "transcription_available",
            "transcription_unchecked",
        }
    )
    VERIFICATION_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {
            "screened",
            "metadata_verified",
            "needs_scan_check",
            "needs_page_review",
            "needs_full_text_review",
        }
    )

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> EvidenceRecord:
        missing = [field for field in cls.REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        values = {field: (row.get(field) or "").strip() for field in cls.REQUIRED_FIELDS}

        if not re.fullmatch(r"EVID-[0-9]{3,}", values["evidence_id"]):
            raise ValueError("evidence_id must resemble EVID-001")
        if not re.fullmatch(r"T369-[0-9]{3,}", values["claim_id"]):
            raise ValueError("claim_id must resemble T369-001")
        if not values["title"]:
            raise ValueError("title cannot be empty")
        if not values["publisher_or_archive"]:
            raise ValueError("publisher_or_archive cannot be empty")
        if not _valid_http_url(values["source_url"]):
            raise ValueError(f"invalid source URL: {values['source_url']!r}")
        if not _valid_iso_date(values["access_date"]):
            raise ValueError("access_date must use YYYY-MM-DD")

        try:
            source_rank = int(values["source_rank"])
        except ValueError as exc:
            raise ValueError("source_rank must be an integer") from exc
        if source_rank not in range(1, 6):
            raise ValueError("source_rank must be between 1 and 5")

        bool_text = values["is_primary"].lower()
        if bool_text not in {"true", "false"}:
            raise ValueError("is_primary must be true or false")
        is_primary = bool_text == "true"

        if values["stance"] not in cls.STANCES:
            raise ValueError(f"invalid stance: {values['stance']!r}")
        if len(values["evidence_summary"]) < 10:
            raise ValueError("evidence_summary must contain at least 10 characters")
        if not values["provenance_family_id"]:
            raise ValueError("provenance_family_id cannot be empty")
        if values["parent_evidence_id"] == values["evidence_id"]:
            raise ValueError("an evidence record cannot be its own parent")
        if values["provenance_relation"] not in cls.RELATIONS:
            raise ValueError(f"invalid provenance_relation: {values['provenance_relation']!r}")
        if (
            values["provenance_relation"] not in {"original", "unknown"}
            and not values["parent_evidence_id"]
        ):
            raise ValueError("non-original provenance relations require parent_evidence_id")
        if values["provenance_confidence"] not in cls.PROVENANCE_CONFIDENCE:
            raise ValueError("provenance_confidence must be high, medium, or low")
        if values["ocr_status"] not in cls.OCR_STATUSES:
            raise ValueError(f"invalid ocr_status: {values['ocr_status']!r}")
        if values["verification_status"] not in cls.VERIFICATION_STATUSES:
            raise ValueError(f"invalid verification_status: {values['verification_status']!r}")
        if not values["annotator_id"]:
            raise ValueError("annotator_id cannot be empty")

        return cls(
            evidence_id=values["evidence_id"],
            claim_id=values["claim_id"],
            title=values["title"],
            creator=values["creator"],
            publisher_or_archive=values["publisher_or_archive"],
            publication_date=values["publication_date"],
            source_url=values["source_url"],
            access_date=values["access_date"],
            source_type=values["source_type"],
            source_rank=source_rank,
            is_primary=is_primary,
            location_pointer=values["location_pointer"],
            stance=values["stance"],
            evidence_summary=values["evidence_summary"],
            provenance_family_id=values["provenance_family_id"],
            parent_evidence_id=values["parent_evidence_id"],
            provenance_relation=values["provenance_relation"],
            provenance_confidence=values["provenance_confidence"],
            ocr_status=values["ocr_status"],
            verification_status=values["verification_status"],
            annotator_id=values["annotator_id"],
            notes=values["notes"],
        )


def validate_evidence_csv(path: str | Path) -> tuple[list[EvidenceRecord], list[str]]:
    csv_path = Path(path)
    records: list[EvidenceRecord] = []
    errors: list[str] = []
    if not csv_path.is_file():
        return records, [f"file not found: {csv_path}"]

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _required_columns(reader.fieldnames, EvidenceRecord.REQUIRED_FIELDS)
        if missing:
            return records, [f"missing CSV columns: {', '.join(missing)}"]

        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                record = EvidenceRecord.from_mapping(row)
                if record.evidence_id in seen_ids:
                    raise ValueError(f"duplicate evidence_id: {record.evidence_id}")
                seen_ids.add(record.evidence_id)
                records.append(record)
            except ValueError as exc:
                errors.append(f"line {line_number}: {exc}")
    if not records and not errors:
        errors.append("CSV contains no data rows")
    return records, errors


SEARCH_REQUIRED_FIELDS: tuple[str, ...] = (
    "search_id",
    "claim_id",
    "primary_repositories",
    "exact_query",
    "fragment_or_concept_query",
    "date_frozen",
    "status",
    "assigned_to",
    "notes",
)


def validate_search_plan_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    csv_path = Path(path)
    records: list[dict[str, str]] = []
    errors: list[str] = []
    if not csv_path.is_file():
        return records, [f"file not found: {csv_path}"]

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _required_columns(reader.fieldnames, SEARCH_REQUIRED_FIELDS)
        if missing:
            return records, [f"missing CSV columns: {', '.join(missing)}"]

        seen_ids: set[str] = set()
        for line_number, raw_row in enumerate(reader, start=2):
            row = {field: (raw_row.get(field) or "").strip() for field in SEARCH_REQUIRED_FIELDS}
            row_errors: list[str] = []
            if not re.fullmatch(r"SEARCH-[0-9]{3,}", row["search_id"]):
                row_errors.append("search_id must resemble SEARCH-001")
            if row["search_id"] in seen_ids:
                row_errors.append(f"duplicate search_id: {row['search_id']}")
            if not re.fullmatch(r"T369-[0-9]{3,}", row["claim_id"]):
                row_errors.append("claim_id must resemble T369-001")
            if not row["primary_repositories"]:
                row_errors.append("primary_repositories cannot be empty")
            if not row["exact_query"] or not row["fragment_or_concept_query"]:
                row_errors.append("both query fields are required")
            if not _valid_iso_date(row["date_frozen"]):
                row_errors.append("date_frozen must use YYYY-MM-DD")
            if row["status"] not in {"planned", "in_progress", "completed", "blocked"}:
                row_errors.append("invalid search status")
            if not row["assigned_to"]:
                row_errors.append("assigned_to cannot be empty")

            if row_errors:
                errors.extend(f"line {line_number}: {message}" for message in row_errors)
            else:
                seen_ids.add(row["search_id"])
                records.append(row)
    if not records and not errors:
        errors.append("CSV contains no data rows")
    return records, errors
