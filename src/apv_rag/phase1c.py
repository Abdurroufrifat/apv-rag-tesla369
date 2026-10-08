"""Records and validation for Phase 1C blinded bilingual human review."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import ClassVar
from urllib.parse import urlparse


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


def _parse_bool(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{field} must be true or false")
    return normalized == "true"


def _valid_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _split_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


@dataclass(frozen=True)
class SourceCapture:
    """Metadata for a private evidence capture whose bytes are not committed."""

    capture_id: str
    claim_id: str
    evidence_id: str
    source_url: str
    source_date: str
    article_title: str
    printed_page: int
    column: int
    private_relative_path: str
    sha256: str
    redistribution_status: str
    review_status: str
    protocol_version: str

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "capture_id",
        "claim_id",
        "evidence_id",
        "source_url",
        "source_date",
        "article_title",
        "printed_page",
        "column",
        "private_relative_path",
        "sha256",
        "redistribution_status",
        "review_status",
        "protocol_version",
    )

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> SourceCapture:
        values = {field: (row.get(field) or "").strip() for field in cls.REQUIRED_FIELDS}
        if not re.fullmatch(r"CAP-[0-9]{3,}", values["capture_id"]):
            raise ValueError("capture_id must resemble CAP-001")
        if not re.fullmatch(r"T369-[0-9]{3,}", values["claim_id"]):
            raise ValueError("claim_id must resemble T369-001")
        if not re.fullmatch(r"EVID-[0-9]{3,}", values["evidence_id"]):
            raise ValueError("evidence_id must resemble EVID-001")
        if not _valid_https_url(values["source_url"]):
            raise ValueError("source_url must be an HTTPS URL")
        if not _valid_iso_date(values["source_date"]):
            raise ValueError("source_date must use YYYY-MM-DD")
        if len(values["article_title"]) < 5:
            raise ValueError("article_title is too short")
        try:
            printed_page = int(values["printed_page"])
            column = int(values["column"])
        except ValueError as exc:
            raise ValueError("printed_page and column must be integers") from exc
        if printed_page < 1 or column < 1:
            raise ValueError("printed_page and column must be positive")
        relative_path = PurePosixPath(values["private_relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("private_relative_path must be a safe relative path")
        if relative_path.parts[:2] != ("data", "raw"):
            raise ValueError("private_relative_path must be inside data/raw")
        sha256 = values["sha256"].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        if values["redistribution_status"] != "private_research_copy_do_not_commit":
            raise ValueError("the archival image must remain a private research copy")
        if values["review_status"] != "machine_located_pending_bilingual_review":
            raise ValueError("source capture must remain pending bilingual review")
        if values["protocol_version"] != "0.2":
            raise ValueError("source-capture protocol_version must be 0.2")
        parsed = dict(values)
        parsed["printed_page"] = printed_page
        parsed["column"] = column
        parsed["sha256"] = sha256
        return cls(**parsed)


@dataclass(frozen=True)
class HumanReview:
    """A completed and frozen independent bilingual review record."""

    review_id: str
    claim_id: str
    reviewer_code: str
    bilingual_qualification: bool
    independent_review_confirmed: bool
    review_date: str
    canonical_claim: str
    source_url: str
    source_identifier: str
    article_title: str
    source_date: str
    printed_page: int
    column: int
    serbian_transcription: str
    literal_translation: str
    normalized_english: str
    translation_equivalence: str
    verdict: str
    sufficiency: str
    confidence: float
    decisive_evidence_ids: str
    provenance_family_ids: str
    rationale: str
    missing_evidence: str
    limitations: str
    frozen: bool
    frozen_at_utc: str
    protocol_version: str

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "review_id",
        "claim_id",
        "reviewer_code",
        "bilingual_qualification",
        "independent_review_confirmed",
        "review_date",
        "canonical_claim",
        "source_url",
        "source_identifier",
        "article_title",
        "source_date",
        "printed_page",
        "column",
        "serbian_transcription",
        "literal_translation",
        "normalized_english",
        "translation_equivalence",
        "verdict",
        "sufficiency",
        "confidence",
        "decisive_evidence_ids",
        "provenance_family_ids",
        "rationale",
        "missing_evidence",
        "limitations",
        "frozen",
        "frozen_at_utc",
        "protocol_version",
    )
    VERDICTS: ClassVar[frozenset[str]] = frozenset(
        {"authenticated", "misattributed", "contradicted", "insufficient"}
    )
    EQUIVALENCE_LABELS: ClassVar[frozenset[str]] = frozenset(
        {"exact", "materially_equivalent", "non_equivalent", "uncertain"}
    )

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> HumanReview:
        values = {field: (row.get(field) or "").strip() for field in cls.REQUIRED_FIELDS}
        if not re.fullmatch(r"REVIEW-T369-[0-9]{3,}-[AB]", values["review_id"]):
            raise ValueError("review_id must resemble REVIEW-T369-004-A")
        if not re.fullmatch(r"T369-[0-9]{3,}", values["claim_id"]):
            raise ValueError("claim_id must resemble T369-004")
        if values["claim_id"] not in values["review_id"]:
            raise ValueError("review_id and claim_id do not match")
        if not re.fullmatch(r"R[AB]-[A-Z0-9]{4,}", values["reviewer_code"]):
            raise ValueError("reviewer_code must be anonymous, for example RA-7K2M")
        bilingual = _parse_bool(values["bilingual_qualification"], "bilingual_qualification")
        independent = _parse_bool(
            values["independent_review_confirmed"], "independent_review_confirmed"
        )
        if values["claim_id"] == "T369-004" and not bilingual:
            raise ValueError("T369-004 requires a Serbian-English bilingual reviewer")
        if not independent:
            raise ValueError("the reviewer must confirm independent work")
        if not _valid_iso_date(values["review_date"]):
            raise ValueError("review_date must use YYYY-MM-DD")
        if len(values["canonical_claim"]) < 20:
            raise ValueError("canonical_claim is too short")
        if not _valid_https_url(values["source_url"]):
            raise ValueError("source_url must be an HTTPS URL")
        if len(values["source_identifier"]) < 10:
            raise ValueError("source_identifier is too short")
        if len(values["article_title"]) < 5:
            raise ValueError("article_title is too short")
        if not _valid_iso_date(values["source_date"]):
            raise ValueError("source_date must use YYYY-MM-DD")
        try:
            printed_page = int(values["printed_page"])
            column = int(values["column"])
        except ValueError as exc:
            raise ValueError("printed_page and column must be integers") from exc
        if printed_page < 1 or column < 1:
            raise ValueError("printed_page and column must be positive")
        if len(values["serbian_transcription"]) < 40 or not re.search(
            r"[А-Яа-я]", values["serbian_transcription"]
        ):
            raise ValueError("serbian_transcription must contain the reviewed Serbian text")
        if len(values["literal_translation"]) < 20:
            raise ValueError("literal_translation is too short")
        if len(values["normalized_english"]) < 20:
            raise ValueError("normalized_english is too short")
        if values["translation_equivalence"] not in cls.EQUIVALENCE_LABELS:
            raise ValueError("invalid translation_equivalence")
        if values["verdict"] not in cls.VERDICTS:
            raise ValueError("invalid verdict")
        if values["sufficiency"] not in {"sufficient", "insufficient"}:
            raise ValueError("sufficiency must be sufficient or insufficient")
        expected_sufficiency = (
            "insufficient" if values["verdict"] == "insufficient" else "sufficient"
        )
        if values["sufficiency"] != expected_sufficiency:
            raise ValueError("verdict and sufficiency are inconsistent")
        try:
            confidence = float(values["confidence"])
        except ValueError as exc:
            raise ValueError("confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        evidence_ids = _split_ids(values["decisive_evidence_ids"])
        if not evidence_ids or any(
            not re.fullmatch(r"EVID-[0-9]{3,}", item) for item in evidence_ids
        ):
            raise ValueError("decisive_evidence_ids must contain valid evidence IDs")
        family_ids = _split_ids(values["provenance_family_ids"])
        if not family_ids or any(
            not re.fullmatch(r"PF-[A-Z0-9-]+", item) for item in family_ids
        ):
            raise ValueError("provenance_family_ids must contain valid family IDs")
        if len(values["rationale"]) < 80:
            raise ValueError("rationale must contain at least 80 characters")
        frozen = _parse_bool(values["frozen"], "frozen")
        if not frozen:
            raise ValueError("a completed human review must be frozen")
        if not _valid_iso_datetime(values["frozen_at_utc"]):
            raise ValueError("frozen_at_utc must be a timezone-aware ISO datetime")
        if values["protocol_version"] != "0.2":
            raise ValueError("human-review protocol_version must be 0.2")
        parsed = dict(values)
        parsed["bilingual_qualification"] = bilingual
        parsed["independent_review_confirmed"] = independent
        parsed["printed_page"] = printed_page
        parsed["column"] = column
        parsed["confidence"] = confidence
        parsed["frozen"] = frozen
        return cls(**parsed)


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


def validate_source_capture_csv(path: str | Path) -> tuple[list[SourceCapture], list[str]]:
    """Validate a coordinator-only source-capture register."""

    return _validate_csv(path, SourceCapture)


def validate_human_review_csv(path: str | Path) -> tuple[list[HumanReview], list[str]]:
    """Validate one or more completed, frozen human review records."""

    return _validate_csv(path, HumanReview)
