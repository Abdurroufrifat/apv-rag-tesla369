"""Claim-level records and validation for the SciAttr-369 pilot dataset."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse


@dataclass(frozen=True)
class ClaimRecord:
    """A validated claim-level pilot record.

    Evidence will move to its own one-to-many table after the pilot. Keeping this
    model deliberately small makes early annotation errors easy to inspect.
    """

    claim_id: str
    canonical_claim_id: str
    claim_text: str
    claimed_person: str
    claim_type: str
    language: str
    verdict: str
    verdict_confidence: float
    sufficiency: str
    source_urls: str
    provenance_family_ids: str
    review_status: str
    annotator_id: str
    notes: str

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "claim_id",
        "canonical_claim_id",
        "claim_text",
        "claimed_person",
        "claim_type",
        "language",
        "verdict",
        "verdict_confidence",
        "sufficiency",
        "source_urls",
        "provenance_family_ids",
        "review_status",
        "annotator_id",
        "notes",
    )
    CLAIM_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"quotation", "discovery", "belief", "behavior", "prediction", "other"}
    )
    VERDICTS: ClassVar[frozenset[str]] = frozenset(
        {"authenticated", "misattributed", "contradicted", "insufficient"}
    )
    SUFFICIENCY: ClassVar[frozenset[str]] = frozenset({"sufficient", "insufficient"})
    REVIEW_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"candidate", "pilot", "reviewed", "adjudicated"}
    )

    @classmethod
    def from_mapping(cls, row: Mapping[str, str]) -> ClaimRecord:
        """Parse and validate one CSV-like mapping."""

        missing = [field for field in cls.REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

        values = {field: (row.get(field) or "").strip() for field in cls.REQUIRED_FIELDS}

        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]*", values["claim_id"]):
            raise ValueError("claim_id must use uppercase letters, digits, underscores, or hyphens")
        if not values["canonical_claim_id"]:
            raise ValueError("canonical_claim_id cannot be empty")
        if len(values["claim_text"]) < 10:
            raise ValueError("claim_text must contain at least 10 characters")
        if not values["claimed_person"]:
            raise ValueError("claimed_person cannot be empty")
        if values["claim_type"] not in cls.CLAIM_TYPES:
            raise ValueError(f"invalid claim_type: {values['claim_type']!r}")
        if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Z]{2})?", values["language"]):
            raise ValueError("language must resemble 'en' or 'en-US'")
        if values["verdict"] not in cls.VERDICTS:
            raise ValueError(f"invalid verdict: {values['verdict']!r}")
        if values["sufficiency"] not in cls.SUFFICIENCY:
            raise ValueError(f"invalid sufficiency: {values['sufficiency']!r}")
        if values["review_status"] not in cls.REVIEW_STATUSES:
            raise ValueError(f"invalid review_status: {values['review_status']!r}")
        if not values["annotator_id"]:
            raise ValueError("annotator_id cannot be empty")

        try:
            confidence = float(values["verdict_confidence"])
        except ValueError as exc:
            raise ValueError("verdict_confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("verdict_confidence must be between 0 and 1")

        if values["verdict"] == "insufficient" and values["sufficiency"] != "insufficient":
            raise ValueError("an insufficient verdict must have insufficient evidence")
        if values["verdict"] != "insufficient" and values["sufficiency"] != "sufficient":
            raise ValueError("a definitive verdict requires sufficient evidence")

        for url in _split_multi_value(values["source_urls"]):
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"invalid source URL: {url!r}")

        return cls(
            claim_id=values["claim_id"],
            canonical_claim_id=values["canonical_claim_id"],
            claim_text=values["claim_text"],
            claimed_person=values["claimed_person"],
            claim_type=values["claim_type"],
            language=values["language"],
            verdict=values["verdict"],
            verdict_confidence=confidence,
            sufficiency=values["sufficiency"],
            source_urls=values["source_urls"],
            provenance_family_ids=values["provenance_family_ids"],
            review_status=values["review_status"],
            annotator_id=values["annotator_id"],
            notes=values["notes"],
        )


def _split_multi_value(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def validate_claim_csv(path: str | Path) -> tuple[list[ClaimRecord], list[str]]:
    """Validate every row and return records plus human-readable errors."""

    csv_path = Path(path)
    records: list[ClaimRecord] = []
    errors: list[str] = []

    if not csv_path.is_file():
        return records, [f"file not found: {csv_path}"]

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return records, ["CSV has no header"]

        missing_columns = [
            field for field in ClaimRecord.REQUIRED_FIELDS if field not in reader.fieldnames
        ]
        if missing_columns:
            return records, [f"missing CSV columns: {', '.join(missing_columns)}"]

        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                record = ClaimRecord.from_mapping(row)
                if record.claim_id in seen_ids:
                    raise ValueError(f"duplicate claim_id: {record.claim_id}")
                seen_ids.add(record.claim_id)
                records.append(record)
            except ValueError as exc:
                errors.append(f"line {line_number}: {exc}")

    if not records and not errors:
        errors.append("CSV contains no data rows")
    return records, errors
