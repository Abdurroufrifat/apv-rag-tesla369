import csv
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from apv_rag.phase1c import HumanReview, validate_source_capture_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKET_SCRIPT = PROJECT_ROOT / "scripts" / "create_phase1c_packets.py"
PACKET_SPEC = importlib.util.spec_from_file_location("create_phase1c_packets", PACKET_SCRIPT)
assert PACKET_SPEC is not None and PACKET_SPEC.loader is not None
PACKET_MODULE = importlib.util.module_from_spec(PACKET_SPEC)
PACKET_SPEC.loader.exec_module(PACKET_MODULE)
FORBIDDEN_MARKERS = PACKET_MODULE.FORBIDDEN_MARKERS
build_packets = PACKET_MODULE.build_packets


def _valid_review_mapping() -> dict[str, str]:
    return {
        "review_id": "REVIEW-T369-004-A",
        "claim_id": "T369-004",
        "reviewer_code": "RA-7K2M",
        "bilingual_qualification": "true",
        "independent_review_confirmed": "true",
        "review_date": "2026-09-06",
        "canonical_claim": (
            "Nikola Tesla said: The present is theirs; the future, for which I really "
            "worked, is mine."
        ),
        "source_url": (
            "https://digitalna.nb.rs/view/"
            "URN:NB:RS:SD_2F6F6602455A67B1B521D786232CBF4A-1927-04-27"
        ),
        "source_identifier": (
            "URN:NB:RS:SD_2F6F6602455A67B1B521D786232CBF4A-1927-04-27"
        ),
        "article_title": "Посета г. Николи Тесли",
        "source_date": "1927-04-27",
        "printed_page": "2",
        "column": "1",
        "serbian_transcription": (
            "Ово је синтетички запис који служи само за проверу софтверске шеме "
            "и није људска анотација историјског извора."
        ),
        "literal_translation": (
            "This is a synthetic record used only to test the software validation schema."
        ),
        "normalized_english": (
            "Synthetic unit-test text; it is not a translation of the historical source."
        ),
        "translation_equivalence": "uncertain",
        "verdict": "insufficient",
        "sufficiency": "insufficient",
        "confidence": "0.88",
        "decisive_evidence_ids": "EVID-112",
        "provenance_family_ids": "PF-POLITIKA-1927",
        "rationale": (
            "This synthetic unit-test fixture checks validation behavior only and must never "
            "be interpreted as a human judgment, historical transcription, or gold label."
        ),
        "missing_evidence": "A contemporaneous English rendering is not available.",
        "limitations": "Historical spelling and scan quality require explicit disclosure.",
        "frozen": "true",
        "frozen_at_utc": "2026-09-06T04:30:00Z",
        "protocol_version": "0.2",
    }


def test_source_capture_register_is_valid_and_uses_known_hash() -> None:
    path = PROJECT_ROOT / "data" / "pilot" / "tesla_phase1c_source_register_v0_1.csv"
    records, errors = validate_source_capture_csv(path)
    assert errors == []
    assert len(records) == 1
    record = records[0]
    assert record.claim_id == "T369-004"
    assert record.printed_page == 2
    assert record.column == 1
    assert record.sha256 == (
        "39a7695ede14d18c8f012882fe92a6d4ab9bd995fdc68dea35d3d6efc80cb2a6"
    )
    assert record.private_relative_path.startswith("data/raw/")


def test_completed_bilingual_review_is_valid() -> None:
    record = HumanReview.from_mapping(_valid_review_mapping())
    assert record.bilingual_qualification is True
    assert record.independent_review_confirmed is True
    assert record.frozen is True
    assert record.translation_equivalence == "uncertain"


def test_t369004_rejects_non_bilingual_or_non_independent_review() -> None:
    row = _valid_review_mapping()
    row["bilingual_qualification"] = "false"
    with pytest.raises(ValueError, match="bilingual"):
        HumanReview.from_mapping(row)

    row = _valid_review_mapping()
    row["independent_review_confirmed"] = "false"
    with pytest.raises(ValueError, match="independent"):
        HumanReview.from_mapping(row)


def test_verdict_and_sufficiency_must_agree() -> None:
    row = _valid_review_mapping()
    row["verdict"] = "insufficient"
    row["sufficiency"] = "sufficient"
    with pytest.raises(ValueError, match="inconsistent"):
        HumanReview.from_mapping(row)


def test_blinded_template_has_no_completed_label() -> None:
    template = (
        PROJECT_ROOT
        / "data"
        / "templates"
        / "t369004_bilingual_review_template_v0_1.csv"
    )
    with template.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["claim_id"] == "T369-004"
    assert row["verdict"] == ""
    assert row["confidence"] == ""
    assert row["translation_equivalence"] == ""
    assert row["frozen"] == "false"


def test_packet_builder_separates_roles_and_leaks_no_machine_label(tmp_path: Path) -> None:
    targets = build_packets(tmp_path)
    assert [path.name for path in targets] == [
        "Reviewer_A_T369-004_blinded.zip",
        "Reviewer_B_T369-004_blinded.zip",
    ]
    for role, target in zip(("A", "B"), targets, strict=True):
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            assert names == {
                "README.md",
                "SOURCE_METADATA.txt",
                f"T369-004_review_{role}.csv",
                "packet_manifest.sha256",
            }
            combined = b"\n".join(archive.read(name) for name in sorted(names)).decode(
                "utf-8"
            )
            for marker in FORBIDDEN_MARKERS:
                assert marker.lower() not in combined.lower()
            assert "politika_1927-04-27_p2_col1_target.png" not in names
            review_text = archive.read(f"T369-004_review_{role}.csv").decode("utf-8")
            assert f"REVIEW-T369-004-{role}" in review_text


def test_human_review_schema_is_closed_and_parseable() -> None:
    with (PROJECT_ROOT / "schemas" / "human_review.schema.json").open(
        encoding="utf-8"
    ) as handle:
        schema = json.load(handle)
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["frozen"]["const"] is True
