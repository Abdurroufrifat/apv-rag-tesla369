#!/usr/bin/env python3
"""Create separate machine-label-free review packets for T369-004."""

from __future__ import annotations

import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.phase1c import HumanReview  # noqa: E402

TEMPLATE = (
    PROJECT_ROOT / "data" / "templates" / "t369004_bilingual_review_template_v0_1.csv"
)
REVIEWER_GUIDE = PROJECT_ROOT / "docs" / "PHASE_1C_REVIEWER_GUIDE.md"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "phase1c_blinded"
FORBIDDEN_MARKERS = (
    "machine_assisted_provisional",
    "tesla_phase1b_verdicts",
    "provisional_verdict",
    "verdict_confidence",
    "0.90",
)


def _read_template() -> tuple[list[str], dict[str, str]]:
    with TEMPLATE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(HumanReview.REQUIRED_FIELDS):
            raise ValueError("Phase 1C template header does not match the review schema")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError("Phase 1C template must contain exactly one blank assignment row")
    row = {key: (value or "") for key, value in rows[0].items()}
    if row["claim_id"] != "T369-004":
        raise ValueError("Phase 1C template must target T369-004")
    return list(HumanReview.REQUIRED_FIELDS), row


def _render_csv(fieldnames: list[str], template_row: dict[str, str], role: str) -> bytes:
    row = dict(template_row)
    row["review_id"] = f"REVIEW-T369-004-{role}"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _source_metadata() -> bytes:
    text = (
        "T369-004 blinded source assignment\n\n"
        "Canonical claim: Nikola Tesla said: The present is theirs; the future, "
        "for which I really worked, is mine.\n"
        "Repository: National Library of Serbia digital archive\n"
        "Newspaper: Politika\n"
        "Date: 1927-04-27\n"
        "Article: Посета г. Николи Тесли\n"
        "Printed page: 2\n"
        "Column: 1 (leftmost)\n"
        "Identifier: URN:NB:RS:SD_2F6F6602455A67B1B521D786232CBF4A-1927-04-27\n"
        "URL: https://digitalna.nb.rs/view/"
        "URN:NB:RS:SD_2F6F6602455A67B1B521D786232CBF4A-1927-04-27\n\n"
        "Open and inspect the original archive scan. This packet intentionally "
        "contains no source image and no prior label.\n"
    )
    return text.encode("utf-8")


def _manifest(files: dict[str, bytes]) -> bytes:
    lines = [f"{hashlib.sha256(content).hexdigest()}  {name}" for name, content in files.items()]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _check_blinding(files: dict[str, bytes]) -> None:
    combined = b"\n".join(files.values()).decode("utf-8", errors="replace").lower()
    leaked = [marker for marker in FORBIDDEN_MARKERS if marker.lower() in combined]
    if leaked:
        raise ValueError(f"reviewer packet contains forbidden machine-label markers: {leaked}")


def _write_deterministic_zip(target: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


def build_packets(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Build the two role-separated ZIP files and return their paths."""

    fieldnames, template_row = _read_template()
    guide = REVIEWER_GUIDE.read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        output_dir / "Reviewer_A_T369-004_blinded.zip",
        output_dir / "Reviewer_B_T369-004_blinded.zip",
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing reviewer packet(s): {names}. "
            "Move them to a dated private archive before rebuilding."
        )

    for role, target in zip(("A", "B"), targets, strict=True):
        files = {
            "README.md": guide,
            "SOURCE_METADATA.txt": _source_metadata(),
            f"T369-004_review_{role}.csv": _render_csv(fieldnames, template_row, role),
        }
        _check_blinding(files)
        files["packet_manifest.sha256"] = _manifest(files)
        _write_deterministic_zip(target, files)
    return targets


def main() -> int:
    print("Phase 1C packet creation is retired by the machine-only protocol.")
    print("Do not generate or distribute human-review packets.")
    print("Run: python scripts/validate_phase2.py")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
