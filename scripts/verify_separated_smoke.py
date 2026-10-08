"""Verify original smoke failure and separately replay revised numeric provenance."""

import json
from pathlib import Path

from smoke_test_generative_interface import CASES

from apv_rag.generation_integrity import new_numeric_values
from apv_rag.generative_rag import LABELS
from apv_rag.input_numeric_integrity import numeric_provenance
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/separated_instruction_smoke_received"
    hashes = json.loads((folder / "output_manifest.json").read_text())
    for name in ("input_manifest.json", "smoke_results.json", "smoke_summary.json"):
        assert sha256(folder / name) == hashes[name], name
    identity = json.loads((folder / "input_manifest.json").read_text())
    for name, digest in identity["code_sha256"].items():
        assert sha256(root / name) == digest, name
    assert sha256(root / "docs/SEPARATED_INSTRUCTION_V5.md") == identity["protocol_sha256"]
    rows = json.loads((folder / "smoke_results.json").read_text())
    assert len(rows) == len(CASES)
    original_passed, revised_passed, replay = 0, 0, []
    for row, case in zip(rows, CASES, strict=True):
        for k, value in case.items():
            assert row[k] == value
        reasons = []
        if case["evidence"]:
            if row["generated_verdict"] not in LABELS:
                reasons.append("invalid_verdict")
            if row["generated_verdict"] in LABELS and not row["generated_explanation"]:
                reasons.append("empty_explanation")
            values = new_numeric_values(row["generated_explanation"] or "", case["evidence"])
            if values:
                reasons.append("novel_numeric_value")
            provenance = numeric_provenance(
                row["generated_explanation"] or "", case["claim"], case["evidence"]
            )
            revised = (
                row["generated_verdict"] == case["expected_verdict"]
                and bool(row["generated_explanation"])
                and not provenance["absent_from_inputs"]
            )
        else:
            reasons = ["no_evidence"]
            values = []
            assert row["generated_verdict"] is None and row["generated_explanation"] is None
            provenance = numeric_provenance("", case["claim"], case["evidence"])
            revised = True
        assert row["reasons"] == reasons and row["novel_numeric_values"] == values
        assert row["status"] == ("abstain" if reasons else "machine_candidate")
        expected = not reasons and row["generated_verdict"] == case["expected_verdict"]
        if not case["evidence"]:
            expected = True
        assert row["expected_action_passed"] == expected
        original_passed += expected
        revised_passed += revised
        replay.append(
            {
                "case": case["id"],
                "numeric_provenance": provenance,
                "revised_structural_action_passed": revised,
            }
        )
    summary = json.loads((folder / "smoke_summary.json").read_text())
    assert summary["passed_expected_actions"] == original_passed
    assert summary["full_run_ready"] == (original_passed == len(rows))
    out = root / "artifacts/separated_instruction_smoke_verification"
    out.mkdir(exist_ok=True)
    report = {
        "verified_cases": len(rows),
        "original_passed_actions": original_passed,
        "original_full_run_ready": summary["full_run_ready"],
        "revised_guard_replay_passed_actions": revised_passed,
        "replay": replay,
        "scope": "post-result guard repair, not independent validation or new inference",
        "semantic_grounding_verified": False,
        "limitation": "Input-present numbers can still be misrepresented; grounding remains open",
        "input_sha256": {n: sha256(folder / n) for n in hashes},
    }
    write_json_atomic(out / "verification.json", report)
    lines = [
        "# Separated smoke verification",
        "",
        f"Original contract: {original_passed}/4 actions passed, readiness false.",
        f"Revised numeric-provenance replay: {revised_passed}/4 actions passed.",
        "",
        "All three generated verdicts match the fixtures. Empty evidence abstains.",
        "The original guard rejected the NEI explanation for referring to 20 degrees "
        "from the claim. The revised check records this as a claim-only reference, "
        "not evidence support. Numbers absent from both inputs remain rejected.",
        "Original results are unchanged. This adaptive replay is not independent "
        "validation, semantic verification, or benchmark performance.",
        "No additional neural inference was performed.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json_atomic(
        out / "analysis_manifest.json",
        {
            "code_sha256": sha256(Path(__file__)),
            "guard_sha256": sha256(root / "src/apv_rag/input_numeric_integrity.py"),
            "outputs": {n: sha256(out / n) for n in ("verification.json", "RESULTS.md")},
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
