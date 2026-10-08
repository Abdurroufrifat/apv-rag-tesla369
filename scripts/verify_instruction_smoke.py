"""Verify the replacement generator export without changing its frozen parser."""

import json
from pathlib import Path

from smoke_test_generative_interface import CASES

from apv_rag.generation_integrity import new_numeric_values
from apv_rag.generative_rag import parse_answer
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/instruction_model_smoke_received"
    hashes = json.loads((folder / "output_manifest.json").read_text())
    for name in ("input_manifest.json", "smoke_results.json", "smoke_summary.json"):
        assert sha256(folder / name) == hashes[name], name
    identity = json.loads((folder / "input_manifest.json").read_text())
    for name, digest in identity["code_sha256"].items():
        assert sha256(root / name) == digest, name
    assert sha256(root / "docs/INSTRUCTION_MODEL_SMOKE_V4.md") == identity["protocol_sha256"]
    rows = json.loads((folder / "smoke_results.json").read_text())
    assert len(rows) == len(CASES)
    passed = 0
    for row, case in zip(rows, CASES, strict=True):
        for k, value in case.items():
            assert row[k] == value
        if case["evidence"]:
            result = parse_answer(row["raw_generation"], [d["id"] for d in case["evidence"]])
            numbers = new_numeric_values(result["explanation"], case["evidence"])
            if numbers:
                result["status"] = "abstain"
                result["candidate_label"] = None
                result["reasons"].append("novel_numeric_value")
            correct = result["candidate_label"] == case["expected_verdict"]
            assert row["decision_origin"] == "generator"
        else:
            assert row["raw_generation"] is None and row["chat_prompt"] is None
            result = {
                "status": "abstain",
                "candidate_label": None,
                "reasons": ["no_evidence"],
                "explanation": "",
                "citation_ids": [],
            }
            numbers, correct = [], True
            assert row["decision_origin"] == "empty-evidence gate"
        assert row["result"] == result and row["novel_numeric_values"] == numbers
        assert row["expected_action_passed"] == correct
        passed += correct
    summary = json.loads((folder / "smoke_summary.json").read_text())
    assert summary["cases"] == len(rows)
    assert summary["passed_expected_actions"] == passed
    assert summary["full_run_ready"] == (passed == len(rows))
    out = root / "artifacts/instruction_model_smoke_verification"
    out.mkdir(exist_ok=True)
    report = {
        "verified_cases": len(rows),
        "passed_expected_actions": passed,
        "full_run_ready": summary["full_run_ready"],
        "findings": [
            "All three model-generated answers failed the output contract.",
            "The support answer used a literal verdict placeholder and an unknown ID format.",
            "The other two generated answers omitted the delimiter and explanation.",
            "The only passing case was the application empty-evidence gate.",
        ],
        "verification_limit": "parser and numeric checks replayed; model inference not rerun",
        "input_sha256": {n: sha256(folder / n) for n in hashes},
    }
    write_json_atomic(out / "verification.json", report)
    lines = [
        "# Replacement generator smoke verification",
        "",
        f"{passed}/4 expected actions passed. Full-run readiness: false.",
        "",
        *report["findings"],
        "",
        "No demonstrated improvement in usable cited generation. Preserve the export and "
        "keep the full benchmark blocked. Do not reinterpret malformed outputs as valid "
        "answers after observing this failure. These fixtures are not research results.",
        report["verification_limit"] + ".",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json_atomic(
        out / "analysis_manifest.json",
        {
            "code_sha256": sha256(Path(__file__)),
            "outputs": {n: sha256(out / n) for n in ("verification.json", "RESULTS.md")},
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
