"""Verify a received synthetic generator smoke export without neural inference."""

import json
from pathlib import Path

from smoke_test_generative_interface import CASES

from apv_rag.generative_interface import verdict_prompt
from apv_rag.generative_rag import LABELS, parse_answer
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/generative_interface_smoke_received"
    hashes = json.loads((folder / "output_manifest.json").read_text())
    for n in ("input_manifest.json", "smoke_results.json", "smoke_summary.json"):
        assert sha256(folder / n) == hashes[n], n
    identity = json.loads((folder / "input_manifest.json").read_text())
    for n, digest in identity["code_sha256"].items():
        assert sha256(root / n) == digest, n
    assert sha256(root / "docs/GENERATIVE_INTERFACE_V2.md") == identity["protocol_sha256"]
    rows = json.loads((folder / "smoke_results.json").read_text())
    assert len(rows) == len(CASES)
    for row, case in zip(rows, CASES, strict=True):
        for key, value in case.items():
            assert row[key] == value
        assert row["verdict_prompt"] == verdict_prompt(case["claim"], case["evidence"])
        assert row["generated_verdict"] in LABELS
        assert row["assembled_answer"] == (
            f"{row['generated_verdict']} | {row['generated_explanation']}"
        )
        assert row["result"] == parse_answer(
            row["assembled_answer"], [d["id"] for d in case["evidence"]]
        )
        assert row["verdict_correct"] == (row["generated_verdict"] == case["expected_verdict"])
    correct = sum(r["verdict_correct"] for r in rows)
    valid = sum(r["result"]["status"] == "machine_candidate" for r in rows)
    summary = json.loads((folder / "smoke_summary.json").read_text())
    assert summary["cases"] == len(rows)
    assert summary["correct_verdicts"] == correct
    assert summary["structurally_valid_answers"] == valid
    assert summary["full_run_ready"] == (correct == valid == len(rows))
    assert summary["semantic_explanation_support_verified"] is False
    output = root / "artifacts/generative_interface_smoke_verification"
    output.mkdir(exist_ok=True)
    report = {
        "verified_cases": len(rows),
        "correct_verdicts": correct,
        "structurally_valid_answers": valid,
        "full_run_ready": summary["full_run_ready"],
        "verification_limit": "model inference and explanation support not independently verified",
        "findings": [
            "Support verdict was correct but its explanation omitted a citation.",
            "Contradiction case was labeled NEI; its explanation repeated the contradicted claim.",
            "Unrelated-evidence case was correctly labeled NEI.",
            "Empty-evidence case was labeled Supported and omitted a citation.",
        ],
        "input_sha256": {n: sha256(folder / n) for n in hashes},
    }
    write_json_atomic(output / "verification.json", report)
    lines = [
        "# Generative interface smoke verification",
        "",
        f"{correct}/{len(rows)} correct verdicts; {valid}/{len(rows)} structurally valid answers.",
        "Full benchmark readiness: failed.",
        "",
        *report["findings"],
        "",
        "These are synthetic software fixtures, not research benchmark observations.",
        "Constrained verdict decoding solved label formatting but did not solve "
        "evidence use, citation generation or explanation grounding. Do not run the "
        "full benchmark with this interface or relax the gate after observing failure.",
        report["verification_limit"] + ".",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json_atomic(
        output / "analysis_manifest.json",
        {
            "code_sha256": sha256(Path(__file__)),
            "outputs": {n: sha256(output / n) for n in ("verification.json", "RESULTS.md")},
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
