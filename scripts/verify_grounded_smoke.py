"""Reproduce hybrid smoke decisions from exported machine scores."""

import json
from pathlib import Path

from smoke_test_generative_interface import CASES

from apv_rag.grounded_generation import evidence_decision, explanation_passes
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/grounded_generation_smoke_received"
    hashes = json.loads((folder / "output_manifest.json").read_text())
    for n in ("input_manifest.json", "smoke_results.json", "smoke_summary.json"):
        assert sha256(folder / n) == hashes[n], n
    identity = json.loads((folder / "input_manifest.json").read_text())
    for n, digest in identity["code_sha256"].items():
        assert sha256(root / n) == digest, n
    assert sha256(root / "docs/GROUNDED_GENERATION_V3.md") == identity["protocol_sha256"]
    rows = json.loads((folder / "smoke_results.json").read_text())
    assert len(rows) == len(CASES)
    for row, case in zip(rows, CASES, strict=True):
        for k, v in case.items():
            assert row[k] == v
        assert len(row["claim_nli_scores_cen"]) == len(case["evidence"])
        decision = evidence_decision(row["claim_nli_scores_cen"])
        assert row["nli_decision"] == decision
        if decision["selected"] is None:
            assert row["generation_prompt"] is None
            assert row["raw_generated_explanation"] is None
            assert row["explanation_nli_scores_cen"] is None
            assert row["selected_source_id"] is None
            accepted = False
        else:
            doc = case["evidence"][decision["selected"]]
            assert row["selected_source_id"] == doc["id"]
            assert row["generation_prompt"] == (
                f"Summarize the following evidence in one sentence: {doc['text']}"
            )
            accepted = explanation_passes(
                row["raw_generated_explanation"], row["explanation_nli_scores_cen"]
            )
        assert row["explanation_accepted"] == accepted
        assert row["status"] == ("machine_candidate" if accepted else "abstain")
        assert row["verdict_correct"] == (decision["label"] == case["expected_verdict"])
    correct = sum(r["verdict_correct"] for r in rows)
    accepted = sum(r["explanation_accepted"] for r in rows)
    ready = all(
        r["verdict_correct"]
        and (
            r["explanation_accepted"]
            if r["expected_verdict"] != "Not Enough Evidence"
            else r["status"] == "abstain" and r["raw_generated_explanation"] is None
        )
        for r in rows
    )
    summary = json.loads((folder / "smoke_summary.json").read_text())
    assert summary["correct_nli_verdicts"] == correct
    assert summary["accepted_generated_explanations"] == accepted
    assert summary["full_run_ready"] == ready
    out = root / "artifacts/grounded_generation_smoke_verification"
    out.mkdir(exist_ok=True)
    report = {
        "verified_cases": len(rows),
        "correct_nli_verdicts": correct,
        "accepted_generated_summaries": accepted,
        "full_run_ready": ready,
        "findings": [
            "Support summary changed 20 degrees Celsius to 0.005; the NLI gate rejected it.",
            "Contradiction verdict and its generated summary passed.",
            "Unrelated evidence was incorrectly labeled Refuted; its summary passed entailment.",
            "Empty evidence correctly abstained without generation.",
        ],
        "verification_limit": "decisions replayed from exported scores; neural inference not rerun",
        "input_sha256": {n: sha256(folder / n) for n in hashes},
    }
    write_json_atomic(out / "verification.json", report)
    lines = [
        "# Hybrid generation smoke verification",
        "",
        f"{correct}/4 correct NLI verdicts; {accepted}/4 accepted generated summaries.",
        f"Full-run readiness: {ready}.",
        "",
        *report["findings"],
        "",
        "Summary entailment does not establish claim relevance or correct stance. "
        "The full benchmark remains blocked. Synthetic cases are not research results.",
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
