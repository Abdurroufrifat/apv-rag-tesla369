"""Verify the raw generative run without relaxing its frozen answer parser."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from apv_rag.generative_rag import LABELS, make_prompt, parse_answer
from apv_rag.retrieval import BM25Index
from apv_rag.scifact import target_label
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/generative_scifact_received"
    manifest = json.loads((folder / "output_manifest.json").read_text())
    for n in ("predictions.json", "input_manifest.json", "generation_summary.json"):
        assert sha256(folder / n) == manifest[n], n
    identity = json.loads((folder / "input_manifest.json").read_text())
    for n, digest in identity["code_sha256"].items():
        assert sha256(root / n) == digest, n
    assert sha256(root / "docs/GENERATIVE_SCIFACT.md") == identity["protocol_sha256"]
    source = root / "data/external/scifact/sealed_v1"
    for n, digest in identity["source_sha256"].items():
        assert sha256(source / n) == digest, n
    prior = root / "artifacts/scifact_received/predictions.json"
    assert sha256(prior) == identity["cohort_sha256"]
    earlier = json.loads(prior.read_text())
    seed = min(r["seed"] for r in earlier)
    cohort = [
        r["claim_id"]
        for r in earlier
        if r["seed"] == seed and r["representation"] == "base_7" and r["rule"] == "argmax"
    ]
    claims = {
        r["id"]: r
        for r in [json.loads(x) for x in (source / "claims_dev.jsonl").read_text().splitlines()]
    }
    corpus = sorted(
        [json.loads(x) for x in (source / "corpus.jsonl").read_text().splitlines()],
        key=lambda r: r["doc_id"],
    )
    index = BM25Index([" ".join(r["abstract"]) for r in corpus])
    rows = json.loads((folder / "predictions.json").read_text())
    assert [r["claim_id"] for r in rows] == cohort and len(rows) == 300
    reasons, raw = Counter(), Counter()
    truth, predicted = [], []
    for row in rows:
        claim = claims[row["claim_id"]]
        assert row["claim"] == claim["claim"]
        assert row["true_label"] == target_label(claim)
        hits = [(i, score) for i, score in index.search(claim["claim"], top_k=3) if score > 0]
        assert [d["id"] for d in row["evidence"]] == [corpus[i]["doc_id"] for i, _ in hits]
        assert np.allclose([d["bm25_score"] for d in row["evidence"]], [s for _, s in hits])
        shown_claim = row["prompt"].split("Claim: ", 1)[1].split("\nEvidence:", 1)[0]
        assert row["prompt"] == make_prompt(shown_claim, row["evidence"])
        assert 0 < row["prompt_tokens"] <= 512
        parsed = parse_answer(row["raw_generation"], [d["id"] for d in row["evidence"]])
        assert parsed == row["result"]
        reasons.update(parsed["reasons"])
        raw.update([row["raw_generation"].strip()])
        truth.append(row["true_label"])
        predicted.append(parsed["candidate_label"] or "abstain")
    covered = [i for i, p in enumerate(predicted) if p != "abstain"]
    computed = {
        "rows": len(rows),
        "coverage": len(covered) / len(rows),
        "invalid_outputs": len(rows) - len(covered),
        "accuracy_all_claims_abstentions_as_errors": float(accuracy_score(truth, predicted)),
        "macro_f1_all_claims_abstentions_as_errors": float(
            f1_score(truth, predicted, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "covered_accuracy": float(np.mean([truth[i] == predicted[i] for i in covered]))
        if covered
        else None,
    }
    summary = json.loads((folder / "generation_summary.json").read_text())
    for k, v in computed.items():
        assert summary[k] == v or (v is not None and np.isclose(summary[k], v))
    output = root / "artifacts/generative_scifact_verification"
    output.mkdir(exist_ok=True)
    result = {
        "verified_predictions": len(rows),
        "metrics": computed,
        "abstention_reasons": dict(reasons),
        "most_common_raw_outputs": raw.most_common(10),
        "verification_limit": (
            "inference, tokenizer clipping and token counts not independently replayed"
        ),
        "input_sha256": {n: sha256(folder / n) for n in manifest},
    }
    write_json_atomic(output / "verification.json", result)
    lines = [
        "# Generative baseline verification",
        "",
        f"300 outputs verified. Coverage: {computed['coverage']:.4f}. "
        f"Accuracy: {computed['accuracy_all_claims_abstentions_as_errors']:.4f}. "
        f"Macro-F1: {computed['macro_f1_all_claims_abstentions_as_errors']:.4f}.",
        "",
        f"Abstention reasons: {dict(reasons)}.",
        f"Most common raw outputs: {raw.most_common(10)}.",
        "",
        "The frozen output contract remains unchanged. Invalid responses are not "
        "converted into valid verdicts after observing results. This run exposes an "
        "answer-format failure; it does not provide a usable cited generative pipeline.",
        result["verification_limit"] + ".",
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
