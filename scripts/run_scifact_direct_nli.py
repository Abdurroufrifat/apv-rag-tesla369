"""Evaluate an untuned direct NLI control using the completed target audit."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from apv_rag.direct_nli import TARGET_LABELS, direct_probabilities
from apv_rag.metrics import multiclass_brier
from apv_rag.scifact import target_label
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = root / "artifacts/scifact_transfer"
    output = root / "artifacts/scifact_direct_nli"
    if output.exists():
        raise FileExistsError("Completed baseline exists; refusing overwrite")
    manifest = json.loads((inputs / "output_manifest.json").read_text())
    for name in ("retrieval_audit.json", "predictions.json", "input_manifest.json"):
        if sha256(inputs / name) != manifest[name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    identity = json.loads((inputs / "input_manifest.json").read_text())
    claims_path = root / "data/external/scifact/sealed_v1/claims_dev.jsonl"
    if sha256(claims_path) != identity["target_inputs"]["claims_dev.jsonl"]:
        raise ValueError("Target checksum mismatch")
    claims = {
        r["id"]: r for r in [json.loads(line) for line in claims_path.read_text().splitlines()]
    }
    audit = json.loads((inputs / "retrieval_audit.json").read_text())
    by_id = {r["claim_id"]: r for r in audit}
    if len(by_id) != len(audit):
        raise ValueError("Duplicate audit claims")
    previous = json.loads((inputs / "predictions.json").read_text())
    seed = identity["seeds"][0]
    cohort = [
        r
        for r in previous
        if r["representation"] == "base_7" and r["rule"] == "argmax" and r["seed"] == seed
    ]
    if len({r["claim_id"] for r in cohort}) != len(cohort) or not cohort:
        raise ValueError("Invalid reference cohort")
    results = []
    for record in cohort:
        claim_id = record["claim_id"]
        if target_label(claims[claim_id]) != record["true_label"]:
            raise ValueError("Target truth mismatch")
        item = by_id[claim_id]
        if not len(item["premises"]) == len(item["doc_ids"]) == len(item["nli_probabilities"]):
            raise ValueError("Audit alignment mismatch")
        if len(item["premises"]) > 5:
            raise ValueError("Unexpected retrieval count")
        p = direct_probabilities(item["nli_probabilities"])
        results.append(
            {
                "claim_id": claim_id,
                "group": record["group"],
                "true_label": record["true_label"],
                "predicted_label": TARGET_LABELS[int(p.argmax())],
                "probabilities": p.tolist(),
            }
        )
    y = [r["true_label"] for r in results]
    pred = [r["predicted_label"] for r in results]
    p = np.array([r["probabilities"] for r in results])
    summary = {
        "scope": "post-transfer untuned diagnostic control; not new confirmation",
        "rule": "mean C/E/N scores; map to target classes; fixed first-index tie break",
        "missing_evidence_rule": "Not Enough Evidence probability one",
        "seeds": "deterministic; not five independent fitted models",
        "claims": len(results),
        "groups": len({r["group"] for r in results}),
        "macro_f1": float(
            f1_score(y, pred, labels=list(TARGET_LABELS), average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y, pred)),
        "multiclass_brier": multiclass_brier(y, p, TARGET_LABELS),
        "per_class": classification_report(
            y, pred, labels=list(TARGET_LABELS), output_dict=True, zero_division=0
        ),
        "limitation": (
            "NLI neutral is not a validated measure of claim-level evidence insufficiency"
        ),
    }
    output.mkdir()
    write_json_atomic(output / "baseline_summary.json", summary)
    write_json_atomic(output / "predictions.json", results)
    write_json_atomic(
        output / "input_manifest.json",
        {
            "inputs": {
                name: sha256(inputs / name)
                for name in ("retrieval_audit.json", "predictions.json", "input_manifest.json")
            },
            "script_sha256": sha256(Path(__file__)),
            "aggregation_sha256": sha256(root / "src/apv_rag/direct_nli.py"),
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.iterdir() if p.name != "output_manifest.json"},
    )
    print(output / "baseline_summary.json")


if __name__ == "__main__":
    main()
