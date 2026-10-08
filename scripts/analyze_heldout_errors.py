"""Descriptive post-evaluation diagnostics; never tune models or decision rules."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from apv_rag.evidence_baseline import LABELS
from apv_rag.splits import sha256, write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inputs = args.input_dir or root / "artifacts/heldout_received"
    rows = json.loads((inputs / "predictions.json").read_text())
    audit = json.loads((inputs / "overlap_audit.json").read_text())
    summary = json.loads((inputs / "heldout_summary.json").read_text())
    manifest = json.loads((inputs / "output_manifest.json").read_text())
    for name in ("predictions.json", "overlap_audit.json", "heldout_summary.json"):
        if sha256(inputs / name) != manifest[name]:
            raise ValueError(f"received checksum mismatch: {name}")
    protocol = json.loads((root / "artifacts/frozen_heldout_protocol/protocol.json").read_text())
    lookup = {(r["method"], r["seed"], r["dev_index"]): r for r in rows}
    expected = {
        (m, s, i) for m in protocol["methods"] for s in protocol["seeds"] for i in range(500)
    }
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("incomplete or duplicate paired predictions")
    ids = audit["independent_ids"]
    if summary["primary_independent"]["claims"] != len(ids):
        raise ValueError("primary population differs from uploaded summary")
    truth = [lookup[("bm25_nli", protocol["seeds"][0], i)]["true_label"] for i in ids]
    results = []
    for method in protocol["methods"]:
        matrices = {rule: [] for rule in ("argmax", "selected")}
        transitions, wrong_sets, confidence_errors = [], [], []
        for seed in protocol["seeds"]:
            batch = [lookup[(method, seed, i)] for i in ids]
            if [r["true_label"] for r in batch] != truth:
                raise ValueError("truth differs across methods or seeds")
            original = np.asarray([r["argmax_label"] == r["true_label"] for r in batch])
            selected = np.asarray([r["selected_label"] == r["true_label"] for r in batch])
            transitions.append(
                {
                    "seed": seed,
                    "corrected": int((~original & selected).sum()),
                    "new_errors": int((original & ~selected).sum()),
                    "both_wrong": int((~original & ~selected).sum()),
                    "both_correct": int((original & selected).sum()),
                }
            )
            wrong_sets.append(
                {r["dev_index"] for r, ok in zip(batch, selected, strict=True) if not ok}
            )
            confidence_errors.append(
                sum(
                    not ok and max(r["probabilities"]) >= 0.8
                    for r, ok in zip(batch, original, strict=True)
                )
            )
            for rule in matrices:
                matrices[rule].append(
                    confusion_matrix(truth, [r[f"{rule}_label"] for r in batch], labels=LABELS)
                )
        mean_matrices = {rule: np.mean(value, axis=0) for rule, value in matrices.items()}
        assert all(np.isclose(m.sum(), len(ids)) for m in mean_matrices.values())
        class_rows = []
        for i, label in enumerate(LABELS):
            record = {"label": label, "support": truth.count(label)}
            for rule, matrix in mean_matrices.items():
                record[rule] = {
                    "mean_predicted_count": float(matrix[:, i].sum()),
                    "mean_recall": float(matrix[i, i] / matrix[i].sum())
                    if matrix[i].sum()
                    else None,
                }
            class_rows.append(record)
        result = {
            "method": method,
            "per_class": class_rows,
            "mean_confusion_matrices": {k: v.tolist() for k, v in mean_matrices.items()},
            "seed_transitions": transitions,
            "mean_corrected": float(np.mean([r["corrected"] for r in transitions])),
            "mean_new_errors": float(np.mean([r["new_errors"] for r in transitions])),
            "selected_wrong_in_all_seeds": sorted(set.intersection(*wrong_sets)),
            "mean_raw_argmax_errors_at_confidence_ge_0_8": float(np.mean(confidence_errors)),
        }
        results.append(result)
    output = root / "artifacts/heldout_error_analysis"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "descriptive post-evaluation analysis; no tuning or new confirmation",
        "population": "461 independent claims under frozen exact-claim/article grouping",
        "claims": len(ids),
        "seeds": protocol["seeds"],
        "labels": LABELS,
        "confusion_orientation": "rows=true label; columns=predicted label; mean across seeds",
        "limitations": [
            "predictions alone cannot establish retrieval or NLI failure causes",
            "source-level audits and evidence features were not uploaded",
            "confidence threshold is descriptive, not a selected operating rule",
        ],
        "results": results,
        "input_sha256": {
            n: sha256(inputs / n)
            for n in ("predictions.json", "overlap_audit.json", "heldout_summary.json")
        },
    }
    write_json_atomic(output / "error_analysis.json", report)
    text = [
        "# Held-out error analysis",
        "",
        "Descriptive analysis of the frozen evaluation; no model or rule was changed.",
        "",
        f"Primary population: {len(ids)} claims. "
        f"Training-linked exclusions: {len(audit['excluded'])}.",
        "Five seeds are repeated model runs, not five independent datasets.",
        "",
    ]
    for result in results:
        text.extend(
            [
                f"## {result['method']}",
                "",
                "| Class | Claims | Argmax recall | Selected recall | Mean selected count |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in result["per_class"]:
            a, b = row["argmax"]["mean_recall"], row["selected"]["mean_recall"]
            text.append(
                f"| {row['label']} | {row['support']} | {a:.3f} | {b:.3f} | "
                f"{row['selected']['mean_predicted_count']:.1f} |"
            )
        text.extend(
            [
                "",
                f"Mean corrected decisions per seed: {result['mean_corrected']:.1f}.",
                f"Mean new errors per seed: {result['mean_new_errors']:.1f}.",
                f"Claims wrong under the selected rule in all five seeds: "
                f"{len(result['selected_wrong_in_all_seeds'])}.",
                "",
            ]
        )
        for rule, matrix in result["mean_confusion_matrices"].items():
            accuracy = np.trace(np.asarray(matrix)) / len(ids)
            text.append(f"Mean {rule} accuracy: {accuracy:.4f}.")
        text.append("")
    text.extend(
        [
            "## Interpretation",
            "",
            "The selected rules increase Macro-F1, but both frozen primary confidence intervals",
            "include zero and both Holm-adjusted p-values exceed 0.05. These diagnostics do not",
            "change that result. Class recall and confusion counts describe the observed errors;",
            "they do not establish whether retrieval, evidence quality or NLI caused them.",
            "",
            "The next implementation should be developed on training/internal validation only.",
            "This official-dev evaluation is now observed and cannot be reused as an untouched",
            "confirmation set for a revised method. Separate new evaluation data is required.",
            "",
            "Scope remains an oracle answer-excerpt comparison. Full open-web APV-RAG, planned",
            "ablations and transfer experiments remain unfinished.",
            "",
        ]
    )
    (output / "ERROR_ANALYSIS.md").write_text("\n".join(text), encoding="utf-8")
    print(
        json.dumps(
            {
                "claims": len(ids),
                "methods": [
                    {k: r[k] for k in ("method", "mean_corrected", "mean_new_errors")}
                    for r in results
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
