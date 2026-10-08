"""Select decisions with grouped training folds; reuse saved NLI features only."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from apv_rag.decision_rules import decision_indices
from apv_rag.evidence_baseline import LABELS
from apv_rag.nli_comparison import SEEDS
from apv_rag.splits import sha256, write_json_atomic


def fit_model(x, y, seed):
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    model = CalibratedClassifierCV(
        base, method="sigmoid", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    )
    model.fit(x, y)
    return model


def ordered_probabilities(model, x):
    return model.predict_proba(x)[:, [list(model.classes_).index(label) for label in LABELS]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inputs = args.input_dir or root / "artifacts/retrieval_nli_comparison"
    original_manifest = json.loads((inputs / "output_manifest.json").read_text())
    input_manifest = json.loads((inputs / "input_manifest.json").read_text())
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    if sha256(source) != input_manifest["source_sha256"]:
        raise ValueError("source checksum mismatch")
    ids = {
        s: json.loads((split / f"{s}_indices.json").read_text()) for s in ("train", "validation")
    }
    for s in ids:
        if sha256(split / f"{s}_indices.json") != input_manifest[f"{s}_indices_sha256"]:
            raise ValueError(f"{s} split checksum mismatch")
    records = json.loads(source.read_text())
    y = {s: np.asarray([records[i]["label"] for i in indices]) for s, indices in ids.items()}
    groups = json.loads((split / "group_assignments.json").read_text())
    train_groups = np.asarray([groups[str(i)] for i in ids["train"]])
    if set(train_groups) & {groups[str(i)] for i in ids["validation"]}:
        raise ValueError("provenance groups cross split boundary")
    original_predictions = json.loads((inputs / "predictions.json").read_text())
    if sha256(inputs / "predictions.json") != original_manifest["predictions.json"]:
        raise ValueError("original predictions checksum mismatch")
    output = root / "artifacts/cached_decision_comparison"
    if output.exists():
        raise FileExistsError("refusing to overwrite decision-comparison results")
    grid = [0, 0.25, 0.5, 0.75, 1]
    summaries, saved_predictions = {}, []
    full_priors = np.asarray([(y["train"] == label).mean() for label in LABELS])
    for method in ("bm25_nli", "dense_nli"):
        features = {}
        for s in ids:
            name = f"{s}_{method}_features.npy"
            if sha256(inputs / name) != original_manifest[name]:
                raise ValueError(f"feature checksum mismatch: {name}")
            features[s] = np.load(inputs / name, allow_pickle=False)
            if features[s].shape != (len(y[s]), 7) or not np.isfinite(features[s]).all():
                raise ValueError("invalid feature matrix")
        scores = {str(alpha): [] for alpha in grid}
        fold_audit = []
        for seed in SEEDS:
            oof = {str(alpha): np.empty(len(y["train"]), dtype=object) for alpha in grid}
            outer = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            for fold, (fit_ids, held_ids) in enumerate(
                outer.split(features["train"], y["train"], train_groups)
            ):
                model = fit_model(features["train"][fit_ids], y["train"][fit_ids], seed)
                probabilities = ordered_probabilities(model, features["train"][held_ids])
                priors = np.asarray([(y["train"][fit_ids] == label).mean() for label in LABELS])
                for alpha in grid:
                    positions = decision_indices(probabilities, priors, alpha)
                    oof[str(alpha)][held_ids] = np.asarray(LABELS)[positions]
                fold_audit.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "held_out_upstream_ids": [ids["train"][i] for i in held_ids],
                    }
                )
            for alpha in grid:
                scores[str(alpha)].append(
                    float(
                        f1_score(
                            y["train"],
                            oof[str(alpha)],
                            labels=list(LABELS),
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
            print(f"Training-only selection: {method}, seed {seed}", flush=True)
        means = {key: float(np.mean(values)) for key, values in scores.items()}
        selected = min(grid, key=lambda alpha: (-means[str(alpha)], alpha))
        results = []
        for seed in SEEDS:
            subset = [
                r for r in original_predictions if r["method"] == method and r["seed"] == seed
            ]
            by_id = {r["upstream_index"]: r for r in subset}
            if len(by_id) != len(ids["validation"]):
                raise ValueError("original predictions do not cover validation")
            subset = [by_id[i] for i in ids["validation"]]
            if [r["true_label"] for r in subset] != y["validation"].tolist():
                raise ValueError("original labels do not align with frozen split")
            p = np.asarray([r["probabilities"] for r in subset])
            chosen = np.asarray(LABELS)[decision_indices(p, full_priors, selected)]
            baseline = np.asarray(LABELS)[decision_indices(p, full_priors, 0)]
            if baseline.tolist() != [r["predicted_label"] for r in subset]:
                raise ValueError("original verdicts disagree with probability argmax")
            result = {
                "seed": seed,
                "baseline_macro_f1": float(
                    f1_score(
                        y["validation"],
                        baseline,
                        labels=list(LABELS),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "selected_macro_f1": float(
                    f1_score(
                        y["validation"],
                        chosen,
                        labels=list(LABELS),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "prediction_counts": {label: int((chosen == label).sum()) for label in LABELS},
            }
            results.append(result)
            saved_predictions.extend(
                {**r, "selected_label": str(chosen[i]), "selected_exponent": selected}
                for i, r in enumerate(subset)
            )
        summaries[method] = {
            "selected_exponent": selected,
            "oof_mean_macro_f1": means,
            "oof_seed_scores": scores,
            "results": results,
            "fold_audit": fold_audit,
        }
    output.mkdir(parents=True)
    summary = {
        "official_dev_records_used": 0,
        "scope": "post-diagnostic development analysis on oracle evidence excerpts",
        "selection": "mean training-only grouped OOF Macro-F1 across five seeds",
        "priors": full_priors.tolist(),
        "methods": summaries,
        "probabilities_changed": False,
    }
    write_json_atomic(output / "decision_summary.json", summary)
    write_json_atomic(output / "decision_predictions.json", saved_predictions)
    files = [
        inputs / "predictions.json",
        inputs / "input_manifest.json",
        inputs / "output_manifest.json",
    ] + [inputs / f"{s}_{method}_features.npy" for s in ids for method in ("bm25_nli", "dense_nli")]
    write_json_atomic(
        output / "decision_manifest.json",
        {
            "inputs": {p.name: sha256(p) for p in files},
            "outputs": {
                name: sha256(output / name)
                for name in ("decision_summary.json", "decision_predictions.json")
            },
        },
    )
    for method, result in summaries.items():
        print(
            method,
            "exponent",
            result["selected_exponent"],
            "mean Macro-F1",
            np.mean([r["selected_macro_f1"] for r in result["results"]]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
