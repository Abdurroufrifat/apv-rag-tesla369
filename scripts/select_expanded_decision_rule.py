"""Select prior adjustment on grouped training OOF predictions only."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from run_cached_decision_comparison import fit_model, ordered_probabilities
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from apv_rag.averitec import SPLITS
from apv_rag.decision_rules import decision_indices
from apv_rag.evidence_baseline import LABELS
from apv_rag.metrics import classification_metrics
from apv_rag.nli_comparison import SEEDS
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = root / "artifacts/retrieved_feature_comparison"
    output = root / "artifacts/expanded_training_rule_selection"
    if output.exists():
        raise FileExistsError("refusing to overwrite completed rule selection")
    manifest = json.loads((inputs / "output_manifest.json").read_text())
    original = root / "artifacts/retrieval_nli_comparison"
    original_hashes = json.loads((original / "output_manifest.json").read_text())
    metadata_path = original / "input_manifest.json"
    if sha256(metadata_path) != original_hashes[metadata_path.name]:
        raise ValueError("original metadata checksum mismatch")
    metadata = json.loads(metadata_path.read_text())
    for name in ("numpy", "scikit-learn"):
        if version(name) != metadata["packages"][name]:
            raise ValueError(f"original package version required: {name}")
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    if sha256(source) != SPLITS["train"].sha256:
        raise ValueError("training source checksum mismatch")
    records = json.loads(source.read_text(encoding="utf-8"))
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    ids, y = {}, {}
    for name in ("train", "validation"):
        path = split / f"{name}_indices.json"
        if sha256(path) != metadata[f"{name}_indices_sha256"]:
            raise ValueError(f"split checksum mismatch: {name}")
        ids[name] = json.loads(path.read_text())
        y[name] = np.asarray([records[i]["label"] for i in ids[name]])
    group_path = split / "group_assignments.json"
    groups = json.loads(group_path.read_text())
    train_groups = np.asarray([groups[str(i)] for i in ids["train"]])
    if set(train_groups) & {groups[str(i)] for i in ids["validation"]}:
        raise ValueError("group leakage across internal split")
    grid = [0, 0.25, 0.5, 0.75, 1]
    priors = [(y["train"] == label).mean() for label in LABELS]
    results, saved, inputs_sha = {}, [], {}
    for method in ("bm25_nli", "dense_nli"):
        features = {}
        for name in ids:
            filename = f"{name}_{method}_expanded_features.npy"
            if sha256(inputs / filename) != manifest[filename]:
                raise ValueError(f"feature checksum mismatch: {filename}")
            inputs_sha[filename] = sha256(inputs / filename)
            features[name] = np.load(inputs / filename, allow_pickle=False)
            if (
                features[name].shape != (len(ids[name]), 18)
                or not np.isfinite(features[name]).all()
            ):
                raise ValueError("invalid expanded feature matrix")
        scores = {str(alpha): [] for alpha in grid}
        folds = []
        for seed in SEEDS:
            oof = {str(alpha): np.empty(len(y["train"]), dtype=object) for alpha in grid}
            cv = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
            for fold, (fit_ids, held_ids) in enumerate(
                cv.split(features["train"], y["train"], train_groups)
            ):
                if set(train_groups[fit_ids]) & set(train_groups[held_ids]):
                    raise ValueError("OOF group leakage")
                model = fit_model(features["train"][fit_ids], y["train"][fit_ids], seed)
                p = ordered_probabilities(model, features["train"][held_ids])
                fold_priors = [(y["train"][fit_ids] == label).mean() for label in LABELS]
                for alpha in grid:
                    pred = np.asarray(LABELS)[decision_indices(p, fold_priors, alpha)]
                    oof[str(alpha)][held_ids] = pred
                folds.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "fit_count": len(fit_ids),
                        "held_count": len(held_ids),
                        "group_overlap": 0,
                    }
                )
            for alpha in grid:
                scores[str(alpha)].append(
                    float(
                        f1_score(
                            y["train"],
                            oof[str(alpha)],
                            labels=LABELS,
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
        means = {key: float(np.mean(value)) for key, value in scores.items()}
        selected = min(grid, key=lambda alpha: (-means[str(alpha)], alpha))
        # The exponent is selected before any validation probabilities are computed.
        seed_results = []
        for seed in SEEDS:
            model = fit_model(features["train"], y["train"], seed)
            p = ordered_probabilities(model, features["validation"])
            raw = np.asarray(LABELS)[p.argmax(axis=1)]
            raw_metrics = classification_metrics(
                y["validation"], raw, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
            )
            pred = np.asarray(LABELS)[decision_indices(p, priors, selected)]
            metrics = classification_metrics(
                y["validation"], pred, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
            )
            metrics["expected_calibration_error"] = raw_metrics["expected_calibration_error"]
            metrics.pop("risk_coverage")
            seed_results.append(
                {"seed": seed, "argmax_metrics": raw_metrics, "selected_metrics": metrics}
            )
            for j, index in enumerate(ids["validation"]):
                saved.append(
                    {
                        "method": method,
                        "seed": seed,
                        "upstream_index": index,
                        "true_label": str(y["validation"][j]),
                        "argmax_label": str(raw[j]),
                        "selected_label": str(pred[j]),
                        "probabilities": p[j].tolist(),
                    }
                )
        results[method] = {
            "selected_exponent": selected,
            "training_oof_scores": scores,
            "mean_training_oof_macro_f1": means,
            "fold_audit": folds,
            "validation_results": seed_results,
        }
        print(f"Selected from training only: {method} alpha={selected}", flush=True)
    output.mkdir()
    write_json_atomic(
        output / "selection_summary.json",
        {
            "status": "development rule selection; not new held-out confirmation",
            "official_dev_records_read": 0,
            "selection_data": "grouped training OOF only",
            "tie_break": "smaller exponent",
            "grid": grid,
            "seeds": list(SEEDS),
            "results": results,
            "limitations": [
                "retrieval corpus and cached features fixed across outer OOF folds",
                "not fully nested retrieval evaluation",
                "inner calibration folds are stratified",
                "internal validation has been repeatedly observed",
            ],
        },
    )
    write_json_atomic(output / "predictions.json", saved)
    write_json_atomic(
        output / "input_manifest.json",
        {
            "features": inputs_sha,
            "train_source_sha256": sha256(source),
            "groups_sha256": sha256(group_path),
            "script_sha256": sha256(Path(__file__)),
            "fitting_code_sha256": sha256(root / "scripts/run_cached_decision_comparison.py"),
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.iterdir() if p.name != "output_manifest.json"},
    )
    print(output / "selection_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
