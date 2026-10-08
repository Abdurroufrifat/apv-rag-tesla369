"""Training-only OOF threshold selection for an exploratory benchmark-label proxy."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from apv_rag.averitec import SPLITS
from apv_rag.evidence_representation import FEATURE_NAMES
from apv_rag.nli_comparison import SEEDS
from apv_rag.splits import sha256, write_json_atomic
from apv_rag.sufficiency_proxy import proxy_labels, select_threshold


def fit(x, y, seed):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    return model.fit(x, y)


def metrics(y, probabilities, threshold):
    predicted = probabilities >= threshold
    return {
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "positive_average_precision": float(average_precision_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
        "positive_prediction_fraction": float(predicted.mean()),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/sufficiency_proxy"
    if output.exists():
        raise FileExistsError("Refusing to overwrite proxy run")
    inputs = root / "artifacts/retrieved_feature_comparison"
    if not (inputs / "output_manifest.json").exists():
        raise FileNotFoundError("Existing retrieved_feature_comparison features required")
    manifest = json.loads((inputs / "output_manifest.json").read_text())
    original = root / "artifacts/retrieval_nli_comparison"
    original_hashes = json.loads((original / "output_manifest.json").read_text())
    metadata_path = original / "input_manifest.json"
    if sha256(metadata_path) != original_hashes[metadata_path.name]:
        raise ValueError("Original metadata checksum mismatch")
    metadata = json.loads(metadata_path.read_text())
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    if sha256(source) != SPLITS["train"].sha256:
        raise ValueError("Training source checksum mismatch")
    records = json.loads(source.read_text(encoding="utf-8"))
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    ids, y, digests = {}, {}, {"source": sha256(source)}
    for name in ("train", "validation"):
        path = split / f"{name}_indices.json"
        if sha256(path) != metadata[f"{name}_indices_sha256"]:
            raise ValueError("Frozen split mismatch")
        digests[path.name] = sha256(path)
        ids[name] = json.loads(path.read_text())
        y[name] = proxy_labels([records[i]["label"] for i in ids[name]])
    group_path = split / "group_assignments.json"
    group_map = json.loads(group_path.read_text())
    groups = {name: np.array([group_map[str(i)] for i in indices]) for name, indices in ids.items()}
    if set(groups["train"]) & set(groups["validation"]):
        raise ValueError("Group leakage")
    digests["group_assignments.json"] = sha256(group_path)
    predictions, results = [], []
    for method in ("bm25_nli", "dense_nli"):
        features = {}
        for name in ids:
            filename = f"{name}_{method}_expanded_features.npy"
            path = inputs / filename
            if sha256(path) != manifest[filename]:
                raise ValueError(f"Feature checksum mismatch: {filename}")
            digests[filename] = sha256(path)
            features[name] = np.load(path, allow_pickle=False)
            if (
                features[name].shape != (len(ids[name]), 18)
                or not np.isfinite(features[name]).all()
            ):
                raise ValueError("Invalid feature matrix")
        for representation, columns in (("base7", list(range(7))), ("expanded18", list(range(18)))):
            x = {n: matrix[:, columns] for n, matrix in features.items()}
            for seed in SEEDS:
                oof = np.zeros(len(y["train"]))
                cv = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
                for fit_ids, held_ids in cv.split(x["train"], y["train"], groups["train"]):
                    if set(groups["train"][fit_ids]) & set(groups["train"][held_ids]):
                        raise ValueError("OOF group leakage")
                    model = fit(x["train"][fit_ids], y["train"][fit_ids], seed)
                    oof[held_ids] = model.predict_proba(x["train"][held_ids])[:, 1]
                threshold, grid = select_threshold(y["train"], oof)
                model = fit(x["train"], y["train"], seed)
                probabilities = model.predict_proba(x["validation"])[:, 1]
                majority = int(y["train"].mean() >= 0.5)
                prior = np.full(len(probabilities), y["train"].mean())
                results.append(
                    {
                        "method": method,
                        "representation": representation,
                        "seed": seed,
                        "training_oof_selected_threshold": threshold,
                        "training_oof_grid": grid,
                        "validation_metrics": metrics(y["validation"], probabilities, threshold),
                        "fixed_half_metrics": metrics(y["validation"], probabilities, 0.5),
                        "training_prior_baseline": metrics(y["validation"], prior, 0.5),
                        "training_majority_label": majority,
                        "model_coefficients": model[-1].coef_[0].tolist(),
                        "model_intercept": model[-1].intercept_.tolist(),
                        "scaler_mean": model[0].mean_.tolist(),
                        "scaler_scale": model[0].scale_.tolist(),
                    }
                )
                for position, index in enumerate(ids["validation"]):
                    predictions.append(
                        {
                            "method": method,
                            "representation": representation,
                            "seed": seed,
                            "claim_index": index,
                            "group": str(groups["validation"][position]),
                            "true_proxy_label": int(y["validation"][position]),
                            "probability_non_nei": float(probabilities[position]),
                            "threshold": threshold,
                            "predicted_proxy_label": int(probabilities[position] >= threshold),
                        }
                    )
                print(method, representation, seed, flush=True)
    output.mkdir()
    write_json_atomic(
        output / "input_manifest.json",
        {
            "inputs_sha256": digests,
            "packages": {p: version(p) for p in ("numpy", "scikit-learn")},
            "code_sha256": {
                n: sha256(root / n)
                for n in ("scripts/run_sufficiency_proxy.py", "src/apv_rag/sufficiency_proxy.py")
            },
            "protocol_sha256": sha256(root / "docs/SUFFICIENCY_PROXY.md"),
            "feature_names": list(FEATURE_NAMES),
            "seeds": list(SEEDS),
        },
    )
    write_json_atomic(
        output / "proxy_summary.json",
        {
            "scope": (
                "exploratory internal benchmark-label proxy; "
                "not evidence sufficiency ground truth"
            ),
            "official_dev_records_used": 0,
            "target": "non-NEI = 1, NEI = 0; conflicting evidence included in non-NEI",
            "train_rows": len(ids["train"]),
            "validation_rows": len(ids["validation"]),
            "training_positive_fraction": float(y["train"].mean()),
            "validation_positive_fraction": float(y["validation"].mean()),
            "results": results,
        },
    )
    write_json_atomic(output / "predictions.json", predictions)
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.glob("*.json") if p.name != "output_manifest.json"},
    )
    print(output)


if __name__ == "__main__":
    main()
