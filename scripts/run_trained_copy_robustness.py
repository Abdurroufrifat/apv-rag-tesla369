"""Fit/replay two fixed classifiers with clean versus copied-evidence training."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apv_rag.averitec import SPLITS  # noqa: E402
from apv_rag.evidence_baseline import LABELS, load_split_records  # noqa: E402
from apv_rag.repetition_stress import inject_derivative_copies  # noqa: E402
from apv_rag.splits import sha256, write_json_atomic  # noqa: E402
from apv_rag.trained_copy_robustness import (  # noqa: E402
    build_model, model_records, training_variants,
)

OUT = ROOT / "artifacts/trained_copy_robustness_v1"
COPIES = (0, 1, 5, 10, 25)
SOURCE = ROOT / "data/external/averitec/official_7c62d1e/train.json"
SPLIT = ROOT / "data/processed/averitec/phase2b_split_v0_1"
CONFIG = ROOT / "config/provenance_model.yaml"
INPUTS = (SOURCE, CONFIG, SPLIT / "train_indices.json", SPLIT / "validation_indices.json",
          SPLIT / "group_assignments.json", SPLIT / "split_manifest.json")
CODE = (ROOT / "src/apv_rag/trained_copy_robustness.py", Path(__file__).resolve(),
        ROOT / "src/apv_rag/repetition_stress.py", ROOT / "src/apv_rag/provenance_model.py",
        ROOT / "src/apv_rag/disagreement_model.py", ROOT / "src/apv_rag/evidence_baseline.py")


def load():
    if sha256(SOURCE) != SPLITS["train"].sha256:
        raise ValueError("Pinned training source differs")
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    train = json.loads(INPUTS[2].read_text(encoding="utf-8"))
    validation = json.loads(INPUTS[3].read_text(encoding="utf-8"))
    if len(train) != 2458 or len(validation) != 609 or set(train) & set(validation):
        raise ValueError("Original split differs")
    # Rebuild existing claim/article connected groups to verify the boundary.
    from apv_rag.splits import build_connected_groups
    groups, _ = build_connected_groups(records)
    a, b = set(train), set(validation)
    if any(a.intersection(group) and b.intersection(group) for group in groups):
        raise ValueError("Connected claim/article group crosses the split")
    return load_split_records(records, train), load_split_records(records, validation), validation


def evaluate(models, validation, indices):
    result, summary = [], {}
    truth = [row["label"] for row in validation]
    for name, model in models.items():
        by_count = {}
        columns = [list(model.classes_).index(label) for label in LABELS]
        for count in COPIES:
            data = model_records([inject_derivative_copies(row, count) for row in validation])
            probabilities = model.predict_proba(data)[:, columns]
            predicted = [LABELS[int(i)] for i in probabilities.argmax(axis=1)]
            by_count[count] = probabilities
            shift = probabilities[:, 0] - by_count[0][:, 0]
            summary[f"{name}:copies_{count}"] = {
                "macro_f1": float(f1_score(truth, predicted, labels=list(LABELS), average="macro", zero_division=0)),
                "accuracy": sum(t == p for t, p in zip(truth, predicted, strict=True)) / len(truth),
                "mean_absolute_support_probability_shift": float(np.abs(shift).mean()),
                "mean_signed_support_probability_shift": float(shift.mean()),
            }
            result.extend({"model": name, "copies": count, "upstream_index": index,
                           "true_label": label, "probabilities": probs.tolist()}
                          for index, label, probs in zip(indices, truth, probabilities, strict=True))
    return result, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    train, validation, indices = load()
    with threadpool_limits(limits=1):
        if args.verify:
            receipt = json.loads((OUT / "receipt.json").read_text(encoding="utf-8"))
            for group in ("inputs", "code", "outputs"):
                for name, expected in receipt[group].items():
                    path = OUT / name if group == "outputs" else ROOT / name
                    if sha256(path) != expected:
                        raise ValueError(f"{group} SHA mismatch: {name}")
            models = {name: joblib.load(OUT / f"{name}.joblib") for name in ("control", "copy_augmented")}
        else:
            if OUT.exists():
                raise FileExistsError("Output exists; use --verify")
            config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            models = {}
            started = time.monotonic()
            for name, counts in (("control", (0,)), ("copy_augmented", (0, 5, 10))):
                print("Fitting", name, "on training-side variants", counts, flush=True)
                data, labels, weights, _ = training_variants(train, counts)
                model = build_model(config)
                model.fit(data, labels, classifier__sample_weight=weights)
                models[name] = model
            OUT.mkdir()
            for name, model in models.items():
                joblib.dump(model, OUT / f"{name}.joblib", compress=3)
            write_json_atomic(OUT / "training_receipt.json", {
                "seconds": time.monotonic() - started, "train_parents": len(train),
                "control_rows": len(train), "augmented_rows": len(train) * 3,
                "parent_total_training_weight": 1.0, "official_dev_records_used": 0,
                "packages": {name: version(name) for name in ("numpy", "scikit-learn", "scipy", "joblib")},
                "calibrated": False, "random_seed": 369,
                "scope": "Single-seed exploratory observed internal validation; no deployment selection"})
        predictions, summary = evaluate(models, validation, indices)
        if args.verify:
            saved = json.loads((OUT / "predictions.json").read_text(encoding="utf-8"))
            if len(saved) != len(predictions):
                raise ValueError("Prediction count differs")
            for old, new in zip(saved, predictions, strict=True):
                if any(old[k] != new[k] for k in ("model", "copies", "upstream_index", "true_label")) or not np.allclose(old["probabilities"], new["probabilities"], rtol=0, atol=1e-12):
                    raise ValueError("Saved model prediction replay differs")
            old = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
            if any(abs(old[k][metric] - value) > 1e-12 for k, row in summary.items() for metric, value in row.items()):
                raise ValueError("Saved metric replay differs")
        else:
            write_json_atomic(OUT / "predictions.json", predictions)
            write_json_atomic(OUT / "summary.json", summary)
            write_json_atomic(OUT / "receipt.json", {
                "inputs": {p.relative_to(ROOT).as_posix(): sha256(p) for p in INPUTS},
                "code": {p.relative_to(ROOT).as_posix(): sha256(p) for p in CODE},
                "outputs": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
