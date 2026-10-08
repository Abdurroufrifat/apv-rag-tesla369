"""Fixed-group ablations of cached own-excerpt features; never open dev."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from apv_rag.averitec import SPLITS
from apv_rag.decision_rules import decision_indices
from apv_rag.evidence_baseline import LABELS
from apv_rag.evidence_representation import FEATURE_NAMES
from apv_rag.feature_ablation import GROUPS, ablation_columns
from apv_rag.metrics import classification_metrics
from apv_rag.splits import build_connected_groups, sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/feature_group_ablation"
    if output.exists():
        raise FileExistsError("refusing to overwrite completed ablation")
    prior = root / "artifacts/expanded_evidence_comparison"
    hashes = json.loads((prior / "output_manifest.json").read_text())
    required = (
        "train_expanded_features.npy",
        "validation_expanded_features.npy",
        "expanded_evidence_summary.json",
        "predictions.json",
        "input_manifest.json",
    )
    for name in required:
        if sha256(prior / name) != hashes[name]:
            raise ValueError(f"cached input checksum mismatch: {name}")
    prior_summary = json.loads((prior / "expanded_evidence_summary.json").read_text())
    if (
        prior_summary["feature_names"] != list(FEATURE_NAMES)
        or prior_summary["official_dev_records_read"] != 0
    ):
        raise ValueError("prior representation identity mismatch")
    previous_inputs = json.loads((prior / "input_manifest.json").read_text())
    previous_paths = {
        name.replace("\\", "/"): value for name, value in previous_inputs["inputs"].items()
    }
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    if sha256(source) != SPLITS["train"].sha256:
        raise ValueError("training source checksum mismatch")
    metadata_path = root / "artifacts/retrieval_nli_comparison/input_manifest.json"
    original_hashes = json.loads(metadata_path.with_name("output_manifest.json").read_text())
    if sha256(metadata_path) != original_hashes[metadata_path.name]:
        raise ValueError("original metadata checksum mismatch")
    metadata = json.loads(metadata_path.read_text())
    for package in ("numpy", "scikit-learn"):
        if version(package) != metadata["packages"][package]:
            raise ValueError(f"original package version required: {package}")
    records = json.loads(source.read_text(encoding="utf-8"))
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    features, ids, y = {}, {}, {}
    for name in ("train", "validation"):
        path = split / f"{name}_indices.json"
        if (
            sha256(path) != metadata[f"{name}_indices_sha256"]
            or sha256(path) != previous_paths[str(path.relative_to(root)).replace("\\", "/")]
        ):
            raise ValueError(f"split checksum mismatch: {name}")
        ids[name] = json.loads(path.read_text())
        features[name] = np.load(prior / f"{name}_expanded_features.npy", allow_pickle=False)
        if features[name].shape != (len(ids[name]), 18) or not np.isfinite(features[name]).all():
            raise ValueError("invalid cached expanded feature matrix")
        y[name] = [records[i]["label"] for i in ids[name]]
    train_ids, val_ids = set(ids["train"]), set(ids["validation"])
    groups, _ = build_connected_groups(records)
    if train_ids & val_ids or any(set(g) & train_ids and set(g) & val_ids for g in groups):
        raise ValueError("connected-group train/validation leakage")
    protocol_path = root / "artifacts/frozen_heldout_protocol/protocol.json"
    if sha256(protocol_path) != previous_inputs["protocol_sha256"]:
        raise ValueError("protocol identity changed")
    protocol = json.loads(protocol_path.read_text())
    priors = [(np.asarray(y["train"]) == label).mean() for label in LABELS]
    if not np.allclose(priors, protocol["training_priors"]):
        raise ValueError("training prior mismatch")
    earlier = json.loads((prior / "predictions.json").read_text())
    lookup = {(r["representation"], r["seed"], r["rule"], r["upstream_index"]): r for r in earlier}
    expected = {
        (rep, seed, rule, i)
        for rep in ("base_7", "expanded_18")
        for seed in protocol["seeds"]
        for rule in ("argmax", "prior_adjusted")
        for i in ids["validation"]
    }
    if len(earlier) != len(expected) or set(lookup) != expected:
        raise ValueError("prior prediction pairing mismatch")
    results, saved = [], []
    variants = ablation_columns()
    for variant, columns in variants.items():
        for seed in protocol["seeds"]:
            model = CalibratedClassifierCV(
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=1, class_weight="balanced", max_iter=2000, random_state=seed
                    ),
                ),
                method="sigmoid",
                cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            )
            model.fit(features["train"][:, columns], y["train"])
            p = model.predict_proba(features["validation"][:, columns])
            p = p[:, [list(model.classes_).index(label) for label in LABELS]]
            if variant in ("base_7", "full_18"):
                rep = "base_7" if variant == "base_7" else "expanded_18"
                old = [lookup[(rep, seed, "argmax", i)] for i in ids["validation"]]
                if [r["true_label"] for r in old] != y["validation"] or not np.allclose(
                    p, [r["probabilities"] for r in old], atol=1e-7, rtol=1e-6
                ):
                    raise ValueError(f"prior probabilities not reproduced: {variant}")
            raw = np.asarray(LABELS)[p.argmax(axis=1)]
            raw_metrics = classification_metrics(
                y["validation"], raw, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
            )
            for rule, alpha in (("argmax", 0), ("prior_adjusted", 1)):
                pred = np.asarray(LABELS)[decision_indices(p, priors, alpha)]
                metrics = classification_metrics(
                    y["validation"], pred, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
                )
                if alpha:
                    metrics["expected_calibration_error"] = raw_metrics[
                        "expected_calibration_error"
                    ]
                    metrics.pop("risk_coverage")
                results.append({"variant": variant, "seed": seed, "rule": rule, "metrics": metrics})
                for j, i in enumerate(ids["validation"]):
                    saved.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "rule": rule,
                            "upstream_index": i,
                            "true_label": y["validation"][j],
                            "predicted_label": str(pred[j]),
                            "probabilities": p[j].tolist(),
                        }
                    )
        print(f"Completed ablation: {variant}", flush=True)
    summary = {
        "status": "descriptive development ablation; no selected winner or confirmation",
        "official_dev_records_read": 0,
        "training_claims": len(ids["train"]),
        "validation_claims": len(ids["validation"]),
        "seeds": protocol["seeds"],
        "groups": {name: [FEATURE_NAMES[i] for i in columns] for name, columns in GROUPS.items()},
        "variants": {
            name: [FEATURE_NAMES[i] for i in columns] for name, columns in variants.items()
        },
        "results": results,
        "limitations": [
            "group effects depend on other retained features",
            "source/length proxies may reflect benchmark annotation practices",
            "inner calibration is stratified, not grouped",
            "repeated internal-validation analysis is exploratory",
        ],
    }
    output.mkdir()
    write_json_atomic(output / "ablation_summary.json", summary)
    write_json_atomic(output / "predictions.json", saved)
    write_json_atomic(
        output / "input_manifest.json",
        {
            "prior_inputs": {name: hashes[name] for name in required},
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__)),
            "group_code_sha256": sha256(root / "src/apv_rag/feature_ablation.py"),
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.iterdir() if p.name != "output_manifest.json"},
    )
    print(output / "ablation_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
