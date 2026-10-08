"""Compare seven versus eighteen features on internal validation only."""

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
from apv_rag.evidence_representation import FEATURE_NAMES, expanded_features
from apv_rag.metrics import classification_metrics
from apv_rag.nli_comparison import pool_corpus
from apv_rag.splits import build_connected_groups, sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/expanded_evidence_comparison"
    if output.exists():
        raise FileExistsError("refusing to overwrite expanded comparison results")
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    if sha256(source) != SPLITS["train"].sha256:
        raise ValueError("upstream training source checksum mismatch")
    original = root / "artifacts/retrieval_nli_comparison"
    original_hashes = json.loads((original / "output_manifest.json").read_text())
    metadata_path = original / "input_manifest.json"
    if sha256(metadata_path) != original_hashes[metadata_path.name]:
        raise ValueError("original metadata checksum mismatch")
    metadata = json.loads(metadata_path.read_text())
    for package in ("numpy", "scikit-learn"):
        if version(package) != metadata["packages"][package]:
            raise ValueError(f"original package version required: {package}")
    records = json.loads(source.read_text(encoding="utf-8"))
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    ids, features, y, input_hashes = {}, {}, {}, {"train.json": sha256(source)}
    paths = {
        "train": (
            root / "artifacts/own_excerpt_refit",
            "training_excerpt_audit.json",
            "own_train_features.npy",
        ),
        "validation": (
            root / "artifacts/own_excerpt_comparison",
            "own_excerpt_audit.json",
            "own_validation_features.npy",
        ),
    }
    for name, (directory, audit_name, feature_name) in paths.items():
        index_path = split / f"{name}_indices.json"
        if sha256(index_path) != metadata[f"{name}_indices_sha256"]:
            raise ValueError(f"split checksum mismatch: {name}")
        ids[name] = json.loads(index_path.read_text())
        input_hashes[str(index_path.relative_to(root))] = sha256(index_path)
        manifest = json.loads((directory / "output_manifest.json").read_text())
        for filename in (audit_name, feature_name):
            path = directory / filename
            if sha256(path) != manifest[filename]:
                raise ValueError(f"audit input checksum mismatch: {filename}")
            input_hashes[str(path.relative_to(root))] = sha256(path)
        audit = json.loads((directory / audit_name).read_text(encoding="utf-8"))
        if len(audit) != len(ids[name]) or [row["upstream_index"] for row in audit] != ids[name]:
            raise ValueError(f"audit record alignment mismatch: {name}")
        x = []
        for index, row in zip(ids[name], audit, strict=True):
            docs = pool_corpus([records[index]])[:5]
            premises = [d["text"] for d in docs]
            if premises != row["premises"]:
                raise ValueError("audit excerpts differ from original source order")
            x.append(
                expanded_features(
                    records[index]["claim"],
                    premises,
                    [d["source_url"] for d in docs],
                    row["nli_probabilities"],
                )
            )
        x = np.asarray(x)
        base = np.load(directory / feature_name, allow_pickle=False)
        if (
            x.shape != (len(ids[name]), 18)
            or not np.isfinite(x).all()
            or not np.allclose(x[:, :7], base)
        ):
            raise ValueError("expanded features fail base-feature reproduction")
        features[name] = x
        y[name] = [records[i]["label"] for i in ids[name]]
    groups, _ = build_connected_groups(records)
    train_ids, val_ids = set(ids["train"]), set(ids["validation"])
    if train_ids & val_ids or any(set(g) & train_ids and set(g) & val_ids for g in groups):
        raise ValueError("train/validation leakage")
    protocol_path = root / "artifacts/frozen_heldout_protocol/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    priors = [(np.asarray(y["train"]) == label).mean() for label in LABELS]
    if not np.allclose(priors, protocol["training_priors"]):
        raise ValueError("training prior mismatch")
    prior_dir = root / "artifacts/own_excerpt_refit"
    prior_manifest = json.loads((prior_dir / "output_manifest.json").read_text())
    prediction_path = prior_dir / "predictions.json"
    if sha256(prediction_path) != prior_manifest["predictions.json"]:
        raise ValueError("prior prediction checksum mismatch")
    old = json.loads(prediction_path.read_text())
    lookup = {(r["seed"], r["rule"], r["upstream_index"]): r for r in old}
    expected = {
        (s, rule, i)
        for s in protocol["seeds"]
        for rule in ("argmax", "selected")
        for i in ids["validation"]
    }
    if len(old) != len(expected) or set(lookup) != expected:
        raise ValueError("prior refit prediction pairing mismatch")
    input_hashes["own_refit_predictions.json"] = sha256(prediction_path)
    results, predictions = [], []
    for representation, width in (("base_7", 7), ("expanded_18", 18)):
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
            model.fit(features["train"][:, :width], y["train"])
            p = model.predict_proba(features["validation"][:, :width])
            p = p[:, [list(model.classes_).index(label) for label in LABELS]]
            if width == 7:
                earlier = [lookup[(seed, "argmax", i)] for i in ids["validation"]]
                if [r["true_label"] for r in earlier] != y["validation"] or not np.allclose(
                    p, [r["probabilities"] for r in earlier], atol=1e-7, rtol=1e-6
                ):
                    raise ValueError(
                        "base representation does not reproduce prior refit probabilities"
                    )
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
                results.append(
                    {
                        "representation": representation,
                        "seed": seed,
                        "rule": rule,
                        "metrics": metrics,
                    }
                )
                for j, index in enumerate(ids["validation"]):
                    predictions.append(
                        {
                            "representation": representation,
                            "seed": seed,
                            "rule": rule,
                            "upstream_index": index,
                            "true_label": y["validation"][j],
                            "predicted_label": str(pred[j]),
                            "probabilities": p[j].tolist(),
                        }
                    )
    summary = {
        "status": "exploratory internal-validation representation comparison",
        "official_dev_records_read": 0,
        "training_claims": len(ids["train"]),
        "validation_claims": len(ids["validation"]),
        "seeds": protocol["seeds"],
        "feature_names": list(FEATURE_NAMES),
        "results": results,
        "interpretation": "own-excerpt oracle setting; no new confirmation or full RAG score",
        "limitations": [
            "lexical overlap and host counts do not establish evidence sufficiency",
            "source-count features may reflect benchmark annotation practices",
            "inner calibration folds remain stratified, not grouped",
            "feature design follows observed development errors",
        ],
    }
    output.mkdir()
    write_json_atomic(
        output / "input_manifest.json",
        {
            "inputs": input_hashes,
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__)),
            "feature_code_sha256": sha256(root / "src/apv_rag/evidence_representation.py"),
        },
    )
    write_json_atomic(output / "expanded_evidence_summary.json", summary)
    write_json_atomic(output / "predictions.json", predictions)
    for name, x in features.items():
        np.save(output / f"{name}_expanded_features.npy", x)
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.iterdir() if p.name != "output_manifest.json"},
    )
    print(f"Completed without neural inference: {output / 'expanded_evidence_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
