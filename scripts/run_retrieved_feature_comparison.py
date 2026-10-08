"""Compare cached seven/eighteen-feature BM25 and dense excerpt classifiers."""

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
from apv_rag.metrics import classification_metrics
from apv_rag.nli_comparison import pool_corpus
from apv_rag.retrieved_representation import audit_features
from apv_rag.splits import build_connected_groups, sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    original = root / "artifacts/retrieval_nli_comparison"
    output = root / "artifacts/retrieved_feature_comparison"
    if output.exists():
        raise FileExistsError("refusing to overwrite retrieved-feature comparison")
    hashes = json.loads((original / "output_manifest.json").read_text())
    inputs = {}

    def checked(name):
        path = original / name
        if sha256(path) != hashes[name]:
            raise ValueError(f"original checksum mismatch: {name}")
        inputs[name] = sha256(path)
        return path

    metadata = json.loads(checked("input_manifest.json").read_text())
    for package in ("numpy", "scikit-learn"):
        if version(package) != metadata["packages"][package]:
            raise ValueError(f"original package version required: {package}")
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    if sha256(source) != SPLITS["train"].sha256 or sha256(source) != metadata["source_sha256"]:
        raise ValueError("original training source checksum mismatch")
    records = json.loads(source.read_text(encoding="utf-8"))
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    ids, truth, features = {}, {}, {m: {} for m in ("bm25_nli", "dense_nli")}
    for name in ("train", "validation"):
        path = split / f"{name}_indices.json"
        if sha256(path) != metadata[f"{name}_indices_sha256"]:
            raise ValueError(f"split checksum mismatch: {name}")
        inputs[path.name] = sha256(path)
        ids[name] = json.loads(path.read_text())
        rows = [records[i] for i in ids[name]]
        truth[name] = [r["label"] for r in rows]
        documents = json.loads(checked(f"{name}_corpus.json").read_text(encoding="utf-8"))
        if documents != pool_corpus(rows):
            raise ValueError("original corpus differs from reconstructed split-local corpus")
        for method in features:
            audit = json.loads(checked(f"{name}_{method}_retrieval_audit.json").read_text())
            if len(audit) != len(rows) or [r["record_position"] for r in audit] != list(
                range(len(rows))
            ):
                raise ValueError("retrieval audit record alignment mismatch")
            x = np.asarray(
                [
                    audit_features(row["claim"], documents, a)
                    for row, a in zip(rows, audit, strict=True)
                ]
            )
            base = np.load(checked(f"{name}_{method}_features.npy"), allow_pickle=False)
            if (
                x.shape != (len(rows), 18)
                or not np.isfinite(x).all()
                or not np.allclose(x[:, :7], base)
            ):
                raise ValueError("expanded retrieved features fail base reproduction")
            features[method][name] = x
    groups, _ = build_connected_groups(records)
    train_ids, val_ids = set(ids["train"]), set(ids["validation"])
    if train_ids & val_ids or any(set(g) & train_ids and set(g) & val_ids for g in groups):
        raise ValueError("connected-group train/validation leakage")
    protocol_path = root / "artifacts/frozen_heldout_protocol/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    priors = [(np.asarray(truth["train"]) == label).mean() for label in LABELS]
    if not np.allclose(priors, protocol["training_priors"]):
        raise ValueError("training prior mismatch")
    original_rows = json.loads(checked("predictions.json").read_text())
    lookup = {(r["method"], r["seed"], r["upstream_index"]): r for r in original_rows}
    expected = {(m, s, i) for m in features for s in protocol["seeds"] for i in ids["validation"]}
    if len(original_rows) != len(expected) or set(lookup) != expected:
        raise ValueError("original prediction pairing mismatch")
    results, saved = [], []
    for method, matrices in features.items():
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
                model.fit(matrices["train"][:, :width], truth["train"])
                p = model.predict_proba(matrices["validation"][:, :width])
                p = p[:, [list(model.classes_).index(label) for label in LABELS]]
                if width == 7:
                    earlier = [lookup[(method, seed, i)] for i in ids["validation"]]
                    if [r["true_label"] for r in earlier] != truth["validation"] or not np.allclose(
                        p, [r["probabilities"] for r in earlier], atol=1e-7, rtol=1e-6
                    ):
                        raise ValueError("original retrieved probabilities not reproduced")
                raw = np.asarray(LABELS)[p.argmax(axis=1)]
                raw_metrics = classification_metrics(
                    truth["validation"], raw, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
                )
                for rule, alpha in (("argmax", 0), ("prior_adjusted", 1)):
                    pred = np.asarray(LABELS)[decision_indices(p, priors, alpha)]
                    metrics = classification_metrics(
                        truth["validation"], pred, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
                    )
                    if alpha:
                        metrics["expected_calibration_error"] = raw_metrics[
                            "expected_calibration_error"
                        ]
                        metrics.pop("risk_coverage")
                    results.append(
                        {
                            "method": method,
                            "representation": representation,
                            "seed": seed,
                            "rule": rule,
                            "metrics": metrics,
                        }
                    )
                    for j, index in enumerate(ids["validation"]):
                        saved.append(
                            {
                                "method": method,
                                "representation": representation,
                                "seed": seed,
                                "rule": rule,
                                "upstream_index": index,
                                "true_label": truth["validation"][j],
                                "predicted_label": str(pred[j]),
                                "probabilities": p[j].tolist(),
                            }
                        )
            print(f"Completed: {method} {representation}", flush=True)
    output.mkdir()
    write_json_atomic(
        output / "retrieved_feature_summary.json",
        {
            "status": "exploratory internal-validation representation comparison",
            "official_dev_records_read": 0,
            "training_claims": len(ids["train"]),
            "validation_claims": len(ids["validation"]),
            "seeds": protocol["seeds"],
            "feature_names": list(FEATURE_NAMES),
            "results": results,
            "scope": "split-local oracle-corpus excerpt retrieval; not open-web RAG",
            "limitations": [
                "corpora already contain benchmark-annotated evidence",
                "source and length features are proxies, not sufficiency judgments",
                "inner calibration is stratified, not grouped",
                "observed official dev cannot confirm this revised representation",
            ],
        },
    )
    write_json_atomic(output / "predictions.json", saved)
    for method, matrices in features.items():
        for name, x in matrices.items():
            np.save(output / f"{name}_{method}_expanded_features.npy", x)
    write_json_atomic(
        output / "input_manifest.json",
        {
            "original_inputs": inputs,
            "train_source_sha256": sha256(source),
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__)),
            "feature_code_sha256": sha256(root / "src/apv_rag/evidence_representation.py"),
            "audit_code_sha256": sha256(root / "src/apv_rag/retrieved_representation.py"),
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.iterdir() if p.name != "output_manifest.json"},
    )
    print(output / "retrieved_feature_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
