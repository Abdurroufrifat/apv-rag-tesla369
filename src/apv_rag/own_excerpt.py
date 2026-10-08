"""Fixed-aggregator input substitution on internal validation only."""

import json
import sqlite3
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
from apv_rag.heldout import cache_receipt, check_features
from apv_rag.metrics import classification_metrics
from apv_rag.nli_comparison import _fingerprint, _model_files, _pairs, nli_features, pool_corpus
from apv_rag.splits import sha256, write_json_atomic


def own_premises(record):
    return [d["text"] for d in pool_corpus([record])[:5]]


def run_own_excerpt(root, batch_size=8, threads=4):
    root = Path(root).resolve()
    if batch_size < 1 or threads < 1:
        raise ValueError("batch size and threads must be positive")
    original = root / "artifacts/retrieval_nli_comparison"
    protocol = json.loads((root / "artifacts/frozen_heldout_protocol/protocol.json").read_text())
    hashes = json.loads((original / "output_manifest.json").read_text())
    metadata = json.loads((original / "input_manifest.json").read_text())
    output = root / "artifacts/own_excerpt_comparison"
    if (output / "output_manifest.json").exists():
        raise FileExistsError("completed own-excerpt comparison exists; refusing overwrite")
    required = ["input_manifest.json", "predictions.json"] + [
        f"{s}_{m}_features.npy" for s in ("train", "validation") for m in protocol["methods"]
    ]
    for name in required:
        if sha256(original / name) != hashes[name]:
            raise ValueError(f"original input checksum mismatch: {name}")
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    if sha256(source) != SPLITS["train"].sha256 or sha256(source) != metadata["source_sha256"]:
        raise ValueError("training source checksum mismatch")
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    ids = {}
    for name in ("train", "validation"):
        path = split / f"{name}_indices.json"
        if sha256(path) != metadata[f"{name}_indices_sha256"]:
            raise ValueError(f"split identity mismatch: {name}")
        ids[name] = json.loads(path.read_text())
    if set(ids["train"]) & set(ids["validation"]):
        raise ValueError("overlapping split indices")
    for package in ("torch", "transformers", "scikit-learn", "numpy"):
        if version(package) != metadata["packages"][package]:
            raise ValueError(f"original package version required: {package}")
    nli_path = root / "models/nli-deberta-v3-small"
    if _model_files(nli_path) != metadata["models"]["nli"]:
        raise ValueError("NLI model identity mismatch")
    identity = {
        "scope": "descriptive internal-validation input substitution",
        "official_dev_records_read": 0,
        "runner_sha256": sha256(Path(__file__)),
        "nli_code_sha256": sha256(root / "src/apv_rag/nli_comparison.py"),
        "original_inputs": {name: hashes[name] for name in required},
        "protocol_sha256": sha256(root / "artifacts/frozen_heldout_protocol/protocol.json"),
        "source_sha256": sha256(source),
        "selection": "first five distinct (answer text, source URL) excerpts in source order",
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "input_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("input identity changed; refusing cache reuse")
    write_json_atomic(manifest, identity)
    records = json.loads(source.read_text(encoding="utf-8"))
    validation = [records[i] for i in ids["validation"]]
    truth = [r["label"] for r in validation]
    y_train = [records[i]["label"] for i in ids["train"]]
    priors = [(np.asarray(y_train) == label).mean() for label in LABELS]
    if not np.allclose(priors, protocol["training_priors"]):
        raise ValueError("training prior mismatch")
    feature_path = output / "own_validation_features.npy"
    if feature_path.exists():
        cache_receipt(feature_path)
        own_features = np.load(feature_path, allow_pickle=False)
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads)
        torch.manual_seed(369)
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            nli_path, local_files_only=True
        ).eval()
        if {int(k): str(v).lower() for k, v in model.config.id2label.items()} != {
            0: "contradiction",
            1: "entailment",
            2: "neutral",
        }:
            raise ValueError("unexpected NLI label mapping")
        cache_path = output / "pair_cache.sqlite"
        connection = sqlite3.connect(cache_path)
        connection.execute("CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, scores TEXT)")
        if connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 0:
            old_cache = original / "pair_cache.sqlite"
            if old_cache.exists():
                with sqlite3.connect(old_cache.resolve().as_uri() + "?mode=ro", uri=True) as old:
                    old.backup(connection)
        features, audit = [], []
        for i, record in enumerate(validation):
            premises = own_premises(record)
            scores = _pairs(
                connection,
                model,
                tokenizer,
                torch,
                premises,
                record["claim"],
                _fingerprint(metadata["models"]["nli"]),
                batch_size,
            )
            features.append(nli_features(scores))
            audit.append(
                {
                    "upstream_index": ids["validation"][i],
                    "premises": premises,
                    "nli_probabilities": scores,
                }
            )
            if i % 50 == 0 or i + 1 == len(validation):
                print(f"Own-excerpt NLI: {i + 1}/{len(validation)}", flush=True)
        connection.close()
        own_features = np.asarray(features)
        write_json_atomic(output / "own_excerpt_audit.json", audit)
        np.save(feature_path, own_features)
        cache_receipt(feature_path, save=True)
    check_features(own_features, len(validation))
    original_rows = json.loads((original / "predictions.json").read_text())
    lookup = {(r["method"], r["seed"], r["upstream_index"]): r for r in original_rows}
    expected = {
        (m, seed, i)
        for m in protocol["methods"]
        for seed in protocol["seeds"]
        for i in ids["validation"]
    }
    if len(original_rows) != len(expected) or set(lookup) != expected:
        raise ValueError("original prediction pairing mismatch")
    results, saved = [], []
    for method in protocol["methods"]:
        x_train = np.load(original / f"train_{method}_features.npy", allow_pickle=False)
        x_val = np.load(original / f"validation_{method}_features.npy", allow_pickle=False)
        check_features(x_train, len(ids["train"]))
        check_features(x_val, len(validation))
        for seed in protocol["seeds"]:
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=seed),
            )
            model = CalibratedClassifierCV(
                estimator, method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=seed)
            )
            model.fit(x_train, y_train)
            order = [list(model.classes_).index(label) for label in LABELS]
            retrieved_p = model.predict_proba(x_val)[:, order]
            prior_rows = [lookup[(method, seed, i)] for i in ids["validation"]]
            if [r["true_label"] for r in prior_rows] != truth or not np.allclose(
                retrieved_p, [r["probabilities"] for r in prior_rows], atol=1e-7, rtol=1e-6
            ):
                raise ValueError("refitted aggregator does not reproduce original probabilities")
            own_p = model.predict_proba(own_features)[:, order]
            for rule, alpha in (
                ("argmax", 0),
                ("selected", protocol["methods"][method]["decision_exponent"]),
            ):
                a = np.asarray(LABELS)[decision_indices(retrieved_p, priors, alpha)]
                b = np.asarray(LABELS)[decision_indices(own_p, priors, alpha)]
                metrics = {}
                for name, prediction, p in (("retrieved", a, retrieved_p), ("own", b, own_p)):
                    metrics[name] = classification_metrics(
                        truth, prediction, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
                    )
                    if rule == "selected":
                        raw = np.asarray(LABELS)[p.argmax(axis=1)]
                        raw_metrics = classification_metrics(
                            truth, raw, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
                        )
                        metrics[name]["expected_calibration_error"] = raw_metrics[
                            "expected_calibration_error"
                        ]
                        metrics[name].pop("risk_coverage")
                corrected = int(((a != truth) & (b == truth)).sum())
                broken = int(((a == truth) & (b != truth)).sum())
                results.append(
                    {
                        "method": method,
                        "seed": seed,
                        "rule": rule,
                        "metrics": metrics,
                        "corrected": corrected,
                        "new_errors": broken,
                        "macro_f1_change": metrics["own"]["macro_f1"]
                        - metrics["retrieved"]["macro_f1"],
                    }
                )
                for i, upstream in enumerate(ids["validation"]):
                    saved.append(
                        {
                            "method": method,
                            "seed": seed,
                            "rule": rule,
                            "upstream_index": upstream,
                            "true_label": truth[i],
                            "retrieved_label": str(a[i]),
                            "own_label": str(b[i]),
                            "retrieved_probabilities": retrieved_p[i].tolist(),
                            "own_probabilities": own_p[i].tolist(),
                        }
                    )
    report = {
        "status": "descriptive diagnostic; not new held-out confirmation",
        "official_dev_records_read": 0,
        "validation_claims": len(validation),
        "aggregator": "original retrieved-feature training only; parameters and rules fixed",
        "limitations": [
            "own excerpts are benchmark annotations, not guaranteed sufficient evidence",
            "source-order first five can omit useful excerpts",
            "input substitution changes feature distribution",
            "difference does not isolate retrieval as the sole cause",
        ],
        "results": results,
    }
    write_json_atomic(output / "own_excerpt_summary.json", report)
    write_json_atomic(output / "predictions.json", saved)
    write_json_atomic(
        output / "output_manifest.json",
        {
            p.name: sha256(p)
            for p in output.iterdir()
            if p.suffix in (".json", ".npy") and p.name != "output_manifest.json"
        },
    )
    print(f"Completed: {output / 'own_excerpt_summary.json'}", flush=True)
