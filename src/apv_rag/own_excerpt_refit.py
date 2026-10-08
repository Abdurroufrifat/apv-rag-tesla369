"""Train on own-excerpt NLI features; evaluate internal validation only."""

import json
import sqlite3
from pathlib import Path

import numpy as np

from apv_rag import own_excerpt as own
from apv_rag.decision_rules import decision_indices
from apv_rag.evidence_baseline import LABELS
from apv_rag.heldout import cache_receipt, check_features
from apv_rag.metrics import classification_metrics
from apv_rag.nli_comparison import _fingerprint, _pairs, nli_features
from apv_rag.splits import build_connected_groups, sha256, write_json_atomic


def run_refit(root, batch_size=8, threads=4):
    root = Path(root).resolve()
    if batch_size < 1 or threads < 1:
        raise ValueError("batch size and threads must be positive")
    output = root / "artifacts/own_excerpt_refit"
    if (output / "output_manifest.json").exists():
        raise FileExistsError("completed own-excerpt refit exists; refusing overwrite")
    prior = root / "artifacts/own_excerpt_comparison"
    hashes = json.loads((prior / "output_manifest.json").read_text())
    for name in (
        "own_validation_features.npy",
        "own_excerpt_summary.json",
        "predictions.json",
        "input_manifest.json",
    ):
        if sha256(prior / name) != hashes[name]:
            raise ValueError(f"own-excerpt input checksum mismatch: {name}")
    original = root / "artifacts/retrieval_nli_comparison"
    metadata_path = original / "input_manifest.json"
    original_hashes = json.loads((original / "output_manifest.json").read_text())
    if sha256(metadata_path) != original_hashes[metadata_path.name]:
        raise ValueError("original inference metadata checksum mismatch")
    metadata = json.loads(metadata_path.read_text())
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    if sha256(source) != own.SPLITS["train"].sha256 or sha256(source) != metadata["source_sha256"]:
        raise ValueError("training source checksum mismatch")
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    ids = {}
    for name in ("train", "validation"):
        path = split / f"{name}_indices.json"
        if sha256(path) != metadata[f"{name}_indices_sha256"]:
            raise ValueError(f"split checksum mismatch: {name}")
        ids[name] = json.loads(path.read_text())
    if set(ids["train"]) & set(ids["validation"]):
        raise ValueError("overlapping train/validation records")
    for package in ("torch", "transformers", "scikit-learn", "numpy"):
        if own.version(package) != metadata["packages"][package]:
            raise ValueError(f"original package version required: {package}")
    nli_path = root / "models/nli-deberta-v3-small"
    if own._model_files(nli_path) != metadata["models"]["nli"]:
        raise ValueError("NLI model files differ from original inference")
    protocol_path = root / "artifacts/frozen_heldout_protocol/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if any(rules["decision_exponent"] != 1 for rules in protocol["methods"].values()):
        raise ValueError("this diagnostic expects the previously fixed exponent of one")
    identity = {
        "scope": "development-only own-excerpt training comparison",
        "official_dev_records_read": 0,
        "runner_sha256": sha256(Path(__file__)),
        "own_excerpt_code_sha256": sha256(Path(own.__file__)),
        "nli_code_sha256": sha256(root / "src/apv_rag/nli_comparison.py"),
        "protocol_sha256": sha256(protocol_path),
        "prior_inputs": {
            name: hashes[name]
            for name in (
                "own_validation_features.npy",
                "own_excerpt_summary.json",
                "predictions.json",
                "input_manifest.json",
            )
        },
        "metadata_sha256": sha256(metadata_path),
        "train_source_sha256": sha256(source),
        "selection": "first five distinct answer-text/source-URL excerpts in source order",
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "input_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("diagnostic inputs changed; refusing cache reuse")
    write_json_atomic(manifest, identity)
    records = json.loads(source.read_text(encoding="utf-8"))
    groups, _ = build_connected_groups(records)
    if any(
        set(group) & set(ids["train"]) and set(group) & set(ids["validation"]) for group in groups
    ):
        raise ValueError("connected provenance group crosses train/validation")
    train = [records[i] for i in ids["train"]]
    y_train = [r["label"] for r in train]
    truth = [records[i]["label"] for i in ids["validation"]]
    priors = [(np.asarray(y_train) == label).mean() for label in LABELS]
    if not np.allclose(priors, protocol["training_priors"]):
        raise ValueError("training prior mismatch")
    train_path = output / "own_train_features.npy"
    if train_path.exists():
        cache_receipt(train_path)
        x_train = np.load(train_path, allow_pickle=False)
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
        connection = sqlite3.connect(output / "pair_cache.sqlite")
        connection.execute("CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, scores TEXT)")
        if connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 0:
            prior_cache = prior / "pair_cache.sqlite"
            if prior_cache.exists():
                with sqlite3.connect(
                    prior_cache.resolve().as_uri() + "?mode=ro", uri=True
                ) as cached:
                    cached.backup(connection)
        features, audit = [], []
        for i, record in enumerate(train):
            premises = own.own_premises(record)
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
                    "upstream_index": ids["train"][i],
                    "premises": premises,
                    "nli_probabilities": scores,
                }
            )
            if i % 50 == 0 or i + 1 == len(train):
                print(f"Own training excerpts: {i + 1}/{len(train)}", flush=True)
        connection.close()
        x_train = np.asarray(features)
        write_json_atomic(output / "training_excerpt_audit.json", audit)
        np.save(train_path, x_train)
        cache_receipt(train_path, save=True)
    x_val = np.load(prior / "own_validation_features.npy", allow_pickle=False)
    check_features(x_train, len(train))
    check_features(x_val, len(truth))
    prior_rows = json.loads((prior / "predictions.json").read_text())
    lookup = {(r["method"], r["seed"], r["rule"], r["upstream_index"]): r for r in prior_rows}
    expected = {
        (m, s, rule, i)
        for m in protocol["methods"]
        for s in protocol["seeds"]
        for rule in ("argmax", "selected")
        for i in ids["validation"]
    }
    if len(prior_rows) != len(expected) or set(lookup) != expected:
        raise ValueError("prior comparison prediction pairing mismatch")
    results, saved = [], []
    for seed in protocol["seeds"]:
        model = own.CalibratedClassifierCV(
            own.make_pipeline(
                own.StandardScaler(),
                own.LogisticRegression(
                    C=1, class_weight="balanced", max_iter=2000, random_state=seed
                ),
            ),
            method="sigmoid",
            cv=own.StratifiedKFold(5, shuffle=True, random_state=seed),
        )
        model.fit(x_train, y_train)
        p = model.predict_proba(x_val)[:, [list(model.classes_).index(label) for label in LABELS]]
        raw = np.asarray(LABELS)[p.argmax(axis=1)]
        raw_metrics = classification_metrics(
            truth, raw, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
        )
        for rule, alpha in (("argmax", 0), ("selected", 1)):
            predictions = np.asarray(LABELS)[decision_indices(p, priors, alpha)]
            metrics = classification_metrics(
                truth, predictions, p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
            )
            if alpha:
                metrics["expected_calibration_error"] = raw_metrics["expected_calibration_error"]
                metrics.pop("risk_coverage")
            comparisons = []
            for method in protocol["methods"]:
                previous = [lookup[(method, seed, rule, i)] for i in ids["validation"]]
                if [r["true_label"] for r in previous] != truth:
                    raise ValueError("prior comparison labels differ")
                old_p = np.asarray([r["own_probabilities"] for r in previous])
                old_prediction = [r["own_label"] for r in previous]
                old_metrics = classification_metrics(
                    truth, old_prediction, old_p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
                )
                comparisons.append(
                    {
                        "retrieved_training_method": method,
                        "fixed_aggregator_own_input_macro_f1": old_metrics["macro_f1"],
                        "own_training_macro_f1_change": metrics["macro_f1"]
                        - old_metrics["macro_f1"],
                    }
                )
            results.append(
                {"seed": seed, "rule": rule, "metrics": metrics, "comparisons": comparisons}
            )
            for i, upstream in enumerate(ids["validation"]):
                saved.append(
                    {
                        "seed": seed,
                        "rule": rule,
                        "upstream_index": upstream,
                        "true_label": truth[i],
                        "predicted_label": str(predictions[i]),
                        "probabilities": p[i].tolist(),
                    }
                )
    report = {
        "status": "development diagnostic; not new held-out confirmation",
        "official_dev_records_read": 0,
        "training_claims": len(train),
        "validation_claims": len(truth),
        "seeds": protocol["seeds"],
        "training_input": "own first-five answer-excerpt NLI features",
        "validation_input": "own first-five answer-excerpt NLI features",
        "rules": "argmax and prior adjustment exponent one; no new search",
        "calibration": "original five stratified inner folds, not grouped",
        "limitations": [
            "oracle benchmark excerpts; not deployable retrieval evaluation",
            "source order and five-excerpt cap can omit useful evidence",
            "official dev is observed and cannot confirm this revised method",
        ],
        "results": results,
    }
    write_json_atomic(output / "own_excerpt_refit_summary.json", report)
    write_json_atomic(output / "predictions.json", saved)
    write_json_atomic(
        output / "output_manifest.json",
        {
            p.name: sha256(p)
            for p in output.iterdir()
            if p.suffix in (".json", ".npy") and p.name != "output_manifest.json"
        },
    )
    print(f"Completed: {output / 'own_excerpt_refit_summary.json'}", flush=True)
