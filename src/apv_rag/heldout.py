"""Frozen oracle-excerpt evaluation. Inference imports remain optional."""

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

from apv_rag.decision_rules import decision_indices
from apv_rag.evidence_baseline import LABELS
from apv_rag.metrics import classification_metrics
from apv_rag.nli_comparison import (
    _fingerprint,
    _model_files,
    _pairs,
    nli_features,
    pool_corpus,
    semantic_ranking,
)
from apv_rag.paired_statistics import holm_adjust, macro_score
from apv_rag.retrieval import BM25Index
from apv_rag.splits import build_connected_groups, sha256, write_json_atomic


def overlap_audit(train, fitted_ids, dev):
    groups, empty = build_connected_groups(train + dev)
    fitted = set(fitted_ids)
    offset = len(train)
    excluded, independent, membership = [], [], {}
    for number, group in enumerate(groups):
        linked = bool(fitted.intersection(group))
        for index in group:
            if index < offset:
                continue
            i = index - offset
            membership[str(i)] = str(number)
            if linked:
                excluded.append({"dev_index": i, "reason": "connected to fitted training record"})
            else:
                independent.append(i)
    for index in empty:
        if index >= offset:
            i = index - offset
            membership[str(i)] = f"empty-{i}"
            excluded.append({"dev_index": i, "reason": "empty normalized claim"})
    return {
        "independent_ids": sorted(independent),
        "groups": membership,
        "excluded": sorted(excluded, key=lambda row: row["dev_index"]),
    }


def check_features(x, count):
    if x.shape != (count, 7) or not np.isfinite(x).all():
        raise ValueError("invalid feature matrix")


def cache_receipt(path, save=False):
    receipt = path.with_suffix(path.suffix + ".sha256.json")
    actual = {path.name: sha256(path)}
    if save:
        write_json_atomic(receipt, actual)
    elif not receipt.exists() or json.loads(receipt.read_text()) != actual:
        raise ValueError(f"cache checksum mismatch: {path.name}")


def population_report(ids, truth, probabilities, protocol, groups=None):
    if not ids:
        return {"status": "unavailable: no independent claims", "claims": 0}
    labels = np.asarray(LABELS)
    y = np.asarray([LABELS.index(truth[i]) for i in ids])
    names = sorted({groups[str(i)] for i in ids})
    positions = [np.asarray([j for j, i in enumerate(ids) if groups[str(i)] == g]) for g in names]
    membership = np.asarray([names.index(groups[str(i)]) for i in ids])
    results = []
    for method, rules in protocol["methods"].items():
        baseline, selected, metrics = [], [], []
        for seed in protocol["seeds"]:
            p = probabilities[method][seed][ids]
            a = decision_indices(p, protocol["training_priors"], 0)
            b = decision_indices(p, protocol["training_priors"], rules["decision_exponent"])
            baseline.append(a)
            selected.append(b)
            row = {"seed": seed}
            for key, pred in (("argmax", a), ("selected", b)):
                row[key] = classification_metrics(
                    labels[y], labels[pred], p, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
                )
            # Confidence calibration concerns raw probability argmax, not adjusted decisions.
            row["selected"]["expected_calibration_error"] = row["argmax"][
                "expected_calibration_error"
            ]
            row["selected"].pop("risk_coverage")
            metrics.append(row)
        a, b = np.asarray(baseline), np.asarray(selected)

        def difference(ix, left, right):
            return float(
                np.mean(
                    [
                        macro_score(y[ix], after) - macro_score(y[ix], before)
                        for before, after in zip(left, right, strict=True)
                    ]
                )
            )

        ix = np.arange(len(ids))
        observed = difference(ix, a, b)
        rng = np.random.default_rng(protocol["statistics"]["seed"])
        effects = []
        for _ in range(protocol["statistics"]["bootstrap_samples"]):
            sampled = rng.integers(len(names), size=len(names))
            draw = np.concatenate([positions[i] for i in sampled])
            effects.append(difference(draw, a[:, draw], b[:, draw]))
        extreme = 0
        repetitions = protocol["statistics"]["randomization_samples"]
        for _ in range(repetitions):
            swap = rng.integers(2, size=len(names)).astype(bool)[membership][None, :]
            gain = difference(ix, np.where(swap, b, a), np.where(swap, a, b))
            extreme += abs(gain) >= abs(observed) - 1e-12
        results.append(
            {
                "method": method,
                "seed_metrics": metrics,
                "mean_seed_macro_f1_gain": observed,
                "group_bootstrap_95_interval": np.percentile(effects, [2.5, 97.5]).tolist(),
                "randomization_two_sided_p": (extreme + 1) / (repetitions + 1),
            }
        )
    for row, p in zip(
        results, holm_adjust([r["randomization_two_sided_p"] for r in results]), strict=True
    ):
        row["holm_adjusted_p"] = p
    return {
        "status": "evaluated under frozen excerpt protocol",
        "claims": len(ids),
        "groups": len(names),
        "results": results,
    }


def preflight(root):
    frozen = root / "artifacts/frozen_heldout_protocol"
    protocol_path = frozen / "protocol.json"
    checksums = json.loads((frozen / "protocol_checksum.json").read_text())
    if sha256(protocol_path) != checksums["protocol.json"]:
        raise ValueError("frozen protocol checksum mismatch")
    protocol = json.loads(protocol_path.read_text())
    paths = dict(protocol["code_sha256"])
    paths["artifacts/cached_decision_comparison/decision_summary.json"] = protocol[
        "decision_summary_sha256"
    ]
    paths["data/external/averitec/official_7c62d1e/train.json"] = protocol["train_source_sha256"]
    paths["data/processed/averitec/phase2b_split_v0_1/train_indices.json"] = protocol[
        "train_indices_sha256"
    ]
    for name, expected in paths.items():
        if sha256(root / name) != expected:
            raise ValueError(f"frozen input checksum mismatch: {name}")
    for package, expected in protocol["inference_packages"].items():
        if version(package) != expected:
            raise ValueError(f"package version differs from freeze: {package} needs {expected}")
    for name, directory in (("nli", "nli-deberta-v3-small"), ("embedding", "all-MiniLM-L6-v2")):
        if _model_files(root / "models" / directory) != protocol["model_files"][name]:
            raise ValueError(f"model files differ from freeze: {name}")
    for method, rules in protocol["methods"].items():
        path = root / "artifacts/retrieval_nli_comparison" / f"train_{method}_features.npy"
        if sha256(path) != rules["training_feature_sha256"]:
            raise ValueError(f"training features differ from freeze: {method}")
    return protocol


def run_heldout(root, batch_size=8, threads=4):
    root = Path(root).resolve()
    if batch_size < 1 or threads < 1:
        raise ValueError("batch size and threads must be positive")
    protocol = preflight(root)  # No dev read before every frozen input passes.
    output = root / "artifacts/heldout_excerpt_evaluation"
    if (output / "output_manifest.json").exists():
        raise FileExistsError("completed evaluation exists; refusing to overwrite")
    identity = {
        "protocol_sha256": sha256(root / "artifacts/frozen_heldout_protocol/protocol.json"),
        "evaluator_sha256": sha256(Path(__file__)),
        "grouping_sha256": sha256(root / "src/apv_rag/splits.py"),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "input_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("evaluation inputs changed; refusing cache reuse")
    write_json_atomic(manifest, identity)
    data = root / "data/external/averitec/official_7c62d1e"
    dev_path = data / "dev.json"
    if sha256(dev_path) != protocol["evaluation_source"]["expected_sha256"]:
        raise ValueError("official dev checksum mismatch")
    train = json.loads((data / "train.json").read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if len(dev) != protocol["evaluation_source"]["count"]:
        raise ValueError("official dev record count mismatch")
    ids = json.loads(
        (root / "data/processed/averitec/phase2b_split_v0_1/train_indices.json").read_text()
    )
    audit = overlap_audit(train, ids, dev)
    write_json_atomic(output / "overlap_audit.json", audit)
    print(
        f"Overlap audit: {len(audit['independent_ids'])}/{len(dev)} independent claims", flush=True
    )
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    nli_path = root / "models/nli-deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
    nli = AutoModelForSequenceClassification.from_pretrained(nli_path, local_files_only=True).eval()
    if {int(k): str(v).lower() for k, v in nli.config.id2label.items()} != {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }:
        raise ValueError("unexpected NLI label mapping")
    embedding = SentenceTransformer(
        str(root / "models/all-MiniLM-L6-v2"), device="cpu", local_files_only=True
    )
    documents = pool_corpus(dev)
    if not documents:
        raise ValueError("empty held-out excerpt corpus")
    write_json_atomic(output / "corpus.json", documents)
    texts = [d["text"] for d in documents]
    bm25 = BM25Index(texts)
    vectors_path = output / "embeddings.npz"
    if vectors_path.exists():
        cache_receipt(vectors_path)
        with np.load(vectors_path, allow_pickle=False) as cache:
            vectors, queries = cache["documents"], cache["queries"]
    else:
        vectors = embedding.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
        )
        queries = embedding.encode(
            [r["claim"] for r in dev],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.savez_compressed(vectors_path, documents=vectors, queries=queries)
        cache_receipt(vectors_path, save=True)
    probabilities, saved = {}, []
    truth = [r["label"] for r in dev]
    y_train = [train[i]["label"] for i in ids]
    priors = [(np.asarray(y_train) == label).mean() for label in LABELS]
    if not np.allclose(priors, protocol["training_priors"]):
        raise ValueError("training label priors differ from freeze")
    with sqlite3.connect(output / "pair_cache.sqlite") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, scores TEXT)")
        for method in protocol["methods"]:
            feature_path = output / f"dev_{method}_features.npy"
            if feature_path.exists():
                cache_receipt(feature_path)
                x_dev = np.load(feature_path, allow_pickle=False)
            else:
                rows, retrieval = [], []
                for i, record in enumerate(dev):
                    positions = (
                        [p for p, _ in bm25.search(record["claim"], 5)]
                        if method == "bm25_nli"
                        else semantic_ranking(vectors @ queries[i], 5)
                    )
                    scores = _pairs(
                        connection,
                        nli,
                        tokenizer,
                        torch,
                        [texts[p] for p in positions],
                        record["claim"],
                        _fingerprint(protocol["model_files"]["nli"]),
                        batch_size,
                    )
                    rows.append(nli_features(scores))
                    retrieval.append(
                        {"dev_index": i, "document_ids": positions, "nli_probabilities": scores}
                    )
                    if i % 50 == 0 or i == len(dev) - 1:
                        print(f"{method}: {i + 1}/{len(dev)}", flush=True)
                x_dev = np.asarray(rows)
                write_json_atomic(output / f"{method}_retrieval_audit.json", retrieval)
                np.save(feature_path, x_dev)
                cache_receipt(feature_path, save=True)
            check_features(x_dev, len(dev))
            x_train = np.load(
                root / "artifacts/retrieval_nli_comparison" / f"train_{method}_features.npy",
                allow_pickle=False,
            )
            check_features(x_train, len(ids))
            probabilities[method] = {}
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
                model.fit(x_train, y_train)
                p = model.predict_proba(x_dev)[
                    :, [list(model.classes_).index(label) for label in LABELS]
                ]
                probabilities[method][seed] = p
                selected = decision_indices(
                    p, priors, protocol["methods"][method]["decision_exponent"]
                )
                for i in range(len(dev)):
                    saved.append(
                        {
                            "method": method,
                            "seed": seed,
                            "dev_index": i,
                            "true_label": truth[i],
                            "probabilities": p[i].tolist(),
                            "argmax_label": LABELS[int(p[i].argmax())],
                            "selected_label": LABELS[int(selected[i])],
                        }
                    )
    write_json_atomic(output / "predictions.json", saved)
    report = {
        "scope": protocol["scope"],
        "official_dev_records_used": len(dev),
        "primary_independent": population_report(
            audit["independent_ids"], truth, probabilities, protocol, audit["groups"]
        ),
        "secondary_all_dev": population_report(
            list(range(len(dev))), truth, probabilities, protocol, audit["groups"]
        ),
    }
    write_json_atomic(output / "heldout_summary.json", report)
    write_json_atomic(
        output / "output_manifest.json",
        {
            p.name: sha256(p)
            for p in output.iterdir()
            if p.suffix in (".json", ".npy", ".npz") and p.name != "output_manifest.json"
        },
    )
    print(f"Completed: {output / 'heldout_summary.json'}", flush=True)
