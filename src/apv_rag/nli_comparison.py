"""Local-model retrieval/NLI excerpt comparison, with resumable pair inference."""

import hashlib
import json
import sqlite3
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS
from apv_rag.evidence_baseline import LABELS, load_split_records
from apv_rag.metrics import classification_metrics
from apv_rag.retrieval import BM25Index
from apv_rag.splits import sha256, write_json_atomic

SEEDS = (369, 1369, 2369, 3369, 4369)


def pool_corpus(records):
    documents, seen = [], set()
    for record in records:
        for question in record.get("questions") or []:
            for answer in question.get("answers") or []:
                text = str(answer.get("answer") or "").strip()
                url = str(answer.get("source_url") or "")
                key = (text, url)
                if text and key not in seen:
                    seen.add(key)
                    documents.append({"text": text, "source_url": url})
    return documents


def semantic_ranking(scores, top_k):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or top_k < 0:
        raise ValueError("invalid retrieval scores or top_k")
    return [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:top_k]


def nli_features(scores):
    matrix = np.asarray(scores, dtype=float)
    if matrix.size == 0:
        return np.zeros(7)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 3
        or not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or (matrix > 1).any()
        or not np.allclose(matrix.sum(axis=1), 1)
    ):
        raise ValueError("NLI scores must be normalized three-label probabilities")
    return np.concatenate([matrix.max(axis=0), matrix.mean(axis=0), [len(matrix)]])


def _model_files(directory):
    paths = [p for p in Path(directory).rglob("*") if p.is_file() and ".cache" not in p.parts]
    if not paths:
        raise FileNotFoundError(f"no local model files: {directory}")
    return {str(p.relative_to(directory)).replace("\\", "/"): sha256(p) for p in sorted(paths)}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _pairs(connection, model, tokenizer, torch, premises, claim, model_hash, batch_size):
    keys = [_fingerprint([model_hash, p, claim, 256]) for p in premises]
    values = {}
    missing = []
    for position, key in enumerate(keys):
        cached = connection.execute("SELECT scores FROM pairs WHERE key=?", (key,)).fetchone()
        if cached:
            values[position] = json.loads(cached[0])
        else:
            missing.append(position)
    for start in range(0, len(missing), batch_size):
        positions = missing[start : start + batch_size]
        inputs = tokenizer(
            [premises[i] for i in positions],
            [claim] * len(positions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(dim=-1).cpu().numpy()
        for position, probabilities_row in zip(positions, probabilities, strict=True):
            values[position] = probabilities_row.astype(float).tolist()
            connection.execute(
                "INSERT OR REPLACE INTO pairs VALUES (?,?)",
                (keys[position], json.dumps(values[position])),
            )
        connection.commit()
    return [values[i] for i in range(len(premises))]


def run_comparison(root, *, batch_size=8, threads=4):
    # Optional inference libraries are loaded only when running this experiment.
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    root = Path(root).resolve()
    if batch_size < 1 or threads < 1:
        raise ValueError("batch size and threads must be positive")
    started = time.perf_counter()
    torch.set_num_threads(threads)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    source = root / "data/external/averitec" / DATASET_DIR_NAME / "train.json"
    if sha256(source) != SPLITS["train"].sha256:
        raise ValueError("upstream train.json checksum mismatch")
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    train_ids = json.loads((split / "train_indices.json").read_text(encoding="utf-8"))
    val_ids = json.loads((split / "validation_indices.json").read_text(encoding="utf-8"))
    if set(train_ids) & set(val_ids):
        raise ValueError("training and validation records overlap")
    upstream = json.loads(source.read_text(encoding="utf-8"))
    records = {
        "train": load_split_records(upstream, train_ids),
        "validation": load_split_records(upstream, val_ids),
    }
    nli_path = root / "models/nli-deberta-v3-small"
    embedding_path = root / "models/all-MiniLM-L6-v2"
    model_files = {"nli": _model_files(nli_path), "embedding": _model_files(embedding_path)}
    model_hash = _fingerprint(model_files["nli"])
    output = root / "artifacts/retrieval_nli_comparison"
    output.mkdir(parents=True, exist_ok=True)
    input_identity = {
        "scope": "split-local oracle evidence-excerpt corpora; not open-web retrieval",
        "source_sha256": sha256(source),
        "models": model_files,
        "train_indices_sha256": sha256(split / "train_indices.json"),
        "validation_indices_sha256": sha256(split / "validation_indices.json"),
        "top_k": 5,
        "max_length": 256,
        "seeds": list(SEEDS),
        "packages": {
            name: version(name)
            for name in ("torch", "transformers", "sentence-transformers", "scikit-learn", "numpy")
        },
    }
    manifest_path = output / "input_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != input_identity:
            raise ValueError("inputs changed; do not reuse this experiment cache")
    else:
        write_json_atomic(manifest_path, input_identity)
    tokenizer = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
    nli = AutoModelForSequenceClassification.from_pretrained(nli_path, local_files_only=True)
    nli.eval()
    mapping = {int(k): str(v).lower() for k, v in nli.config.id2label.items()}
    if mapping != {0: "contradiction", 1: "entailment", 2: "neutral"}:
        raise ValueError(f"unexpected NLI label mapping: {mapping}")
    embedding = SentenceTransformer(str(embedding_path), device="cpu", local_files_only=True)
    connection = sqlite3.connect(output / "pair_cache.sqlite")
    connection.execute("CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, scores TEXT)")
    features = {method: {} for method in ("bm25_nli", "dense_nli")}
    for split_name, split_records in records.items():
        documents = pool_corpus(split_records)
        if not documents:
            raise ValueError("empty evidence corpus")
        corpus_path = output / f"{split_name}_corpus.json"
        write_json_atomic(corpus_path, documents)
        texts = [document["text"] for document in documents]
        bm25 = BM25Index(texts)
        vector_path = output / f"{split_name}_embeddings.npz"
        if vector_path.exists():
            with np.load(vector_path) as cache:
                doc_vectors, query_vectors = cache["documents"], cache["queries"]
        else:
            print(f"Encoding {split_name} documents and claims...", flush=True)
            doc_vectors = embedding.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
            )
            query_vectors = embedding.encode(
                [r["claim"] for r in split_records],
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            np.savez_compressed(vector_path, documents=doc_vectors, queries=query_vectors)
        for method in features:
            feature_path = output / f"{split_name}_{method}_features.npy"
            if feature_path.exists():
                features[method][split_name] = np.load(feature_path)
                continue
            rows, audit = [], []
            for i, record in enumerate(split_records):
                positions = (
                    [p for p, _ in bm25.search(record["claim"], 5)]
                    if method == "bm25_nli"
                    else semantic_ranking(doc_vectors @ query_vectors[i], 5)
                )
                probabilities = _pairs(
                    connection,
                    nli,
                    tokenizer,
                    torch,
                    [texts[p] for p in positions],
                    record["claim"],
                    model_hash,
                    batch_size,
                )
                rows.append(nli_features(probabilities))
                audit.append(
                    {
                        "record_position": i,
                        "document_ids": positions,
                        "nli_probabilities": probabilities,
                    }
                )
                if i % 50 == 0 or i + 1 == len(split_records):
                    print(f"{split_name} {method}: {i + 1}/{len(split_records)}", flush=True)
            features[method][split_name] = np.asarray(rows)
            write_json_atomic(output / f"{split_name}_{method}_retrieval_audit.json", audit)
            np.save(feature_path, features[method][split_name])
    connection.close()
    truth_train = [row["label"] for row in records["train"]]
    truth_val = [row["label"] for row in records["validation"]]
    results, predictions = [], []
    for method in features:
        for seed in SEEDS:
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=seed),
            )
            model = CalibratedClassifierCV(
                estimator,
                method="sigmoid",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
            )
            model.fit(features[method]["train"], truth_train)
            probabilities = model.predict_proba(features[method]["validation"])
            probabilities = probabilities[
                :, [list(model.classes_).index(label) for label in LABELS]
            ]
            predicted = [LABELS[p] for p in probabilities.argmax(axis=1)]
            metrics = classification_metrics(
                truth_val, predicted, probabilities, LABELS, ece_bins=15, coverages=[0.5, 0.8, 1]
            )
            results.append({"method": method, "seed": seed, "metrics": metrics})
            for i, upstream_id in enumerate(val_ids):
                predictions.append(
                    {
                        "method": method,
                        "seed": seed,
                        "upstream_index": upstream_id,
                        "true_label": truth_val[i],
                        "predicted_label": predicted[i],
                        "probabilities": probabilities[i].tolist(),
                    }
                )
    write_json_atomic(output / "predictions.json", predictions)
    summary = {
        "official_dev_records_used": 0,
        "scope": input_identity["scope"],
        "counts": {name: len(value) for name, value in records.items()},
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output / "comparison_summary.json", summary)
    outputs = [p for p in output.iterdir() if p.suffix in (".json", ".npy", ".npz")]
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in outputs if p.name != "output_manifest.json"},
    )
    print("Comparison complete; official development records used: 0", flush=True)
    print(output / "comparison_summary.json", flush=True)
    return output


def validate_comparison(output):
    output, errors = Path(output), []
    try:
        hashes = json.loads((output / "output_manifest.json").read_text(encoding="utf-8"))
        for name, expected in hashes.items():
            if Path(name).name != name or not (output / name).is_file():
                errors.append(f"missing or invalid output: {name}")
            elif sha256(output / name) != expected:
                errors.append(f"output checksum mismatch: {name}")
        if errors:
            return errors
        summary = json.loads((output / "comparison_summary.json").read_text(encoding="utf-8"))
        if summary["official_dev_records_used"] != 0:
            errors.append("official development records must remain unused")
        rows = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
        results = summary["results"]
        expected_keys = {(method, seed) for method in ("bm25_nli", "dense_nli") for seed in SEEDS}
        if {(r["method"], r["seed"]) for r in results} != expected_keys or len(results) != 10:
            errors.append("comparison must contain both methods and all five seeds")
        for result in results:
            selected = [
                r for r in rows if r["method"] == result["method"] and r["seed"] == result["seed"]
            ]
            if len(selected) != summary["counts"]["validation"]:
                errors.append("prediction count mismatch")
                continue
            if len({r["upstream_index"] for r in selected}) != len(selected):
                errors.append("duplicate prediction IDs")
            metrics = classification_metrics(
                [r["true_label"] for r in selected],
                [r["predicted_label"] for r in selected],
                np.asarray([r["probabilities"] for r in selected]),
                LABELS,
                ece_bins=15,
                coverages=[0.5, 0.8, 1],
            )
            if metrics != result["metrics"]:
                errors.append("recorded metrics differ from saved predictions")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        errors.append(str(exc))
    return errors
