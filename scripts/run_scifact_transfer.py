"""Fixed AVeriTeC-to-SciFact classifier transfer with BM25 abstract retrieval."""

import argparse
import json
import sqlite3
from importlib.metadata import version
from pathlib import Path

import numpy as np
from run_cached_decision_comparison import fit_model, ordered_probabilities
from sklearn.metrics import accuracy_score, classification_report, f1_score

from apv_rag.averitec import SPLITS
from apv_rag.decision_rules import decision_indices
from apv_rag.evidence_baseline import LABELS
from apv_rag.evidence_representation import expanded_features
from apv_rag.nli_comparison import SEEDS, _fingerprint, _model_files, _pairs
from apv_rag.retrieval import BM25Index
from apv_rag.scifact import connected_groups, normalized_claim, target_label
from apv_rag.splits import sha256, write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("Positive batch size and thread count required")
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/scifact_transfer"
    if (output / "output_manifest.json").exists():
        raise FileExistsError("Completed transfer exists; refusing overwrite")
    source_dir = root / "data/external/scifact/sealed_v1"
    manifest = json.loads((source_dir / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(source_dir / name) != digest:
            raise ValueError(f"SciFact checksum mismatch: {name}")
    original = root / "artifacts/retrieval_nli_comparison"
    old_manifest = json.loads((original / "output_manifest.json").read_text())
    meta_path = original / "input_manifest.json"
    if sha256(meta_path) != old_manifest[meta_path.name]:
        raise ValueError("Original metadata mismatch")
    meta = json.loads(meta_path.read_text())
    for name in ("numpy", "scikit-learn", "torch", "transformers"):
        if version(name) != meta["packages"][name]:
            raise ValueError(f"Original package version required: {name}")
    model_path = root / "models/nli-deberta-v3-small"
    if _model_files(model_path) != meta["models"]["nli"]:
        raise ValueError("Original NLI model files required")
    averitec = root / "data/external/averitec/official_7c62d1e"
    for name in ("train", "dev"):
        if sha256(averitec / f"{name}.json") != SPLITS[name].sha256:
            raise ValueError(f"AVeriTeC source mismatch: {name}")
    train = json.loads((averitec / "train.json").read_text(encoding="utf-8"))
    prior_dev = json.loads((averitec / "dev.json").read_text(encoding="utf-8"))
    seen = {normalized_claim(r["claim"]) for r in train + prior_dev}
    split = root / "data/processed/averitec/phase2b_split_v0_1/train_indices.json"
    if sha256(split) != meta["train_indices_sha256"]:
        raise ValueError("Training partition mismatch")
    ids = json.loads(split.read_text())
    y_train = np.asarray([train[i]["label"] for i in ids])
    priors = [(y_train == label).mean() for label in LABELS]
    feature_dir = root / "artifacts/retrieved_feature_comparison"
    feature_manifest = json.loads((feature_dir / "output_manifest.json").read_text())
    filename = "train_bm25_nli_expanded_features.npy"
    if sha256(feature_dir / filename) != feature_manifest[filename]:
        raise ValueError("Training feature checksum mismatch")
    x_train = np.load(feature_dir / filename, allow_pickle=False)
    if x_train.shape != (len(ids), 18) or not np.isfinite(x_train).all():
        raise ValueError("Invalid training features")
    identity = {
        "protocol_sha256": sha256(root / "docs/SCIFACT_TRANSFER_PROTOCOL.md"),
        "runner_sha256": sha256(Path(__file__)),
        "code_sha256": {
            n: sha256(root / n)
            for n in (
                "src/apv_rag/scifact.py",
                "src/apv_rag/evidence_representation.py",
                "src/apv_rag/nli_comparison.py",
                "src/apv_rag/retrieval.py",
                "src/apv_rag/decision_rules.py",
                "scripts/run_cached_decision_comparison.py",
            )
        },
        "target_inputs": manifest["files"],
        "training_features_sha256": feature_manifest[filename],
        "training_indices_sha256": sha256(split),
        "metadata_sha256": sha256(meta_path),
        "training_priors": priors,
        "seeds": list(SEEDS),
        "exponent": 1,
        "batch_size": args.batch_size,
        "threads": args.threads,
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "input_manifest.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Inputs changed; refusing cache reuse")
    write_json_atomic(receipt, identity)
    # Target labels and cited identifiers are never passed to retrieval or feature extraction.
    claims = [
        json.loads(line) for line in (source_dir / "claims_dev.jsonl").read_text().splitlines()
    ]
    if len({r["id"] for r in claims}) != len(claims):
        raise ValueError("Duplicate target claim identifiers")
    overlap = [r["id"] for r in claims if normalized_claim(r["claim"]) in seen]
    remaining = [r for r in claims if r["id"] not in set(overlap)]
    docs = [json.loads(line) for line in (source_dir / "corpus.jsonl").read_text().splitlines()]
    docs.sort(key=lambda d: d["doc_id"])
    texts = [" ".join(d["abstract"]) for d in docs]
    index = BM25Index(texts, k1=1.2, b=0.75)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).eval()
    if {int(k): str(v).lower() for k, v in model.config.id2label.items()} != {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }:
        raise ValueError("Unexpected NLI mapping")
    features, audit = [], []
    with sqlite3.connect(output / "pair_cache.sqlite") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, scores TEXT)")
        for j, claim in enumerate(remaining):
            positions = [p for p, score in index.search(claim["claim"], 5) if score > 0]
            premises = [texts[p] for p in positions]
            urls = ["https://github.com/allenai/scifact" for p in positions]
            # Dataset-host identifier only; not individual article source URLs.
            scores = _pairs(
                connection,
                model,
                tokenizer,
                torch,
                premises,
                claim["claim"],
                _fingerprint(meta["models"]["nli"]),
                args.batch_size,
            )
            features.append(expanded_features(claim["claim"], premises, urls, scores))
            audit.append(
                {
                    "claim_id": claim["id"],
                    "doc_ids": [docs[p]["doc_id"] for p in positions],
                    "premises": premises,
                    "nli_probabilities": scores,
                }
            )
            if j % 25 == 0:
                print(f"SciFact inference: {j + 1}/{len(remaining)}", flush=True)
    # Read labels only for exclusions and scoring after feature extraction.
    keep = [i for i, r in enumerate(remaining) if target_label(r) is not None]
    conflicting = [r["id"] for r in remaining if target_label(r) is None]
    evaluated = [remaining[i] for i in keep]
    if not evaluated:
        raise ValueError("No target claims remain")
    x = np.asarray(features)[keep]
    truth = [target_label(r) for r in evaluated]
    groups = connected_groups(evaluated)
    results, predictions = [], []
    for representation, width in [("base_7", 7), ("expanded_18", 18)]:
        for seed in SEEDS:
            classifier = fit_model(x_train[:, :width], y_train, seed)
            p = ordered_probabilities(classifier, x[:, :width])
            for rule, alpha in [("argmax", 0), ("prior_adjusted", 1)]:
                pred = np.asarray(LABELS)[decision_indices(p, priors, alpha)]
                target_labels = list(LABELS[:3])
                results.append(
                    {
                        "representation": representation,
                        "seed": seed,
                        "rule": rule,
                        "macro_f1": float(
                            f1_score(
                                truth, pred, labels=target_labels, average="macro", zero_division=0
                            )
                        ),
                        "accuracy": float(accuracy_score(truth, pred)),
                        "out_of_target_rate": float(np.mean(pred == LABELS[3])),
                        "per_class": classification_report(
                            truth, pred, labels=target_labels, output_dict=True, zero_division=0
                        ),
                    }
                )
                for j, r in enumerate(evaluated):
                    predictions.append(
                        {
                            "claim_id": r["id"],
                            "group": groups[j],
                            "representation": representation,
                            "seed": seed,
                            "rule": rule,
                            "true_label": truth[j],
                            "predicted_label": str(pred[j]),
                            "probabilities": p[j].tolist(),
                        }
                    )
    write_json_atomic(output / "retrieval_audit.json", audit)
    write_json_atomic(output / "predictions.json", predictions)
    write_json_atomic(
        output / "transfer_summary.json",
        {
            "scope": "fixed classifier transfer; not official joint SciFact metric",
            "claims": len(evaluated),
            "groups": len(set(groups)),
            "results": results,
            "exact_overlap_excluded": overlap,
            "conflicting_labels_excluded": conflicting,
            "statistics_status": "paired grouped statistics pending separate analysis",
            "source_host_feature": "one dataset host for all retrieved abstracts",
            "averitec_dev_use": "previously observed claims for overlap exclusion only",
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {
            p.name: sha256(p)
            for p in output.iterdir()
            if p.suffix == ".json" and p.name != "output_manifest.json"
        },
    )
    print(output / "transfer_summary.json")


if __name__ == "__main__":
    main()
