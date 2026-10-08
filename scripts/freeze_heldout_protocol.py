"""Freeze the held-out excerpt protocol without opening official dev.json."""

import argparse
import json
from pathlib import Path

from apv_rag.averitec import SPLITS
from apv_rag.evidence_baseline import LABELS
from apv_rag.nli_comparison import SEEDS
from apv_rag.splits import sha256, write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inputs = args.input_dir or root / "artifacts/retrieval_nli_comparison"
    decision = root / "artifacts/cached_decision_comparison"
    selected = json.loads((decision / "decision_summary.json").read_text())
    identity = json.loads((inputs / "input_manifest.json").read_text())
    original = json.loads((inputs / "output_manifest.json").read_text())
    if sha256(inputs / "input_manifest.json") != original["input_manifest.json"]:
        raise ValueError("original model identity checksum mismatch")
    if selected["official_dev_records_used"] != 0:
        raise ValueError("development data must be sealed at freeze")
    if identity["source_sha256"] != SPLITS["train"].sha256:
        raise ValueError("unexpected training source")
    output = root / "artifacts/frozen_heldout_protocol"
    if output.exists():
        raise FileExistsError("refusing to change an existing frozen protocol")
    methods = {}
    for method in ("bm25_nli", "dense_nli"):
        name = f"train_{method}_features.npy"
        if sha256(inputs / name) != original[name]:
            raise ValueError(f"training feature checksum mismatch: {name}")
        methods[method] = {
            "training_feature_sha256": original[name],
            "decision_exponent": selected["methods"][method]["selected_exponent"],
            "baseline_decision_exponent": 0,
        }
    files = [
        "src/apv_rag/nli_comparison.py",
        "src/apv_rag/decision_rules.py",
        "src/apv_rag/retrieval.py",
        "src/apv_rag/paired_statistics.py",
        "scripts/run_cached_decision_comparison.py",
        "scripts/freeze_heldout_protocol.py",
    ]
    protocol = {
        "protocol_version": "heldout_excerpt_v1",
        "status": "frozen before opening official development data",
        "scope": "oracle evidence-excerpt evaluation, not open-web RAG or official AVeriTeC score",
        "official_dev_records_read_at_freeze": 0,
        "evaluation_source": {
            "name": "AVeriTeC official dev",
            "count": SPLITS["dev"].count,
            "expected_sha256": SPLITS["dev"].sha256,
        },
        "train_source_sha256": SPLITS["train"].sha256,
        "train_indices_sha256": identity["train_indices_sha256"],
        "training_priors": selected["priors"],
        "labels": list(LABELS),
        "seeds": list(SEEDS),
        "model_files": identity["models"],
        "inference_packages": identity["packages"],
        "methods": methods,
        "retrieval": {
            "top_k": 5,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "dense_score": "normalized embedding dot product",
            "candidate_corpus": "deduplicated official-dev answer excerpts only",
            "nli_premise": "retrieved answer",
            "nli_hypothesis": "claim",
            "maximum_pair_tokens": 256,
        },
        "aggregator": {
            "C": 1,
            "class_weight": "balanced",
            "max_iter": 2000,
            "standardization": "fitted inside training folds",
            "calibration": "sigmoid",
            "training_calibration_folds": 5,
        },
        "primary_endpoint": "five-seed mean Macro-F1 difference: selected rule minus argmax",
        "primary_comparisons": [
            "bm25_nli selected versus argmax",
            "dense_nli selected versus argmax",
        ],
        "statistics": {
            "bootstrap_samples": 2000,
            "randomization_samples": 10000,
            "seed": 369,
            "resampling_unit": "connected provenance/claim group",
            "pairing": "same groups across rules and all five seeds",
            "p_value": "two-sided with plus-one correction",
            "multiplicity": "Holm across the two primary comparisons",
        },
        "secondary_endpoints": [
            "per-class F1",
            "balanced accuracy",
            "Brier score of raw probabilities",
            "ECE of raw probability argmax",
            "seed variation",
        ],
        "contamination_policy": {
            "audit_before_scoring": "connected groups using frozen build_connected_groups policy",
            "training_members": "the 2458 indices actually used for aggregator fitting",
            "primary_population": (
                "dev records with no connected group shared with fitted training records"
            ),
            "secondary_population": "all 500 official dev records, clearly labeled",
            "empty_independent_population": "report unavailable; do not claim confirmation",
            "excluded_records": "report upstream IDs and reasons without changing labels",
        },
        "no_tuning": [
            "no new exponent search",
            "no fitting on dev labels",
            "no calibration on dev",
            "no model or feature selection from dev results",
        ],
        "interpretation": (
            "confirmation limited to this excerpt protocol; full APV-RAG remains incomplete"
        ),
        "code_sha256": {name: sha256(root / name) for name in files},
        "decision_summary_sha256": sha256(decision / "decision_summary.json"),
        "model_identity_sha256": sha256(inputs / "input_manifest.json"),
    }
    output.mkdir(parents=True)
    write_json_atomic(output / "protocol.json", protocol)
    write_json_atomic(
        output / "protocol_checksum.json", {"protocol.json": sha256(output / "protocol.json")}
    )
    print("Held-out excerpt protocol frozen.")
    print("Official development records read: 0")
    print(output / "protocol.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
