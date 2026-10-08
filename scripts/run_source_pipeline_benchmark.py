"""Cached SciFact source-pipeline ablations without invented provenance labels."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from apv_rag.direct_nli import TARGET_LABELS
from apv_rag.retrieval import BM25Index
from apv_rag.source_pipeline import PipelineConfig, verify_claim
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    prior = root / "artifacts/scifact_transfer"
    output = root / "artifacts/source_pipeline_benchmark"
    if output.exists():
        raise FileExistsError("Refusing to overwrite completed benchmark")
    manifest = json.loads((prior / "output_manifest.json").read_text())
    for name in ("retrieval_audit.json", "predictions.json", "input_manifest.json"):
        if sha256(prior / name) != manifest[name]:
            raise ValueError(f"Prior input mismatch: {name}")
    identity = json.loads((prior / "input_manifest.json").read_text())
    source = root / "data/external/scifact/sealed_v1"
    for name in ("corpus.jsonl", "claims_dev.jsonl"):
        if sha256(source / name) != identity["target_inputs"][name]:
            raise ValueError(f"Target source mismatch: {name}")
    claims = {
        r["id"]: r
        for r in [
            json.loads(line) for line in (source / "claims_dev.jsonl").read_text().splitlines()
        ]
    }
    corpus = [json.loads(line) for line in (source / "corpus.jsonl").read_text().splitlines()]
    corpus.sort(key=lambda d: d["doc_id"])
    docs = [
        {
            "id": d["doc_id"],
            "text": " ".join(d["abstract"]),
            "language": "en",
            "source_rank": 5,
            "source_url": "https://github.com/allenai/scifact",
            "is_primary": False,
        }
        for d in corpus
    ]
    index = BM25Index([d["text"] for d in docs])
    earlier = json.loads((prior / "predictions.json").read_text())
    seed = identity["seeds"][0]
    cohort = [
        r
        for r in earlier
        if r["representation"] == "base_7" and r["seed"] == seed and r["rule"] == "argmax"
    ]
    if len({r["claim_id"] for r in cohort}) != len(cohort) or not cohort:
        raise ValueError("Invalid target cohort")
    audit = json.loads((prior / "retrieval_audit.json").read_text())
    scores = {}
    for row in audit:
        if len(row["premises"]) != len(row["nli_probabilities"]):
            raise ValueError("Unaligned prior NLI scores")
        for text, probabilities in zip(row["premises"], row["nli_probabilities"], strict=True):
            scores[row["claim_id"], text] = probabilities
    configs = {
        "full_heuristic": PipelineConfig(),
        "without_family_collapse": PipelineConfig(collapse_families=False),
        "without_source_weights": PipelineConfig(weight_sources=False),
        "without_score_thresholds": PipelineConfig(minimum_confidence=0, minimum_margin=0),
        "missing_primary_required": PipelineConfig(require_primary=True),
    }
    results, rows = [], []
    for variant, config in configs.items():
        selected_rows = []
        for j, record in enumerate(cohort):
            claim_id = record["claim_id"]

            def scorer(claim, premises, current_id=claim_id):
                try:
                    return [scores[current_id, text] for text in premises]
                except KeyError as exc:
                    raise ValueError(
                        "Required NLI pair absent from existing audit; no fabricated score"
                    ) from exc

            result = verify_claim(
                claims[claim_id]["claim"], "en", docs, scorer, config, retrieval_index=index
            )
            row = {
                "variant": variant,
                "claim_id": claim_id,
                "group": record["group"],
                "true_label": record["true_label"],
                **result,
            }
            rows.append(row)
            selected_rows.append(row)
            if j % 100 == 0:
                print(f"{variant}: {j + 1}/{len(cohort)}", flush=True)
        covered = [r for r in selected_rows if r["status"] == "machine_candidate"]
        y = [r["true_label"] for r in covered]
        pred = [r["candidate_label"] for r in covered]
        results.append(
            {
                "variant": variant,
                "config": asdict(config),
                "claims": len(cohort),
                "covered_claims": len(covered),
                "coverage": len(covered) / len(cohort),
                "covered_accuracy": float(accuracy_score(y, pred)) if covered else None,
                "covered_macro_f1": float(
                    f1_score(y, pred, labels=list(TARGET_LABELS), average="macro", zero_division=0)
                )
                if covered
                else None,
                "covered_per_class": classification_report(
                    y, pred, labels=list(TARGET_LABELS), output_dict=True, zero_division=0
                )
                if covered
                else None,
                "correct_candidates_per_all_claims": sum(
                    r["candidate_label"] == r["true_label"] for r in covered
                )
                / len(cohort),
                "abstention_counts": {
                    reason: sum(reason in r["abstention_reasons"] for r in selected_rows)
                    for reason in sorted(
                        {reason for r in selected_rows for reason in r["abstention_reasons"]}
                    )
                },
            }
        )
    full = [r for r in rows if r["variant"] == "full_heuristic"]
    no_weights = [r for r in rows if r["variant"] == "without_source_weights"]
    if any(
        not np.allclose(a["stance_scores"], b["stance_scores"])
        for a, b in zip(full, no_weights, strict=True)
    ):
        raise ValueError("Equal-source weights should not change stance")
    output.mkdir()
    write_json_atomic(
        output / "benchmark_summary.json",
        {
            "scope": "post-transfer exploratory selective prediction; not new confirmation",
            "results": results,
            "limitations": [
                "equal source ranks; no genuine source-weight efficacy test",
                (
                    "missing-primary scenario sets every primary flag false; "
                    "synthetic metadata gate test"
                ),
                "no new NLI inference; uncovered cached pairs cause failure",
                "coverage differs; covered accuracy is not full-set accuracy",
                "language tags assumed English; multilingual evaluation not included",
                "no learned sufficiency model or generative RAG evaluation",
            ],
        },
    )
    write_json_atomic(output / "predictions.json", rows)
    write_json_atomic(
        output / "input_manifest.json",
        {
            "prior_inputs": {
                n: sha256(prior / n)
                for n in ("retrieval_audit.json", "predictions.json", "input_manifest.json")
            },
            "source_inputs": identity["target_inputs"],
            "code_sha256": {
                n: sha256(root / n)
                for n in (
                    "scripts/run_source_pipeline_benchmark.py",
                    "src/apv_rag/source_pipeline.py",
                    "src/apv_rag/retrieval.py",
                    "src/apv_rag/direct_nli.py",
                )
            },
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.iterdir() if p.name != "output_manifest.json"},
    )
    print(output / "benchmark_summary.json")


if __name__ == "__main__":
    main()
