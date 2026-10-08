"""Diagnose split-local retrieval without opening official dev or neural models."""

import json
from pathlib import Path

from apv_rag.averitec import SPLITS
from apv_rag.evidence_baseline import LABELS
from apv_rag.nli_comparison import pool_corpus
from apv_rag.retrieval import BM25Index
from apv_rag.retrieval_diagnostics import ownership_metrics
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    if sha256(source) != SPLITS["train"].sha256:
        raise ValueError("upstream train checksum mismatch")
    records = json.loads(source.read_text(encoding="utf-8"))
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    original = root / "artifacts/retrieval_nli_comparison"
    output = root / "artifacts/retrieval_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    results, inputs = {}, {"train.json": sha256(source)}
    for name in ("train", "validation"):
        path = split / f"{name}_indices.json"
        inputs[path.name] = sha256(path)
        ids = json.loads(path.read_text())
        rows = [records[i] for i in ids]
        documents = pool_corpus(rows)
        bm25 = BM25Index([d["text"] for d in documents])
        rankings = [[i for i, _ in bm25.search(row["claim"], 5)] for row in rows]
        methods = {"bm25_nli": rankings}
        dense_path = original / f"{name}_dense_nli_retrieval_audit.json"
        if dense_path.exists():
            manifest = json.loads((original / "output_manifest.json").read_text())
            if sha256(dense_path) != manifest[dense_path.name]:
                raise ValueError("dense retrieval audit checksum mismatch")
            corpus_path = original / f"{name}_corpus.json"
            if (
                sha256(corpus_path) != manifest[corpus_path.name]
                or json.loads(corpus_path.read_text(encoding="utf-8")) != documents
            ):
                raise ValueError("dense audit corpus differs from reconstructed corpus")
            audit = json.loads(dense_path.read_text())
            if len(audit) != len(rows) or [r["record_position"] for r in audit] != list(
                range(len(rows))
            ):
                raise ValueError("dense audit record alignment mismatch")
            methods["dense_nli"] = [r["document_ids"] for r in audit]
            inputs[dense_path.name] = sha256(dense_path)
        results[name] = {}
        for method, ranked in methods.items():
            result = ownership_metrics(rows, documents, ranked)
            result["by_class"] = {}
            for label in LABELS:
                positions = [i for i, row in enumerate(rows) if row["label"] == label]
                result["by_class"][label] = ownership_metrics(
                    [rows[i] for i in positions], documents, [ranked[i] for i in positions]
                )
            results[name][method] = result
    report = {
        "scope": "split-local oracle answer-excerpt retrieval diagnostic",
        "official_dev_records_read": 0,
        "denominator": "claims containing at least one nonempty answer excerpt",
        "interpretation": "ownership proxy; does not measure relevance or sufficiency",
        "dense_status": "requires original dense retrieval audits and corpus",
        "input_sha256": inputs,
        "results": results,
    }
    write_json_atomic(output / "retrieval_diagnostics.json", report)
    lines = [
        "# Retrieval ownership diagnostics",
        "",
        "Split-local oracle answer-excerpt corpora. Official dev is not read.",
        "",
        "| Split | Method | Eligible claims | Hit@1 | Hit@5 | MRR@5 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, methods in results.items():
        for method, r in methods.items():
            lines.append(
                f"| {name} | {method} | {r['claims_with_answer_excerpts']} | "
                f"{r['own_excerpt_hit_at_1']:.3f} | {r['own_excerpt_hit_at_5']:.3f} | "
                f"{r['own_excerpt_mrr_at_5']:.3f} |"
            )
    lines.extend(
        [
            "",
            "A hit means an excerpt was attached to that claim in the benchmark.",
            "Other claims' excerpts may be relevant. Shared excerpts can belong to several claims.",
            "Ownership proxies do not establish evidence sufficiency or failure causes.",
            "Dense results require original audits and corpus. Missing results are not zero.",
            "Per-class results and excerpt coverage are saved in the JSON report.",
            "",
        ]
    )
    (output / "RETRIEVAL_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
