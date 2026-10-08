"""Run BM25 on benchmark-provided excerpts, not an open-web retrieval evaluation."""

import json
import math
from pathlib import Path

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS
from apv_rag.evidence_baseline import load_split_records
from apv_rag.retrieval import BM25Index
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/external/averitec" / DATASET_DIR_NAME / "train.json"
    if sha256(source) != SPLITS["train"].sha256:
        raise ValueError("upstream train.json checksum mismatch")
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    indices_path = split / "validation_indices.json"
    indices = json.loads(indices_path.read_text(encoding="utf-8"))
    records = load_split_records(json.loads(source.read_text(encoding="utf-8")), indices)
    documents, key_to_position, relevance = [], {}, []
    for record in records:
        relevant = set()
        for question in record.get("questions") or []:
            for answer in question.get("answers") or []:
                text = str(answer.get("answer") or "").strip()
                if not text:
                    continue
                url = str(answer.get("source_url") or "")
                key = (text, url)
                if key not in key_to_position:
                    key_to_position[key] = len(documents)
                    documents.append({"text": text, "source_url": url})
                relevant.add(key_to_position[key])
        relevance.append(relevant)
    index = BM25Index([document["text"] for document in documents])
    results, totals = [], {"recall_at_5": [], "recall_at_10": [], "ndcg_at_10": []}
    for upstream, record, relevant in zip(indices, records, relevance, strict=True):
        ranked = index.search(record["claim"], 10)
        positions = [position for position, _ in ranked]
        row = {"upstream_index": upstream, "relevant_ids": sorted(relevant), "ranking": ranked}
        if relevant:
            for k in (5, 10):
                value = len(set(positions[:k]) & relevant) / len(relevant)
                row[f"recall_at_{k}"] = value
                totals[f"recall_at_{k}"].append(value)
            dcg = sum(1 / math.log2(rank + 2) for rank, p in enumerate(positions) if p in relevant)
            ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(relevant))))
            row["ndcg_at_10"] = dcg / ideal
            totals["ndcg_at_10"].append(dcg / ideal)
        results.append(row)
    output = root / "artifacts/bm25_excerpt_diagnostic"
    if output.exists():
        raise FileExistsError("refusing to overwrite existing diagnostic artifact")
    output.mkdir(parents=True)
    for name, data in (("corpus.json", documents), ("rankings.json", results)):
        write_json_atomic(output / name, data)
    summary = {
        "scope": "oracle benchmark evidence-excerpt corpus; not open-web retrieval or NLI",
        "official_dev_records_used": 0,
        "validation_claims": len(records),
        "scored_claims": len(totals["recall_at_10"]),
        "unscored_no_nonempty_answer": len(records) - len(totals["recall_at_10"]),
        "unique_documents": len(documents),
        "metrics": {
            key: sum(values) / len(values) if values else None for key, values in totals.items()
        },
        "inputs": {
            "train_sha256": sha256(source),
            "validation_indices_sha256": sha256(indices_path),
        },
        "parameters": {"k1": 1.2, "b": 0.75, "top_k": 10},
        "outputs": {name: sha256(output / name) for name in ("corpus.json", "rankings.json")},
    }
    write_json_atomic(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
