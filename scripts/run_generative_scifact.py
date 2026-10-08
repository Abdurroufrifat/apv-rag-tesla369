"""Fixed BM25 plus local text generation on the existing SciFact cohort."""

import json
import sqlite3
from importlib.metadata import version
from pathlib import Path

import numpy as np
from download_generative_model import MODEL_ID, REVISION
from sklearn.metrics import accuracy_score, f1_score

from apv_rag.generative_rag import LABELS, make_prompt, parse_answer
from apv_rag.nli_comparison import _model_files
from apv_rag.retrieval import BM25Index
from apv_rag.scifact import target_label
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/generative_scifact"
    if (output / "output_manifest.json").exists():
        raise FileExistsError("Completed generation exists; refusing overwrite")
    source = root / "data/external/scifact/sealed_v1"
    manifest = json.loads((source / "manifest.json").read_text())
    for n in ("corpus.jsonl", "claims_dev.jsonl"):
        if sha256(source / n) != manifest["files"][n]:
            raise ValueError("SciFact checksum mismatch")
    prior = root / "artifacts/scifact_transfer"
    hashes = json.loads((prior / "output_manifest.json").read_text())
    if sha256(prior / "predictions.json") != hashes["predictions.json"]:
        raise ValueError("Prior cohort checksum mismatch")
    previous = json.loads((prior / "predictions.json").read_text())
    seed = min(r["seed"] for r in previous)
    cohort = [
        r["claim_id"]
        for r in previous
        if r["seed"] == seed and r["representation"] == "base_7" and r["rule"] == "argmax"
    ]
    if len(cohort) != 300 or len(set(cohort)) != 300:
        raise ValueError("Unexpected evaluation cohort")
    model_path = root / "models/flan-t5-base"
    model_meta = json.loads((root / "models/generative_model_manifest.json").read_text())
    if model_meta["model_id"] != MODEL_ID or model_meta["revision"] != REVISION:
        raise ValueError("Pinned model required")
    if _model_files(model_path) != model_meta["files"]:
        raise ValueError("Model checksum mismatch")
    identity = {
        "source_sha256": {n: sha256(source / n) for n in ("corpus.jsonl", "claims_dev.jsonl")},
        "cohort_sha256": sha256(prior / "predictions.json"),
        "model": model_meta,
        "packages": {n: version(n) for n in ("torch", "transformers", "numpy", "scikit-learn")},
        "settings": {
            "top_k": 3,
            "abstract_token_limit": 96,
            "claim_token_limit": 64,
            "max_input_tokens": 512,
            "max_new_tokens": 96,
            "do_sample": False,
            "num_beams": 1,
            "seed": 369,
            "threads": 4,
            "dtype": "float32",
        },
        "code_sha256": {
            n: sha256(root / n)
            for n in (
                "scripts/run_generative_scifact.py",
                "scripts/download_generative_model.py",
                "src/apv_rag/generative_rag.py",
                "src/apv_rag/retrieval.py",
                "src/apv_rag/scifact.py",
            )
        },
        "protocol_sha256": sha256(root / "docs/GENERATIVE_SCIFACT.md"),
    }
    output.mkdir(exist_ok=True)
    receipt = output / "input_manifest.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Inputs changed; refusing cached run reuse")
    write_json_atomic(receipt, identity)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    claims = {
        r["id"]: r
        for r in [json.loads(x) for x in (source / "claims_dev.jsonl").read_text().splitlines()]
    }
    corpus = sorted(
        [json.loads(x) for x in (source / "corpus.jsonl").read_text().splitlines()],
        key=lambda d: d["doc_id"],
    )
    abstracts = [" ".join(d["abstract"]) for d in corpus]
    index = BM25Index(abstracts)

    def clipped(text, limit):
        return tokenizer.decode(
            tokenizer.encode(text, add_special_tokens=False)[:limit], skip_special_tokens=True
        )

    rows = []
    with sqlite3.connect(output / "generation_cache.sqlite") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS answers (id INTEGER PRIMARY KEY, text TEXT)")
        for position, claim_id in enumerate(cohort):
            claim = claims[claim_id]["claim"]
            hits = index.search(claim, top_k=3)
            evidence = [
                {
                    "id": corpus[i]["doc_id"],
                    "text": clipped(abstracts[i], 96),
                    "bm25_score": float(score),
                }
                for i, score in hits
                if score > 0
            ]
            prompt = make_prompt(clipped(claim, 64), evidence)
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            if tokens.input_ids.shape[1] > 512:
                raise ValueError("Prompt exceeds fixed input budget; no silent truncation")
            cached = connection.execute(
                "SELECT text FROM answers WHERE id=?", (claim_id,)
            ).fetchone()
            if cached:
                answer = cached[0]
            else:
                with torch.inference_mode():
                    generated = model.generate(
                        **tokens, max_new_tokens=96, do_sample=False, num_beams=1
                    )
                answer = tokenizer.decode(generated[0], skip_special_tokens=True)
                connection.execute("INSERT INTO answers VALUES (?,?)", (claim_id, answer))
                connection.commit()
            parsed = parse_answer(answer, [d["id"] for d in evidence])
            rows.append(
                {
                    "claim_id": claim_id,
                    "claim": claim,
                    "prompt": prompt,
                    "prompt_tokens": int(tokens.input_ids.shape[1]),
                    "evidence": evidence,
                    "raw_generation": answer,
                    "result": parsed,
                    "true_label": target_label(claims[claim_id]),
                }
            )
            if position % 10 == 0:
                print(f"Generation {position + 1}/{len(cohort)}", flush=True)
    truth = [r["true_label"] for r in rows]
    predicted = [r["result"]["candidate_label"] or "abstain" for r in rows]
    covered = [i for i, label in enumerate(predicted) if label != "abstain"]
    summary = {
        "scope": "exploratory BM25 plus FLAN-T5 generation; structural citations only",
        "rows": len(rows),
        "coverage": len(covered) / len(rows),
        "accuracy_all_claims_abstentions_as_errors": float(accuracy_score(truth, predicted)),
        "macro_f1_all_claims_abstentions_as_errors": float(
            f1_score(truth, predicted, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "covered_accuracy": float(np.mean([truth[i] == predicted[i] for i in covered]))
        if covered
        else None,
        "invalid_outputs": len(rows) - len(covered),
        "no_target_fitting": True,
        "limitations": [
            "citation existence does not establish entailment",
            "SciFact results already observed; exploratory evaluation",
            "pretraining overlap cannot be ruled out",
            "no new human gold annotations or archival authentication",
        ],
    }
    write_json_atomic(output / "predictions.json", rows)
    write_json_atomic(output / "generation_summary.json", summary)
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.glob("*.json") if p.name != "output_manifest.json"},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
