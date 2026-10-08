"""Frozen multilingual Qwen verdict benchmark and explanation subset."""
import json
import sqlite3
from importlib.metadata import version
from pathlib import Path

from apv_rag.multilingual_generation import explanation_ids, explanation_question, verdict_question
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2
from apv_rag.splits import sha256, write_json_atomic
from apv_rag.xfever import LABEL_MAP, validate_parallel
from download_instruction_model import MODEL_ID, REVISION
from run_gated_generation import live_backend
from run_sentence_rag_scifact import summarize

FILES = ["en/test.6h.jsonl"] + [f"{language}/test.6h{suffix}.jsonl"
         for language in ("es", "fr", "id", "ja", "zh") for suffix in ("", ".human")]
ARCHIVE_SHA256 = "8b7948894c8724d9a52e86e18fe0e369c5d58b12ddba853e836b8d80611a4895"


def prepare_inputs(root):
    folder = root / "data/external/xfever/zenodo_8206962/evaluation_inputs_v1"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Unexpected source archive")
    files = {}
    sets = {}
    for name in FILES:
        files[name] = sha256(folder / name)
        if files[name] != manifest["files"]["data/" + name]["sha256"]:
            raise ValueError(f"Input checksum mismatch: {name}")
        sets[name] = [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines()]
    if validate_parallel(sets) != 600:
        raise ValueError("Expected 600 aligned rows per file")
    ids = explanation_ids(sets[FILES[0]])
    return sets, files, ids, sha256(folder / "manifest.json")


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts/xfever_generation_v1"
    if (out / "output_manifest.json").exists():
        raise FileExistsError("Completed run exists; refusing overwrite")
    sets, file_hashes, ids, input_manifest_hash = prepare_inputs(root)
    selected = set(ids)
    metadata_path = root / "models/instruction_model_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model_id"] != MODEL_ID or metadata["revision"] != REVISION:
        raise ValueError("Pinned Qwen model required")
    identity = {
        "files": file_hashes, "dataset_manifest_sha256": input_manifest_hash,
        "generator": metadata, "model_manifest_sha256": sha256(metadata_path),
        "explanation_claim_ids": ids,
        "settings": {"claim_tokens": 192, "evidence_tokens": 512, "max_input_tokens": 1024,
            "max_new_tokens": 128, "seed": 369, "threads": 4, "dtype": "float32", "do_sample": False,
            "num_beams": 1, "rows_per_file": 600, "explanation_unique_claims": 60,
            "gate": "none; multilingual completeness transfer unvalidated"},
        "packages": {n: version(n) for n in ("torch", "transformers", "numpy", "scikit-learn")},
        "code_sha256": {n: sha256(root / n) for n in ("scripts/run_xfever_generation.py",
            "src/apv_rag/multilingual_generation.py", "scripts/run_gated_generation.py",
            "scripts/download_instruction_model.py", "src/apv_rag/generative_interface.py",
            "src/apv_rag/nli_comparison.py", "src/apv_rag/xfever.py",
            "src/apv_rag/numeric_integrity_v2.py", "src/apv_rag/input_numeric_integrity.py")},
        "protocol_sha256": sha256(root / "docs/XFEVER_GENERATION.md"),
    }
    out.mkdir(exist_ok=True)
    receipt = out / "input_manifest.json"
    if (out / "generation_cache.sqlite").exists() and not receipt.exists():
        raise ValueError("Cache without identity")
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != identity:
        raise ValueError("Run identity changed; cache reuse refused")
    write_json_atomic(receipt, identity)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(root / "models/qwen2.5-1.5b-instruct", local_files_only=True)

    def clip(text, limit):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(tokens[:limit], skip_special_tokens=True), len(tokens), len(tokens[:limit])

    predictions = []
    with sqlite3.connect(out / "generation_cache.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT)")
        generate = live_backend(root, metadata, db)
        for name, rows in sets.items():
            language = name.split("/")[0]
            for position, source in enumerate(rows):
                claim, original_claim_tokens, shown_claim_tokens = clip(source["claim"], 192)
                evidence, original_evidence_tokens, shown_evidence_tokens = clip(source["evidence"], 512)
                response = generate("verdict", verdict_question(claim, evidence))
                verdict = response["answer"]
                selected_for_explanation = str(source["id"]) in selected
                explanation = None
                numeric = None
                reasons = []
                if selected_for_explanation:
                    explanation = generate("explanation", explanation_question(claim, evidence, verdict, language))
                    if not explanation["answer"]:
                        reasons.append("empty_explanation")
                    numeric = numeric_provenance_v2(explanation["answer"] or "", claim,
                                                    [{"id": "premise", "text": evidence}])
                    if numeric["absent_from_inputs"]:
                        reasons.append("numeric_value_absent")
                predictions.append({
                    "file": name, "row": position, "claim_id": source["id"],
                    "english_page": sets[FILES[0]][position]["page"], "language": language,
                    "translation_origin": "original" if language == "en" else ("upstream_human" if ".human." in name else "upstream_machine"),
                    "shown_claim": claim, "shown_evidence": evidence,
                    "original_claim_tokens": original_claim_tokens, "shown_claim_tokens": shown_claim_tokens,
                    "original_evidence_tokens": original_evidence_tokens, "shown_evidence_tokens": shown_evidence_tokens,
                    "claim_clipped": original_claim_tokens > 192, "evidence_clipped": original_evidence_tokens > 512,
                    "verdict_prompt": response["prompt"], "verdict_prompt_tokens": response["prompt_tokens"],
                    "generated_verdict": verdict, "raw_candidate_label": verdict,
                    "explanation_selected": selected_for_explanation,
                    "generated_explanation": explanation["answer"] if explanation else None,
                    "explanation_prompt": explanation["prompt"] if explanation else None,
                    "explanation_prompt_tokens": explanation["prompt_tokens"] if explanation else None,
                    "explanation_guarded_label": verdict if selected_for_explanation and not reasons else None,
                    "numeric_provenance": numeric, "reasons": reasons, "true_label": LABEL_MAP[source["label"]],
                })
                if position % 50 == 0:
                    print(f"{name}: {position + 1}/{len(rows)}", flush=True)
        unique_live_answers = db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    english = {r["row"]: r["raw_candidate_label"] for r in predictions if r["file"] == FILES[0]}
    summaries = []
    for name in FILES:
        subset = [r for r in predictions if r["file"] == name]
        explained = [r for r in subset if r["explanation_selected"]]
        summaries.append({
            "file": name, "rows": len(subset), "verdict_metrics": summarize(subset, "raw_candidate_label"),
            "verdict_disagreement_with_english": sum(r["raw_candidate_label"] != english[r["row"]] for r in subset) / len(subset),
            "claim_clipped": sum(r["claim_clipped"] for r in subset), "evidence_clipped": sum(r["evidence_clipped"] for r in subset),
            "explained_rows": len(explained), "explained_unique_claims": len({r["claim_id"] for r in explained}),
            "explanation_subset_raw_verdict_metrics": summarize(explained, "raw_candidate_label"),
            "explanation_subset_guarded_metrics": summarize(explained, "explanation_guarded_label"),
            "numeric_rejections": sum("numeric_value_absent" in r["reasons"] for r in explained),
        })
    write_json_atomic(out / "predictions.json", predictions)
    write_json_atomic(out / "summary.json", {"files": summaries, "unique_live_answers_cached": unique_live_answers,
        "scope": "full aligned supplied-evidence verdict benchmark; fixed 60-ID explanation subset; no retrieval or learned gate",
        "no_target_fitting": True, "explanation_language_verified": False,
        "explanation_correctness_verified": False})
    write_json_atomic(out / "output_manifest.json", {p.name: sha256(p) for p in out.glob("*.json") if p.name != "output_manifest.json"})
    print(out)


if __name__ == "__main__":
    main()
