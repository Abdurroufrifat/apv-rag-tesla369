"""Live Qwen repeated-source/order stress with frozen baseline responses."""
import contextlib
import hashlib
import io
import json
import sqlite3
from importlib.metadata import version
from pathlib import Path

from apv_rag.generation_robustness import CONDITIONS, context_variants, explanation_ids, question
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2
from apv_rag.splits import sha256, write_json_atomic
from download_instruction_model import MODEL_ID, REVISION
from run_gated_generation import live_backend, qwen_prompt
from run_sentence_rag_scifact import summarize
from verify_gated_generation import main as verify_controller


def prepare_sources(root):
    with contextlib.redirect_stdout(io.StringIO()):
        verify_controller()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    provider = root / "artifacts/gated_generation_received"
    metadata = load(provider / "input_manifest.json")
    baseline = {(r["cohort"], str(r["claim_id"])): r for r in load(provider / "predictions.json")
                if r["policy"] == "no_gate"}
    sources = {cohort: load(root / "artifacts" / dirname / "predictions.json") for cohort, dirname in
               (("scifact", "sentence_rag_received"), ("climate_retrieved", "climate_rag_received"))}
    selected = {cohort: explanation_ids(rows, cohort) for cohort, rows in sources.items()}
    for cohort, rows in sources.items():
        for r in rows:
            context_variants(r["evidence"])
            prior = baseline[cohort, str(r["claim_id"])]
            if qwen_prompt(question(r["shown_claim"], r["evidence"])) != prior["verdict_prompt"]:
                raise ValueError("Baseline prompt/source mismatch")
    return sources, baseline, selected, metadata


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts/generation_robustness_v1"
    if (out / "output_manifest.json").exists():
        raise FileExistsError("Completed run exists; refusing overwrite")
    sources, baseline, selected, metadata = prepare_sources(root)
    model_metadata = metadata["generator"]
    if model_metadata["model_id"] != MODEL_ID or model_metadata["revision"] != REVISION:
        raise ValueError("Pinned generator required")
    packages = {n: version(n) for n in ("torch", "transformers", "numpy", "scikit-learn")}
    if any(packages[n] != metadata["packages"][n] for n in packages):
        raise ValueError("Original controller package versions required for baseline reuse")
    identity = {
        "provider_output_manifest_sha256": sha256(root / "artifacts/gated_generation_received/output_manifest.json"),
        "source_sha256": {name: sha256(root / "artifacts" / dirname / "predictions.json") for name, dirname in
                          (("scifact", "sentence_rag_received"), ("climate_retrieved", "climate_rag_received"))},
        "generator": model_metadata, "packages": packages, "explanation_claim_ids": selected,
        "settings": {"copy_count": 3, "conditions": list(CONDITIONS), "explanation_claims_per_cohort": 60,
                     "seed": 369, "threads": 4, "dtype": "float32", "max_input_tokens": 1024,
                     "max_new_tokens": 128, "do_sample": False, "num_beams": 1,
                     "learned_gate": "none; source normalization control only", "context": "original frozen clipped text"},
        "code_sha256": {n: sha256(root / n) for n in ("scripts/run_generation_robustness.py",
            "src/apv_rag/generation_robustness.py", "scripts/run_gated_generation.py", "scripts/verify_gated_generation.py",
            "src/apv_rag/integrated_gate.py", "src/apv_rag/generative_interface.py",
            "src/apv_rag/numeric_integrity_v2.py", "src/apv_rag/input_numeric_integrity.py")},
        "protocol_sha256": sha256(root / "docs/GENERATION_ROBUSTNESS.md"),
    }
    out.mkdir(exist_ok=True)
    receipt = out / "input_manifest.json"
    if (out / "generation_cache.sqlite").exists() and not receipt.exists():
        raise ValueError("Cache without identity")
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != identity:
        raise ValueError("Inputs/code/packages changed; cache reuse refused")
    write_json_atomic(receipt, identity)
    predictions = []
    seeded_keys = set()
    with sqlite3.connect(out / "generation_cache.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT)")
        for cohort, rows in sources.items():
            selected_ids = set(selected[cohort])
            for source in rows:
                prior = baseline[cohort, str(source["claim_id"])]
                kinds = ("verdict", "explanation") if str(source["claim_id"]) in selected_ids else ("verdict",)
                for kind in kinds:
                    prompt = prior[kind + "_prompt"]
                    key = kind + ":" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    response = {"answer": prior["generated_" + kind], "prompt": prompt,
                                "prompt_tokens": prior[kind + "_prompt_tokens"]}
                    cached = db.execute("SELECT response FROM answers WHERE key=?", (key,)).fetchone()
                    if cached and json.loads(cached[0]) != response:
                        raise ValueError("Baseline cache conflict")
                    db.execute("INSERT OR IGNORE INTO answers VALUES (?,?)", (key, json.dumps(response, ensure_ascii=False)))
                    seeded_keys.add(key)
        db.commit()
        generate = live_backend(root, model_metadata, db)
        for cohort, rows in sources.items():
            selected_ids = set(selected[cohort])
            for position, source in enumerate(rows):
                for condition, evidence in context_variants(source["evidence"]).items():
                    response = generate("verdict", question(source["shown_claim"], evidence))
                    verdict = response["answer"]
                    is_selected = str(source["claim_id"]) in selected_ids
                    explanation = None
                    numeric = None
                    reasons = []
                    if is_selected:
                        explanation = generate("explanation", question(source["shown_claim"], evidence, verdict))
                        numeric = numeric_provenance_v2(explanation["answer"] or "", source["shown_claim"], evidence)
                        if not explanation["answer"]:
                            reasons.append("empty_explanation")
                        if numeric["absent_from_inputs"]:
                            reasons.append("numeric_value_absent")
                    prior = baseline[cohort, str(source["claim_id"])]
                    if condition in ("baseline", "copies_collapsed"):
                        if response["answer"] != prior["generated_verdict"] or response["prompt"] != prior["verdict_prompt"]:
                            raise ValueError("Normalized control differs from baseline")
                        if explanation and explanation["answer"] != prior["generated_explanation"]:
                            raise ValueError("Normalized explanation differs from baseline")
                    predictions.append({"cohort": cohort, "claim_id": source["claim_id"], "condition": condition,
                        "shown_claim": source["shown_claim"], "evidence": evidence,
                        "generated_verdict": verdict, "raw_candidate_label": verdict,
                        "verdict_prompt": response["prompt"], "verdict_prompt_tokens": response["prompt_tokens"],
                        "explanation_selected": is_selected, "generated_explanation": explanation["answer"] if explanation else None,
                        "explanation_prompt": explanation["prompt"] if explanation else None,
                        "explanation_prompt_tokens": explanation["prompt_tokens"] if explanation else None,
                        "numeric_provenance": numeric, "reasons": reasons,
                        "explanation_guarded_label": verdict if is_selected and not reasons else None,
                        "true_label": source["true_label"]})
                if position % 20 == 0:
                    print(f"{cohort}: {position + 1}/{len(rows)}", flush=True)
        unique_answers = db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    summaries = {}
    for cohort in sources:
        summaries[cohort] = {}
        for condition in CONDITIONS:
            subset = [r for r in predictions if r["cohort"] == cohort and r["condition"] == condition]
            explained = [r for r in subset if r["explanation_selected"]]
            summaries[cohort][condition] = {"rows": len(subset), "verdict_metrics": summarize(subset, "raw_candidate_label"),
                "verdict_disagreement_with_baseline": sum(r["raw_candidate_label"] != baseline[cohort, str(r["claim_id"])]["generated_verdict"] for r in subset) / len(subset),
                "explained_rows": len(explained), "explanation_subset_raw_verdict_metrics": summarize(explained, "raw_candidate_label"),
                "explanation_subset_guarded_metrics": summarize(explained, "explanation_guarded_label"),
                "numeric_rejections": sum("numeric_value_absent" in r["reasons"] for r in explained)}
    write_json_atomic(out / "predictions.json", predictions)
    write_json_atomic(out / "summary.json", {"cohorts": summaries, "frozen_baseline_prompt_keys_seeded": len(seeded_keys),
        "total_cached_prompt_keys": unique_answers, "new_live_prompt_keys": unique_answers - len(seeded_keys),
        "scope": "synthetic exact-copy/order generator stress on frozen retrieved contexts; no source authentication or learned gate"})
    write_json_atomic(out / "output_manifest.json", {p.name: sha256(p) for p in out.glob("*.json") if p.name != "output_manifest.json"})
    print(out)


if __name__ == "__main__":
    main()
