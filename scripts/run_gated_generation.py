"""Gate before generation on frozen contexts, with cached and live backends."""
import argparse
import contextlib
import hashlib
import io
import json
import sqlite3
from importlib.metadata import version
from pathlib import Path

from apv_rag.gated_generation_flow import execute_generation
from apv_rag.generative_interface import allowed_next_tokens
from apv_rag.generative_rag import LABELS
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256, write_json_atomic
from run_sentence_rag_scifact import summarize
from verify_integrated_gate import main as verify_replay


def qwen_prompt(question):
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")


def live_backend(root, metadata, db):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = root / "models/qwen2.5-1.5b-instruct"
    if _model_files(path) != metadata["files"]:
        raise ValueError("Generator checksum mismatch")
    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32).eval()
    sequences = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(tokenizer.decode(s, skip_special_tokens=True).strip() != label for s, label in zip(sequences, LABELS, strict=True)):
        raise ValueError("Label tokenizer round-trip failed")

    def generate(kind, question):
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt != qwen_prompt(question):
            raise ValueError("Pinned chat template differs from the saved prompts")
        key = kind + ":" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = db.execute("SELECT response FROM answers WHERE key=?", (key,)).fetchone()
        if cached:
            return json.loads(cached[0])
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        count = int(tokens.input_ids.shape[1])
        if count > 1024:
            raise ValueError("Input budget exceeded; no silent truncation")
        options = {}
        if kind == "verdict":
            options["prefix_allowed_tokens_fn"] = lambda batch_id, ids: allowed_next_tokens(
                sequences, ids.tolist()[count:], tokenizer.eos_token_id)
        with torch.inference_mode():
            generated = model.generate(**tokens, **options, max_new_tokens=128, do_sample=False,
                                       num_beams=1, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(generated[0, count:], skip_special_tokens=True).strip()
        if kind == "verdict" and answer not in LABELS:
            raise ValueError("Constrained decoder failed")
        response = {"answer": answer, "prompt": prompt, "prompt_tokens": count}
        db.execute("INSERT INTO answers VALUES (?,?)", (key, json.dumps(response, ensure_ascii=False)))
        db.commit()
        return response

    return generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cached", "live"), default="cached")
    parser.add_argument("--policy", choices=("all", "no_gate", "nli", "embedding", "combined"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with contextlib.redirect_stdout(io.StringIO()):
        verify_replay()
    load = lambda folder, name: json.loads((folder / name).read_text(encoding="utf-8"))
    replay_folder = root / "artifacts/integrated_gate_received"
    replay = {(r["cohort"], str(r["claim_id"])): r for r in load(replay_folder, "predictions.json")}
    policies = ["no_gate", "nli", "embedding", "combined"] if args.policy == "all" else [args.policy]
    source_dirs = {"scifact": "sentence_rag_received", "climate_retrieved": "climate_rag_received",
                   "climate_supplied": "climate_supplied_received"}
    sources = {cohort: load(root / "artifacts" / dirname, "predictions.json") for cohort, dirname in source_dirs.items()}
    generator_metadata = [load(root / "artifacts" / dirname, "input_manifest.json")["models"]["generator"]
                          for dirname in source_dirs.values()]
    if any(m != generator_metadata[0] for m in generator_metadata[1:]):
        raise ValueError("Generation sources used different model identities")
    out = root / f"artifacts/gated_generation_{args.backend}_{args.policy}_v1"
    if (out / "output_manifest.json").exists():
        raise FileExistsError("Completed run exists; refusing overwrite")
    identity = {
        "backend": args.backend, "policies": policies,
        "replay_output_manifest_sha256": sha256(replay_folder / "output_manifest.json"),
        "generator": generator_metadata[0],
        "settings": {"threshold": .5, "seed": 369, "threads": 4, "dtype": "float32",
                     "max_input_tokens": 1024, "max_new_tokens": 128, "do_sample": False,
                     "num_beams": 1, "gate_features": "verified frozen-context cache"},
        "code_sha256": {n: sha256(root / n) for n in ("scripts/run_gated_generation.py",
            "scripts/verify_integrated_gate.py", "src/apv_rag/gated_generation_flow.py",
            "src/apv_rag/generative_interface.py", "src/apv_rag/integrated_gate.py",
            "src/apv_rag/numeric_integrity_v2.py", "src/apv_rag/input_numeric_integrity.py")},
        "protocol_sha256": sha256(root / "docs/GATED_GENERATION.md"),
        "packages": {n: version(n) for n in (("torch", "transformers", "numpy", "scikit-learn")
                     if args.backend == "live" else ("numpy", "scikit-learn"))},
    }
    out.mkdir(exist_ok=True)
    receipt = out / "input_manifest.json"
    if (out / "generation_cache.sqlite").exists() and not receipt.exists():
        raise ValueError("Cache without identity")
    if receipt.exists() and load(out, receipt.name) != identity:
        raise ValueError("Run identity changed; cache reuse refused")
    write_json_atomic(receipt, identity)
    predictions = []
    with sqlite3.connect(out / "generation_cache.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT)")
        live = live_backend(root, generator_metadata[0], db) if args.backend == "live" else None
        for cohort, rows in sources.items():
            for position, original in enumerate(rows):
                prior = replay[(cohort, str(original["claim_id"]))]

                def generate(kind, question):
                    # The context/claim is frozen; live explanations follow the new verdict.
                    prompt = qwen_prompt(question)
                    if (kind == "verdict" or live is None) and prompt != original[kind + "_prompt"]:
                        raise ValueError("Context/prompt changed; saved features cannot be reused")
                    if live is not None:
                        return live(kind, question)
                    return {"answer": original["generated_" + kind], "prompt": prompt,
                            "prompt_tokens": original[kind + "_prompt_tokens"]}

                for policy in policies:
                    probability = None if policy == "no_gate" else prior["gate_probabilities"][policy]
                    r = execute_generation(original["shown_claim"], original["evidence"], generate, probability)
                    if args.backend == "cached" and r["candidate_label"] != prior[policy + "_label"]:
                        raise ValueError("Controller decision differs from verified policy replay")
                    r.update(cohort=cohort, claim_id=original["claim_id"], policy=policy,
                             true_label=original["true_label"], backend=args.backend)
                    predictions.append(r)
                if position % 20 == 0:
                    print(f"{cohort}: {position + 1}/{len(rows)}", flush=True)
        unique_live_answers = db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    summary = {}
    for cohort in sources:
        summary[cohort] = {}
        for policy in policies:
            subset = [r for r in predictions if r["cohort"] == cohort and r["policy"] == policy]
            summary[cohort][policy] = {
                "metrics": summarize(subset, "candidate_label"),
                "gate_rejected_before_generation": sum("gate_below_threshold" in r["reasons"] for r in subset),
                "policy_verdict_requests": sum("verdict" in r["generation_requests"] for r in subset),
                "policy_explanation_requests": sum("explanation" in r["generation_requests"] for r in subset),
            }
    write_json_atomic(out / "predictions.json", predictions)
    write_json_atomic(out / "summary.json", {"backend": args.backend, "cohorts": summary,
        "unique_live_answers_cached": unique_live_answers,
        "scope": "frozen contexts/features; gate precedes generation; request counts are not measured runtime savings"})
    write_json_atomic(out / "output_manifest.json", {p.name: sha256(p) for p in out.glob("*.json") if p.name != "output_manifest.json"})
    print(out)


if __name__ == "__main__":
    main()
