"""Exploratory constrained-verdict Qwen RAG; preserves the original run."""

import json
import re
import sqlite3
from importlib.metadata import version
from pathlib import Path

import numpy as np
from download_instruction_model import MODEL_ID, REVISION
from sklearn.metrics import accuracy_score, f1_score

from apv_rag.generative_rag import LABELS
from apv_rag.generative_interface import allowed_next_tokens
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2 as numeric_provenance
from apv_rag.multilingual_nli import cen_order, normalize_model_scores
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256, write_json_atomic


def summarize(rows, field):
    truth = [r["true_label"] for r in rows]
    predicted = [r[field] or "abstain" for r in rows]
    covered = [i for i, p in enumerate(predicted) if p != "abstain"]
    return {
        "coverage": len(covered) / len(rows),
        "accuracy_all_claims_abstentions_as_errors": float(accuracy_score(truth, predicted)),
        "macro_f1_all_claims_abstentions_as_errors": float(
            f1_score(truth, predicted, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "covered_accuracy": float(np.mean([truth[i] == predicted[i] for i in covered]))
        if covered
        else None,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/climate_supplied_evidence_v1"
    if (output / "output_manifest.json").exists():
        raise FileExistsError("Completed run exists; refusing overwrite")
    source = root / "data/external/climate_fever/frozen_v1"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name in ("claims_dev.jsonl", "corpus.jsonl"):
        if sha256(source / name) != manifest["files"][name]:
            raise ValueError("SciFact checksum mismatch")
    cohort = manifest["cohort"]
    if len(cohort) != 300 or len(set(cohort)) != 300:
        raise ValueError("Unexpected frozen cohort")
    generator_path = root / "models/qwen2.5-1.5b-instruct"
    nli_path = root / "models/nli-deberta-v3-small"
    gen_meta = json.loads((root / "models/instruction_model_manifest.json").read_text(encoding="utf-8"))
    original = root / "artifacts/retrieval_nli_comparison"
    old_hashes = json.loads((original / "output_manifest.json").read_text(encoding="utf-8"))
    if sha256(original / "input_manifest.json") != old_hashes["input_manifest.json"]:
        raise ValueError("NLI metadata mismatch")
    nli_meta = json.loads((original / "input_manifest.json").read_text(encoding="utf-8"))
    if gen_meta["model_id"] != MODEL_ID or gen_meta["revision"] != REVISION:
        raise ValueError("Pinned generator required")
    if _model_files(generator_path) != gen_meta["files"]:
        raise ValueError("Generator checksum mismatch")
    if _model_files(nli_path) != nli_meta["models"]["nli"]:
        raise ValueError("NLI checksum mismatch")
    identity = {
        "source_sha256": {n: sha256(source / n) for n in ("corpus.jsonl", "claims_dev.jsonl", "climate-fever.jsonl")},
        "cohort_sha256": sha256(source / "manifest.json"),
        "models": {"generator": gen_meta, "nli": nli_meta["models"]["nli"]},
        "packages": {n: version(n) for n in ("torch", "transformers", "numpy", "scikit-learn")},
        "settings": {
            "verdict_decoder": "greedy label-token trie; EOS only after complete label",
            "evidence_mode": "all five claim-associated annotated sentences; labels hidden",
            "top_k": 5,
            "passage_tokens": 96,
            "sentence_selection": "none; all five supplied sentences in dataset order; clip96",
            "claim_tokens": 64,
            "max_input_tokens": 1024,
            "max_new_tokens": 128,
            "seed": 369,
            "threads": 4,
            "dtype": "float32",
            "do_sample": False,
        },
        "code_sha256": {
            n: sha256(root / n)
            for n in (
                "scripts/run_climate_supplied_evidence.py",
                "src/apv_rag/numeric_integrity_v2.py",
                "src/apv_rag/generative_interface.py",
                "src/apv_rag/sentence_context.py",
                "src/apv_rag/retrieval.py",
                "src/apv_rag/scifact.py",
                "src/apv_rag/multilingual_nli.py",
            )
        },
        "protocol_sha256": sha256(root / "docs/CLIMATE_SUPPLIED_EVIDENCE.md"),
    }
    output.mkdir(exist_ok=True)
    receipt = output / "input_manifest.json"
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != identity:
        raise ValueError("Inputs changed; cache reuse refused")
    write_json_atomic(receipt, identity)
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    gt = AutoTokenizer.from_pretrained(generator_path, local_files_only=True)
    generator = AutoModelForCausalLM.from_pretrained(
        generator_path, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    nt = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
    nli = AutoModelForSequenceClassification.from_pretrained(
        nli_path, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    order = cen_order(nli.config.id2label)
    claims = {
        r["id"]: r
        for r in [json.loads(x) for x in (source / "claims_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    }
    def clip(text, limit):
        return gt.decode(
            gt.encode(text, add_special_tokens=False)[:limit], skip_special_tokens=True
        )

    raw_path = source / "climate-fever.jsonl"
    if sha256(raw_path) != manifest["raw_sha256"]:
        raise ValueError("Raw annotation checksum mismatch")
    supplied = {r["claim_id"]: r for r in map(json.loads, raw_path.read_text(encoding="utf-8").splitlines())}
    rows = []
    with sqlite3.connect(output / "generation_cache.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, answer TEXT)")

        def generate(key, question):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
            prompt = gt.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = gt.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if tokens.input_ids.shape[1] > 1024:
                raise ValueError("Input budget exceeded; no silent truncation")
            cached = db.execute("SELECT answer FROM answers WHERE key=?", (key,)).fetchone()
            if cached:
                answer = cached[0]
            else:
                decode_options = {}
                if key.endswith(":verdict"):
                    sequences = [gt.encode(label, add_special_tokens=False) for label in LABELS]
                    if any(gt.decode(seq, skip_special_tokens=True).strip() != label
                           for seq, label in zip(sequences, LABELS, strict=True)):
                        raise ValueError("Label tokenizer round-trip failed")
                    prompt_length = int(tokens.input_ids.shape[1])

                    def allowed(batch_id, ids):
                        return allowed_next_tokens(
                            sequences, ids.tolist()[prompt_length:], gt.eos_token_id
                        )

                    decode_options["prefix_allowed_tokens_fn"] = allowed
                with torch.inference_mode():
                    generated = generator.generate(
                        **tokens,
                        **decode_options,
                        max_new_tokens=128,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=gt.eos_token_id,
                    )
                answer = gt.decode(
                    generated[0, tokens.input_ids.shape[1] :], skip_special_tokens=True
                ).strip()
                if key.endswith(":verdict") and answer not in LABELS:
                    raise ValueError("Constrained decoder failed; refusing cache write")
                db.execute("INSERT INTO answers VALUES (?,?)", (key, answer))
                db.commit()
            return prompt, answer, int(tokens.input_ids.shape[1])

        for position, claim_id in enumerate(cohort):
            claim = claims[claim_id]["claim"]
            shown_claim = clip(claim, 64)
            evidence = [
                {"id": e["evidence_id"], "article": e["article"],
                 "text": clip(e["evidence"], 96)}
                for e in supplied[claim_id]["evidences"]
            ]
            if len(evidence) != 5:
                raise ValueError("Expected exactly five supplied sentences")
            context = "\n".join(f"[{d['id']}] {d['text']}" for d in evidence)
            verdict, explanation, vp, ep, vtokens, etokens = None, None, None, None, None, None
            reasons = []
            if not evidence:
                reasons.append("no_evidence")
            else:
                vp, verdict, vtokens = generate(
                    f"{claim_id}:verdict",
                    f"Evidence: {context}\nClaim: {shown_claim}\n"
                    "Reply only Supported, Refuted, or Not Enough Evidence.",
                )
                if verdict not in LABELS:
                    reasons.append("invalid_verdict")
                else:
                    ep, explanation, etokens = generate(
                        f"{claim_id}:explanation",
                        f"Evidence: {context}\nClaim: {shown_claim}\nVerdict: {verdict}\n"
                        "Explain why this evidence supports, contradicts, or cannot establish "
                        "the claim. Use only the supplied evidence and keep the explanation short.",
                    )
                    if not explanation:
                        reasons.append("empty_explanation")
            provenance = numeric_provenance(explanation or "", shown_claim, evidence)
            if provenance["absent_from_inputs"]:
                reasons.append("novel_numeric_value")
            sentence_audit = []
            if explanation:
                for sentence in re.split(r"(?<=[.!?])\s+", explanation):
                    pairs = nt(
                        [d["text"] for d in evidence],
                        [sentence] * len(evidence),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256,
                    )
                    with torch.inference_mode():
                        scores = nli(**pairs).logits.float().softmax(-1).cpu().numpy()[:, order]
                    scores = normalize_model_scores(scores.tolist())
                    sentence_audit.append(
                        {
                            "sentence": sentence,
                            "scores_cen": scores,
                            "max_entailment": max(s[1] for s in scores),
                        }
                    )
            rows.append(
                {
                    "claim_id": claim_id,
                    "claim": claim,
                    "shown_claim": shown_claim,
                    "evidence": evidence,
                    "context_source_ids": [d["id"] for d in evidence],
                    "source_id_origin": "dataset-supplied context provenance, not verified citations",
                    "verdict_prompt": vp,
                    "explanation_prompt": ep,
                    "verdict_prompt_tokens": vtokens,
                    "explanation_prompt_tokens": etokens,
                    "generated_verdict": verdict,
                    "generated_explanation": explanation,
                    "numeric_provenance": provenance,
                    "reasons": reasons,
                    "raw_candidate_label": verdict if verdict in LABELS else None,
                    "structural_candidate_label": verdict if not reasons else None,
                    "explanation_nli_audit": sentence_audit,
                    "true_label": claims[claim_id]["frozen_label"],
                }
            )
            if position % 10 == 0:
                print(f"Completed {position + 1}/{len(cohort)}", flush=True)
    write_json_atomic(output / "predictions.json", rows)
    write_json_atomic(
        output / "generation_summary.json",
        {
            "rows": len(rows),
            "raw_verdict_metrics": summarize(rows, "raw_candidate_label"),
            "structural_guard_metrics": summarize(rows, "structural_candidate_label"),
            "sentences_audited": sum(len(r["explanation_nli_audit"]) for r in rows),
            "scope": (
                "post-hoc supplied-evidence diagnostic; "
                "grounding remains a machine diagnostic"
            ),
            "no_target_fitting": True,
            "source_authentication_verified": False,
            "semantic_grounding_verified": False,
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.glob("*.json") if p.name != "output_manifest.json"},
    )
    print(output)


if __name__ == "__main__":
    main()
