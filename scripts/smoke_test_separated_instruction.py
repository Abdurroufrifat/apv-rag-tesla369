"""Separate four-case smoke evaluation of the pinned causal instruction model."""

import json
from importlib.metadata import version
from pathlib import Path

from download_instruction_model import MODEL_ID, REVISION
from smoke_test_generative_interface import CASES

from apv_rag.generation_integrity import new_numeric_values
from apv_rag.generative_rag import LABELS
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/separated_instruction_smoke_v5"
    if output.exists():
        raise FileExistsError("Smoke run exists; refusing overwrite")
    path = root / "models/qwen2.5-1.5b-instruct"
    meta = json.loads((root / "models/instruction_model_manifest.json").read_text())
    if meta["model_id"] != MODEL_ID or meta["revision"] != REVISION:
        raise ValueError("Pinned model required")
    if _model_files(path) != meta["files"]:
        raise ValueError("Model checksum mismatch")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float32
    ).eval()

    def generate(question):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if tokens.input_ids.shape[1] > 1024:
            raise ValueError("Prompt budget exceeded")
        with torch.inference_mode():
            output_ids = model.generate(
                **tokens,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            output_ids[0, tokens.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return rendered, answer

    rows = []
    for case in CASES:
        verdict_prompt, explanation_prompt, verdict, explanation = None, None, None, None
        reasons, numbers = [], []
        if case["evidence"]:
            if len(case["evidence"]) != 1:
                raise ValueError("Smoke interface is restricted to one supplied passage")
            text = case["evidence"][0]["text"]
            verdict_prompt, verdict = generate(
                f"Evidence: {text}\nClaim: {case['claim']}\n"
                "Reply only Supported, Refuted, or Not Enough Evidence."
            )
            if verdict not in LABELS:
                reasons.append("invalid_verdict")
            else:
                explanation_prompt, explanation = generate(
                    f"Evidence: {text}\nClaim: {case['claim']}\nVerdict: {verdict}\n"
                    "Explain in one sentence why this evidence supports, contradicts, "
                    "or cannot establish the claim. Use only the evidence."
                )
                numbers = new_numeric_values(explanation, case["evidence"])
                if not explanation:
                    reasons.append("empty_explanation")
                if numbers:
                    reasons.append("novel_numeric_value")
        else:
            reasons.append("no_evidence")
        accepted = not reasons
        correct = (accepted and verdict == case["expected_verdict"]) if case["evidence"] else True
        rows.append(
            {
                **case,
                "verdict_prompt": verdict_prompt,
                "explanation_prompt": explanation_prompt,
                "generated_verdict": verdict,
                "generated_explanation": explanation,
                "novel_numeric_values": numbers,
                "status": "machine_candidate" if accepted else "abstain",
                "reasons": reasons,
                "expected_action_passed": correct,
                "context_source_ids": [d["id"] for d in case["evidence"]],
                "source_id_origin": "application context provenance; not generated citations",
                "semantic_grounding_verified": False,
            }
        )
        print(case["id"], verdict, explanation, reasons, flush=True)
    output.mkdir()
    write_json_atomic(
        output / "input_manifest.json",
        {
            "model": meta,
            "seed": 369,
            "threads": 4,
            "dtype": "float32",
            "packages": {n: version(n) for n in ("torch", "transformers")},
            "code_sha256": {
                n: sha256(root / n)
                for n in (
                    "scripts/smoke_test_separated_instruction.py",
                    "scripts/download_instruction_model.py",
                    "src/apv_rag/generation_integrity.py",
                    "src/apv_rag/generative_rag.py",
                    "scripts/smoke_test_generative_interface.py",
                )
            },
            "protocol_sha256": sha256(root / "docs/SEPARATED_INSTRUCTION_V5.md"),
        },
    )
    write_json_atomic(output / "smoke_results.json", rows)
    write_json_atomic(
        output / "smoke_summary.json",
        {
            "scope": "post-failure synthetic development test, not benchmark evidence",
            "cases": len(rows),
            "passed_expected_actions": sum(r["expected_action_passed"] for r in rows),
            "full_run_ready": all(r["expected_action_passed"] for r in rows),
            "qualification": "structural and numeric checks do not establish semantic grounding",
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.glob("*.json") if p.name != "output_manifest.json"},
    )


if __name__ == "__main__":
    main()
