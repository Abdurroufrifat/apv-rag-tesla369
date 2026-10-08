"""Isolate chat formatting, instruction following, stance and explanation failures."""

import json
from importlib.metadata import version
from pathlib import Path

from download_instruction_model import MODEL_ID, REVISION

from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256, write_json_atomic

PROBES = [
    ("copy", "Reply with exactly this word and nothing else: READY", "READY"),
    (
        "format",
        "Reply with exactly this text: Supported | Evidence states 20 degrees [1].",
        "Supported | Evidence states 20 degrees [1].",
    ),
    (
        "support_label",
        "Evidence: The water temperature was 20 degrees Celsius.\n"
        "Claim: The water temperature was 20 degrees Celsius.\n"
        "Reply only Supported, Refuted, or Not Enough Evidence.",
        "Supported",
    ),
    (
        "refute_label",
        "Evidence: The water temperature was 10 degrees Celsius, not 20.\n"
        "Claim: The water temperature was 20 degrees Celsius.\n"
        "Reply only Supported, Refuted, or Not Enough Evidence.",
        "Refuted",
    ),
    (
        "unrelated_label",
        "Evidence: The sample was collected on Tuesday; temperature was not measured.\n"
        "Claim: The water temperature was 20 degrees Celsius.\n"
        "Reply only Supported, Refuted, or Not Enough Evidence.",
        "Not Enough Evidence",
    ),
    (
        "explanation",
        "Evidence [7]: The water temperature was 20 degrees Celsius.\n"
        "Explain in one sentence what this evidence reports. Include [7].",
        None,
    ),
]


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/instruction_interface_diagnostics"
    if output.exists():
        raise FileExistsError("Diagnostics exist; refusing overwrite")
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
    results = []
    for name, question, expected in PROBES:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        legacy = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        rendered_inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        direct = inputs["input_ids"][0].tolist()
        rendered_ids = rendered_inputs.input_ids[0].tolist()
        matches = direct == rendered_ids
        print(
            name,
            "legacy return type:",
            type(legacy).__name__,
            "token sequences equal:",
            matches,
            flush=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = generated[0, inputs.input_ids.shape[1] :].tolist()
        answer = tokenizer.decode(completion, skip_special_tokens=True).strip()
        results.append(
            {
                "probe": name,
                "question": question,
                "rendered_prompt": rendered,
                "input_token_ids": direct,
                "rendered_input_token_ids": rendered_ids,
                "token_sequences_equal": matches,
                "legacy_return_type": type(legacy).__name__,
                "tokenization_path": "direct chat template, return_dict=True",
                "completion_token_ids": completion,
                "raw_answer": answer,
                "expected_exact": expected,
                "exact_match": answer == expected if expected is not None else None,
                "hit_generation_limit": len(completion) == 128,
            }
        )
        print(name, repr(answer), flush=True)
    output.mkdir()
    write_json_atomic(
        output / "diagnostics.json",
        {
            "scope": "component diagnostics only; no publication metrics or readiness claim",
            "model": meta,
            "packages": {n: version(n) for n in ("torch", "transformers")},
            "model_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "model_config": model.config.to_dict(),
            "generation_config": model.generation_config.to_dict(),
            "chat_template": tokenizer.chat_template,
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "settings": {"seed": 369, "threads": 4, "do_sample": False, "max_new_tokens": 128},
            "code_sha256": sha256(Path(__file__)),
            "results": results,
        },
    )
    write_json_atomic(
        output / "output_manifest.json", {"diagnostics.json": sha256(output / "diagnostics.json")}
    )


if __name__ == "__main__":
    main()
