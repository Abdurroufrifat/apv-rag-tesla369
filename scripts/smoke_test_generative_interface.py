"""Check the actual local generator on synthetic interface sanity cases."""

import json
from importlib.metadata import version
from pathlib import Path

from download_generative_model import MODEL_ID, REVISION

from apv_rag.generative_interface import generate_answer
from apv_rag.generative_rag import parse_answer
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256, write_json_atomic

CASES = [
    {
        "id": "support",
        "claim": "The measured water temperature was 20 degrees Celsius.",
        "evidence": [{"id": 1, "text": "The measured water temperature was 20 degrees Celsius."}],
        "expected_verdict": "Supported",
    },
    {
        "id": "refute",
        "claim": "The measured water temperature was 20 degrees Celsius.",
        "evidence": [
            {"id": 2, "text": "The measured water temperature was 10 degrees Celsius, not 20."}
        ],
        "expected_verdict": "Refuted",
    },
    {
        "id": "insufficient",
        "claim": "The measured water temperature was 20 degrees Celsius.",
        "evidence": [
            {
                "id": 3,
                "text": (
                    "The water sample was collected on Tuesday. Its temperature was not measured."
                ),
            }
        ],
        "expected_verdict": "Not Enough Evidence",
    },
    {
        "id": "empty",
        "claim": "The measured water temperature was 20 degrees Celsius.",
        "evidence": [],
        "expected_verdict": "Not Enough Evidence",
    },
]


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/generative_interface_smoke_v2"
    if output.exists():
        raise FileExistsError("Smoke run exists; refusing overwrite")
    model_path = root / "models/flan-t5-base"
    meta = json.loads((root / "models/generative_model_manifest.json").read_text())
    if meta["model_id"] != MODEL_ID or meta["revision"] != REVISION:
        raise ValueError("Pinned model required")
    if _model_files(model_path) != meta["files"]:
        raise ValueError("Model checksum mismatch")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    results = []
    for case in CASES:
        generated = generate_answer(model, tokenizer, torch, case["claim"], case["evidence"])
        parsed = parse_answer(generated["assembled_answer"], [d["id"] for d in case["evidence"]])
        results.append(
            {
                **case,
                **generated,
                "result": parsed,
                "verdict_correct": generated["generated_verdict"] == case["expected_verdict"],
            }
        )
        print(case["id"], generated["assembled_answer"], flush=True)
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
                    "scripts/smoke_test_generative_interface.py",
                    "src/apv_rag/generative_interface.py",
                    "src/apv_rag/generative_rag.py",
                )
            },
            "protocol_sha256": sha256(root / "docs/GENERATIVE_INTERFACE_V2.md"),
        },
    )
    write_json_atomic(output / "smoke_results.json", results)
    write_json_atomic(
        output / "smoke_summary.json",
        {
            "scope": "synthetic interface sanity checks, not research benchmark results",
            "cases": len(results),
            "correct_verdicts": sum(r["verdict_correct"] for r in results),
            "structurally_valid_answers": sum(
                r["result"]["status"] == "machine_candidate" for r in results
            ),
            "full_run_ready": all(
                r["verdict_correct"] and r["result"]["status"] == "machine_candidate"
                for r in results
            ),
            "semantic_explanation_support_verified": False,
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.glob("*.json") if p.name != "output_manifest.json"},
    )
    print(output)


if __name__ == "__main__":
    main()
