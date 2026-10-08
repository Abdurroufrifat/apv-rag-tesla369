"""Four synthetic cases for a distinct NLI-gated hybrid generator."""

import json
from importlib.metadata import version
from pathlib import Path

from smoke_test_generative_interface import CASES

from apv_rag.grounded_generation import evidence_decision, explanation_passes
from apv_rag.multilingual_nli import cen_order
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/grounded_generation_smoke_v3"
    if output.exists():
        raise FileExistsError("Smoke run exists; refusing overwrite")
    nli_path = root / "models/nli-deberta-v3-small"
    generator_path = root / "models/flan-t5-base"
    original = root / "artifacts/retrieval_nli_comparison"
    old_hashes = json.loads((original / "output_manifest.json").read_text())
    assert sha256(original / "input_manifest.json") == old_hashes["input_manifest.json"]
    nli_meta = json.loads((original / "input_manifest.json").read_text())
    gen_meta = json.loads((root / "models/generative_model_manifest.json").read_text())
    if _model_files(nli_path) != nli_meta["models"]["nli"]:
        raise ValueError("Original NLI model checksum mismatch")
    if _model_files(generator_path) != gen_meta["files"]:
        raise ValueError("Generator checksum mismatch")
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    nt = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
    nli = AutoModelForSequenceClassification.from_pretrained(
        nli_path, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    order = cen_order(nli.config.id2label)
    gt = AutoTokenizer.from_pretrained(generator_path, local_files_only=True)
    generator = AutoModelForSeq2SeqLM.from_pretrained(
        generator_path, local_files_only=True, torch_dtype=torch.float32
    ).eval()

    def score(premise, hypothesis):
        inputs = nt(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            values = nli(**inputs).logits.float().softmax(-1)[0].cpu().numpy()
        return values[order].astype(float).tolist()

    rows = []
    for case in CASES:
        scores = [score(d["text"], case["claim"]) for d in case["evidence"]]
        decision = evidence_decision(scores)
        prompt, explanation, explanation_scores = None, None, None
        accepted = False
        selected_source = None
        if decision["selected"] is not None:
            document = case["evidence"][decision["selected"]]
            selected_source = document["id"]
            prompt = f"Summarize the following evidence in one sentence: {document['text']}"
            inputs = gt(prompt, return_tensors="pt", truncation=False)
            if inputs.input_ids.shape[1] > 512:
                raise ValueError("Generator input budget exceeded")
            with torch.inference_mode():
                generated = generator.generate(
                    **inputs, do_sample=False, num_beams=1, max_new_tokens=96
                )
            explanation = gt.decode(generated[0], skip_special_tokens=True).strip()
            explanation_scores = score(document["text"], explanation)
            accepted = explanation_passes(explanation, explanation_scores)
        status = "machine_candidate" if accepted else "abstain"
        rows.append(
            {
                **case,
                "claim_nli_scores_cen": scores,
                "nli_decision": decision,
                "generation_prompt": prompt,
                "raw_generated_explanation": explanation,
                "explanation_nli_scores_cen": explanation_scores,
                "explanation_accepted": accepted,
                "status": status,
                "selected_source_id": selected_source,
                "citation_origin": "application-selected source, not model-generated citation",
                "verdict_correct": decision["label"] == case["expected_verdict"],
            }
        )
        print(case["id"], decision["label"], status, explanation, flush=True)
    output.mkdir()
    write_json_atomic(
        output / "input_manifest.json",
        {
            "model_files": {"nli": nli_meta["models"]["nli"], "generator": gen_meta},
            "packages": {n: version(n) for n in ("torch", "transformers")},
            "seed": 369,
            "threads": 4,
            "dtype": "float32",
            "code_sha256": {
                n: sha256(root / n)
                for n in (
                    "scripts/smoke_test_grounded_generation.py",
                    "src/apv_rag/grounded_generation.py",
                    "src/apv_rag/multilingual_nli.py",
                    "scripts/smoke_test_generative_interface.py",
                )
            },
            "protocol_sha256": sha256(root / "docs/GROUNDED_GENERATION_V3.md"),
        },
    )
    write_json_atomic(output / "smoke_results.json", rows)
    ready = all(
        r["verdict_correct"]
        and (
            r["explanation_accepted"]
            if r["expected_verdict"] != "Not Enough Evidence"
            else r["status"] == "abstain" and r["raw_generated_explanation"] is None
        )
        for r in rows
    )
    write_json_atomic(
        output / "smoke_summary.json",
        {
            "scope": "synthetic hybrid-pipeline sanity test; not benchmark results",
            "correct_nli_verdicts": sum(r["verdict_correct"] for r in rows),
            "accepted_generated_explanations": sum(r["explanation_accepted"] for r in rows),
            "full_run_ready": ready,
            "verdict_origin": "NLI, not generator",
            "qualification": "NLI checks are machine heuristics, not gold semantic verification",
        },
    )
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.glob("*.json") if p.name != "output_manifest.json"},
    )


if __name__ == "__main__":
    main()
