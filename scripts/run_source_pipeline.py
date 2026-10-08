"""Run a source-aware machine-candidate pipeline on a local document collection."""

import argparse
import json
import sqlite3
from pathlib import Path

from apv_rag.nli_comparison import _fingerprint, _model_files, _pairs
from apv_rag.source_pipeline import PipelineConfig, verify_claim
from apv_rag.splits import sha256, write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-primary", action="store_true")
    parser.add_argument("--no-family-collapse", action="store_true")
    parser.add_argument("--no-source-weights", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing output")
    root = Path(__file__).resolve().parents[1]
    data = json.loads(args.input.read_text(encoding="utf-8"))
    model_path = root / "models/nli-deberta-v3-small"
    model_hash = _fingerprint(_model_files(model_path))
    config = PipelineConfig(
        require_primary=args.require_primary,
        collapse_families=not args.no_family_collapse,
        weight_sources=not args.no_source_weights,
    )
    # Load local model only when compatible evidence actually requires scoring.
    state = {}

    def scorer(claim, texts):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not state:
            torch.set_num_threads(4)
            torch.manual_seed(369)
            torch.use_deterministic_algorithms(True)
            state["tokenizer"] = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            state["model"] = AutoModelForSequenceClassification.from_pretrained(
                model_path, local_files_only=True
            ).eval()
            if {int(k): str(v).lower() for k, v in state["model"].config.id2label.items()} != {
                0: "contradiction",
                1: "entailment",
                2: "neutral",
            }:
                raise ValueError("Unexpected NLI mapping")
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE pairs (key TEXT PRIMARY KEY, scores TEXT)")
            return _pairs(
                connection, state["model"], state["tokenizer"], torch, texts, claim, model_hash, 8
            )

    result = verify_claim(data["claim"], data["claim_language"], data["documents"], scorer, config)
    result["input_sha256"] = sha256(args.input)
    result["model_fingerprint"] = model_hash
    result["code_sha256"] = {
        name: sha256(root / name)
        for name in (
            "scripts/run_source_pipeline.py",
            "src/apv_rag/source_pipeline.py",
            "src/apv_rag/retrieval.py",
            "src/apv_rag/direct_nli.py",
            "src/apv_rag/nli_comparison.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
