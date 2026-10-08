"""Fixed English-model transfer on supplied XFEVER multilingual evidence pairs."""

import json
import sqlite3
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from apv_rag.direct_nli import TARGET_LABELS, direct_probabilities
from apv_rag.metrics import multiclass_brier
from apv_rag.nli_comparison import _fingerprint, _model_files, _pairs
from apv_rag.splits import sha256, write_json_atomic
from apv_rag.xfever import LABEL_MAP, validate_parallel


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = root / "data/external/xfever/zenodo_8206962/evaluation_inputs_v1"
    output = root / "artifacts/xfever_nli_stress"
    if (output / "output_manifest.json").exists():
        raise FileExistsError("Completed evaluation exists; refusing overwrite")
    manifest = json.loads((inputs / "manifest.json").read_text())
    expected_archive = "8b7948894c8724d9a52e86e18fe0e369c5d58b12ddba853e836b8d80611a4895"
    if manifest["archive_sha256"] != expected_archive:
        raise ValueError("Unexpected source archive")
    expected = ["en/test.6h.jsonl"] + [
        f"{lang}/test.6h{suffix}.jsonl"
        for lang in ("es", "fr", "id", "ja", "zh")
        for suffix in ("", ".human")
    ]
    files = {}
    for name in expected:
        path = inputs / name
        files[name] = sha256(path)
        if files[name] != manifest["files"]["data/" + name]["sha256"]:
            raise ValueError(f"Target checksum mismatch: {name}")
    original = root / "artifacts/retrieval_nli_comparison"
    original_manifest = json.loads((original / "output_manifest.json").read_text())
    meta_path = original / "input_manifest.json"
    if sha256(meta_path) != original_manifest[meta_path.name]:
        raise ValueError("Original inference metadata differs")
    meta = json.loads(meta_path.read_text())
    for package in ("torch", "transformers", "numpy", "scikit-learn"):
        if version(package) != meta["packages"][package]:
            raise ValueError(f"Original package version required: {package}")
    model_path = root / "models/nli-deberta-v3-small"
    if _model_files(model_path) != meta["models"]["nli"]:
        raise ValueError("Original NLI model required")
    identity = {
        "files": files,
        "original_metadata_sha256": sha256(meta_path),
        "model": meta["models"]["nli"],
        "packages": meta["packages"],
        "rule": "single supplied evidence premise; direct C/E/N argmax; no target fitting",
        "max_pair_tokens": 256,
        "threads": 4,
        "seed": 369,
        "code_sha256": {
            n: sha256(root / n)
            for n in (
                "scripts/run_xfever_nli_stress.py",
                "src/apv_rag/xfever.py",
                "src/apv_rag/direct_nli.py",
                "src/apv_rag/nli_comparison.py",
            )
        },
        "protocol_sha256": sha256(root / "docs/XFEVER_NLI_STRESS.md"),
    }
    output.mkdir(exist_ok=True)
    receipt = output / "input_manifest.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Inputs changed; refusing cached run reuse")
    write_json_atomic(receipt, identity)
    sets = {
        name: [
            json.loads(line) for line in (inputs / name).read_text(encoding="utf-8").splitlines()
        ]
        for name in expected
    }
    count = validate_parallel(sets)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).eval()
    if {int(k): str(v).lower() for k, v in model.config.id2label.items()} != {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }:
        raise ValueError("Unexpected NLI mapping")
    predictions, results, english = [], [], None
    with sqlite3.connect(output / "pair_cache.sqlite") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, scores TEXT)")
        for name, rows in sets.items():
            p_rows = []
            for position, row in enumerate(rows):
                scores = _pairs(
                    connection,
                    model,
                    tokenizer,
                    torch,
                    [row["evidence"]],
                    row["claim"],
                    _fingerprint(meta["models"]["nli"]),
                    8,
                )
                p_rows.append(direct_probabilities(scores))
                if position % 50 == 0:
                    print(f"{name}: {position + 1}/{count}", flush=True)
            p = np.asarray(p_rows)
            pred = np.asarray(TARGET_LABELS)[p.argmax(1)]
            truth = [LABEL_MAP[r["label"]] for r in rows]
            if english is None:
                english = pred.copy()
            results.append(
                {
                    "file": name,
                    "rows": count,
                    "macro_f1": float(
                        f1_score(
                            truth,
                            pred,
                            labels=list(TARGET_LABELS),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "accuracy": float(accuracy_score(truth, pred)),
                    "brier": multiclass_brier(truth, p, TARGET_LABELS),
                    "prediction_disagreement_with_english": float(np.mean(pred != english)),
                    "per_class": classification_report(
                        truth, pred, labels=list(TARGET_LABELS), output_dict=True, zero_division=0
                    ),
                }
            )
            for position, row in enumerate(rows):
                predictions.append(
                    {
                        "file": name,
                        "row": position,
                        "claim_id": row["id"],
                        "english_page": sets["en/test.6h.jsonl"][position]["page"],
                        "true_label": truth[position],
                        "predicted_label": str(pred[position]),
                        "probabilities": p[position].tolist(),
                    }
                )
    write_json_atomic(
        output / "stress_summary.json",
        {
            "scope": (
                "English-model multilingual supplied-evidence transfer stress; not full "
                "retrieval or multilingual model comparison"
            ),
            "rows_per_file": count,
            "unique_claim_ids": len({r["id"] for r in sets["en/test.6h.jsonl"]}),
            "results": results,
            "no_target_fitting": True,
            "translation_origin": (
                "test.6h.human is upstream human translation; other target files are "
                "machine translation"
            ),
            "statistics_status": "grouped paired analysis pending prediction verification",
        },
    )
    write_json_atomic(output / "predictions.json", predictions)
    write_json_atomic(
        output / "output_manifest.json",
        {
            p.name: sha256(p)
            for p in output.iterdir()
            if p.suffix == ".json" and p.name != "output_manifest.json"
        },
    )
    print(output / "stress_summary.json")


if __name__ == "__main__":
    main()
