"""Replay same-claim foreign-pool substitutions without neural inference."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apv_rag.multilingual_source_guard import (  # noqa: E402
    admit_context, build_source_index, execute_language_bound,
)
from apv_rag.splits import sha256, write_json_atomic  # noqa: E402

SOURCE = Path("data/processed/xfever/confirmation_v1")
CLEAN = Path("artifacts/clean_multilingual_confirmation_received_v1")
OUTPUT = Path("artifacts/multilingual_pool_mismatch_v1")
FILES = tuple(f"{lang}/test.jsonl" for lang in ("en", "es", "fr", "id", "ja", "zh"))
INPUTS = (SOURCE / "model_inputs.json", SOURCE / "corpora.json",
          SOURCE / "output_manifest.json", CLEAN / "audit_manifest.json",
          CLEAN / "inference/prepared_contexts.json")
CODE = (Path("src/apv_rag/multilingual_source_guard.py"),
        Path("scripts/audit_multilingual_pool_mismatch.py"))


def load():
    expected = json.loads((ROOT / SOURCE / "output_manifest.json").read_text(encoding="utf-8"))
    for name in ("model_inputs.json", "corpora.json"):
        if sha256(ROOT / SOURCE / name) != expected[name]:
            raise ValueError(f"Pinned confirmation source mismatch: {name}")
    received = json.loads((ROOT / CLEAN / "audit_manifest.json").read_text(encoding="utf-8"))
    path = ROOT / CLEAN / "inference/prepared_contexts.json"
    if sha256(path) != received["files"]["inference/prepared_contexts.json"]:
        raise ValueError("Clean prepared-context identity mismatch")
    rows = json.loads((ROOT / SOURCE / "model_inputs.json").read_text(encoding="utf-8"))
    prepared = json.loads(path.read_text(encoding="utf-8"))
    corpora = json.loads((ROOT / SOURCE / "corpora.json").read_text(encoding="utf-8"))
    if (len(rows) != len(prepared) or set(prepared) != {row["key"] for row in rows}
            or set(corpora) != set(FILES)):
        raise ValueError("Confirmation cohort differs")
    return rows, prepared, build_source_index(corpora)


def analyze(rows, prepared, index):
    by_claim = {(row["claim_id"], row["file"]): row for row in rows}
    if len(rows) != 600 or len(by_claim) != 600:
        raise ValueError("Expected 100 parallel claims in each of six files")
    records = []
    for row in rows:
        current = prepared[row["key"]]
        admitted = admit_context(row, current, index, indexed=True)
        if not admitted:
            raise ValueError(f"Original context unexpectedly unbound: {row['key']}")
        donor_file = FILES[(FILES.index(row["file"]) + 1) % len(FILES)]
        donor = by_claim[(row["claim_id"], donor_file)]
        swapped = copy.deepcopy(current)
        foreign = prepared[donor["key"]]
        swapped["retrieved_evidence"] = copy.deepcopy(foreign["retrieved_evidence"])
        swapped["evidence"] = copy.deepcopy(foreign["evidence"])
        accepted = admit_context(row, swapped, index, indexed=True)
        if not accepted:
            def no_request(*_):
                raise AssertionError("Mismatched evidence reached generation")
            policies = execute_language_bound(row, swapped, index, None, None,
                                              {}, no_request, indexed=True)
            if any(result["candidate_label"] is not None or result["generation_requests"]
                   or result["reasons"] != ["source_language_pool_mismatch"]
                   for result in policies):
                raise ValueError("Mismatched evidence was not withheld")
        records.append({"key": row["key"], "claim_id": row["claim_id"],
                        "file": row["file"], "donor_file": donor_file,
                        "original_bound": admitted, "swapped_bound": accepted,
                        "original_passages": len(current["retrieved_evidence"]),
                        "swapped_passages": len(swapped["retrieved_evidence"])})
    counts = Counter(record["file"] for record in records if not record["swapped_bound"])
    return records, {"scope": "closed excerpt-pool identity on 100 parallel claim IDs",
        "original_contexts_bound": len(records), "parallel_swap_contexts": len(records),
        "parallel_swaps_rejected": sum(not row["swapped_bound"] for row in records),
        "parallel_swaps_admitted": sum(row["swapped_bound"] for row in records),
        "rejected_by_target_file": {name: counts[name] for name in FILES},
        "model_calls_on_rejected_swaps": 0,
        "limitations": ["This guard checks membership in a pinned language-specific pool, not detected language or relevance.",
                        "Shared identical excerpts across pools may legitimately pass.",
                        "The pools contain benchmark-selected excerpts and are not open-web retrieval.",
                        "No new verdict, historical authentication, human label or semantic explanation check was produced."]}


def run():
    directory = ROOT / OUTPUT
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {directory}")
    rows, prepared, index = load()
    records, summary = analyze(rows, prepared, index)
    directory.mkdir(parents=True)
    path = directory / "admissions.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    write_json_atomic(directory / "summary.json", summary)
    write_json_atomic(directory / "receipt.json", {
        "inputs": {p.as_posix(): sha256(ROOT / p) for p in INPUTS},
        "code": {p.as_posix(): sha256(ROOT / p) for p in CODE},
        "outputs": {p.name: sha256(p) for p in (path, directory / "summary.json")},
    })
    return summary


def verify():
    directory = ROOT / OUTPUT
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    for group in ("inputs", "code", "outputs"):
        for name, expected in receipt[group].items():
            path = directory / name if group == "outputs" else ROOT / name
            if sha256(path) != expected:
                raise ValueError(f"{group} SHA-256 mismatch: {name}")
    rows, prepared, index = load()
    records, summary = analyze(rows, prepared, index)
    saved = [json.loads(line) for line in (directory / "admissions.jsonl").read_text(encoding="utf-8").splitlines()]
    if records != saved or summary != json.loads((directory / "summary.json").read_text(encoding="utf-8")):
        raise ValueError("Saved admission replay differs")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else run()
    print(f"Language-pool audit passed: {result['parallel_swaps_rejected']}/"
          f"{result['parallel_swap_contexts']} same-claim foreign-pool contexts withheld")
