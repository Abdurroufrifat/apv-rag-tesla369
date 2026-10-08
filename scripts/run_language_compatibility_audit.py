"""Controlled metadata-gate audit; does not translate or score foreign-language text."""

import copy
import json
from pathlib import Path

from apv_rag.source_pipeline import verify_claim
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = root / "artifacts/source_pipeline_received"
    path = inputs / "predictions.json"
    manifest = json.loads((inputs / "output_manifest.json").read_text())
    if sha256(path) != manifest["predictions.json"]:
        raise ValueError("Received prediction checksum mismatch")
    identity = json.loads((inputs / "input_manifest.json").read_text())
    source = root / "data/external/scifact/sealed_v1"
    for name in ("corpus.jsonl", "claims_dev.jsonl"):
        if sha256(source / name) != identity["source_inputs"][name]:
            raise ValueError("Target source checksum mismatch")
    corpus = {
        r["doc_id"]: r
        for r in [json.loads(line) for line in (source / "corpus.jsonl").read_text().splitlines()]
    }
    claims = {
        r["id"]: r
        for r in [
            json.loads(line) for line in (source / "claims_dev.jsonl").read_text().splitlines()
        ]
    }
    baseline = [r for r in json.loads(path.read_text()) if r["variant"] == "full_heuristic"]
    if len({r["claim_id"] for r in baseline}) != len(baseline):
        raise ValueError("Duplicate baseline claims")
    rows, summaries = [], []
    for scenario in (
        "matching_tags",
        "all_evidence_tags_mismatch",
        "claim_tag_mismatch",
        "mixed_evidence_tags",
        "no_evidence",
    ):
        counts = {
            "claims": len(baseline),
            "candidates": 0,
            "abstentions": 0,
            "scorer_calls": 0,
            "incompatible_selected": 0,
            "language_rejections": 0,
        }
        for record in baseline:
            cid = record["claim_id"]
            claim_language = "en"
            docs = [
                {
                    "id": e["id"],
                    "text": " ".join(corpus[e["id"]]["abstract"]),
                    "language": "en",
                    "source_rank": 5,
                    "source_url": "https://github.com/allenai/scifact",
                }
                for e in record["selected_evidence"]
            ]
            lookup = {
                d["text"]: e["nli_probabilities"]
                for d, e in zip(docs, record["selected_evidence"], strict=True)
            }
            docs = copy.deepcopy(docs)
            if scenario == "all_evidence_tags_mismatch":
                for d in docs:
                    d["language"] = "sr"
            elif scenario == "claim_tag_mismatch":
                claim_language = "sr"
            elif scenario == "mixed_evidence_tags":
                for d in docs[::2]:
                    d["language"] = "sr"
            elif scenario == "no_evidence":
                docs = []
            calls = []

            def scorer(claim, texts, current_calls=calls, current_lookup=lookup):
                current_calls.append(len(texts))
                return [current_lookup[text] for text in texts]

            result = verify_claim(claims[cid]["claim"], claim_language, docs, scorer)
            tags = {d["id"]: d["language"] for d in docs}
            incompatible = sum(tags[e["id"]] != claim_language for e in result["selected_evidence"])
            if incompatible:
                raise ValueError("Incompatible evidence reached scoring")
            if scenario in ("all_evidence_tags_mismatch", "claim_tag_mismatch", "no_evidence"):
                if calls or result["status"] != "abstain":
                    raise ValueError("Unavailable compatible evidence must abstain without scoring")
            if scenario == "matching_tags":
                if (
                    result["status"] != record["status"]
                    or result["candidate_label"] != record["candidate_label"]
                ):
                    raise ValueError("Matching-tag local replay differs from baseline decision")
            counts["candidates"] += result["status"] == "machine_candidate"
            counts["abstentions"] += result["status"] == "abstain"
            counts["scorer_calls"] += len(calls)
            counts["incompatible_selected"] += incompatible
            counts["language_rejections"] += sum(
                e["reason"] == "declared_language_mismatch" for e in result["rejected_evidence"]
            )
            rows.append(
                {
                    "scenario": scenario,
                    "claim_id": cid,
                    "claim_language_declared": claim_language,
                    "scorer_calls": len(calls),
                    **result,
                }
            )
        summaries.append({"scenario": scenario, **counts})
    output = root / "artifacts/language_compatibility_audit"
    if output.exists():
        raise FileExistsError("Completed audit exists; refusing overwrite")
    output.mkdir()
    write_json_atomic(
        output / "audit_summary.json",
        {
            "scope": "controlled language-tag and evidence-absence software audit",
            "results": summaries,
            "limitations": [
                "English texts remain unchanged; sr tags are synthetic mismatch controls",
                "local replay uses earlier retrieved abstracts, not a new full-corpus run",
                "not multilingual NLI accuracy or XFEVER evaluation",
                "no automatic language detection; trusts supplied metadata",
            ],
        },
    )
    write_json_atomic(output / "predictions.json", rows)
    write_json_atomic(
        output / "input_manifest.json",
        {
            "received_predictions_sha256": sha256(path),
            "target_files": {
                name: sha256(source / name) for name in ("claims_dev.jsonl", "corpus.jsonl")
            },
            "code": {
                name: sha256(root / name)
                for name in (
                    "scripts/run_language_compatibility_audit.py",
                    "src/apv_rag/source_pipeline.py",
                    "src/apv_rag/retrieval.py",
                    "src/apv_rag/direct_nli.py",
                )
            },
        },
    )
    lines = [
        "# Language compatibility audit",
        "",
        "Controlled metadata test on unchanged English evidence. Not multilingual accuracy.",
        "",
        "| Scenario | Claims | Candidates | Abstentions | Scorer calls | Incompatible selected |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        (
            f"| {r['scenario']} | {r['claims']} | {r['candidates']} | "
            f"{r['abstentions']} | {r['scorer_calls']} | {r['incompatible_selected']} |"
        )
        for r in summaries
    ]
    lines += [
        "",
        (
            "All mismatch-only and empty-evidence cases abstained without NLI scoring. "
            "The mixed-tag control selected only matching-tag evidence. These are "
            "metadata-gate properties, not evidence of understanding Serbian or of "
            "detecting incorrect tags. XFEVER and genuine multilingual evaluation remain "
            "unfinished."
        ),
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json_atomic(
        output / "output_manifest.json",
        {p.name: sha256(p) for p in output.iterdir() if p.name != "output_manifest.json"},
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
