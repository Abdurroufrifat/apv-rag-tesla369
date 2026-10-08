"""Verify multilingual generation and run grouped exploratory comparisons."""
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np

from analyze_xfever_multilingual import verify as verify_nli
from apv_rag.generative_rag import LABELS
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2
from apv_rag.paired_statistics import holm_adjust
from apv_rag.splits import sha256, write_json_atomic
from apv_rag.xfever import LABEL_MAP
from apv_rag.multilingual_generation import explanation_question, verdict_question
from run_gated_generation import qwen_prompt
from run_sentence_rag_scifact import summarize
from run_xfever_generation import FILES, prepare_inputs
from verify_integrated_gate import check_hashes, load, require


def compare(actual, expected):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), "Summary schema mismatch")
        for name, value in expected.items():
            compare(actual[name], value)
    elif type(expected) is float:
        require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0), "Metric replay mismatch")
    else:
        require(actual == expected, "Summary/count mismatch")


def matrix_score(matrix):
    matrix = np.asarray(matrix).reshape(-1, 3, 3)
    diagonal = np.diagonal(matrix, axis1=1, axis2=2)
    denominator = matrix.sum(1) + matrix.sum(2)
    return np.divide(2 * diagonal, denominator, out=np.zeros(diagonal.shape), where=denominator != 0).mean(1)


def group_matrices(truth, predictions, membership, count):
    result = np.zeros((count, 9), dtype=np.int64)
    np.add.at(result, (membership, truth * 3 + predictions), 1)
    return result


def paired_result(truth, left, right, membership, groups, bootstrap, swaps):
    # Difference is right minus left; whole connected groups are resampled/swapped.
    a = group_matrices(truth, left, membership, groups)
    b = group_matrices(truth, right, membership, groups)
    observed = float(matrix_score(b.sum(0))[0] - matrix_score(a.sum(0))[0])
    effects = matrix_score(bootstrap @ b) - matrix_score(bootstrap @ a)
    exchanged = swaps @ (b - a)
    null = matrix_score(a.sum(0) + exchanged) - matrix_score(b.sum(0) - exchanged)
    return {"macro_f1_difference": observed,
            "marginal_group_bootstrap_95_interval": np.quantile(effects, [.025, .975]).tolist(),
            "group_swap_two_sided_p": float((np.sum(abs(null) >= abs(observed) - 1e-12) + 1) / (len(null) + 1))}


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/xfever_generation_received"
    check_hashes(folder)
    identity = load(folder, "input_manifest.json")
    sets, hashes, ids, manifest_hash = prepare_inputs(root)
    require(identity["files"] == hashes and identity["dataset_manifest_sha256"] == manifest_hash, "Dataset identity mismatch")
    require(identity["explanation_claim_ids"] == ids, "Explanation selection mismatch")
    require(identity["settings"] == {"claim_tokens": 192, "evidence_tokens": 512, "max_input_tokens": 1024,
            "max_new_tokens": 128, "seed": 369, "threads": 4, "dtype": "float32", "do_sample": False,
            "num_beams": 1, "rows_per_file": 600, "explanation_unique_claims": 60,
            "gate": "none; multilingual completeness transfer unvalidated"}, "Settings mismatch")
    prior_generator = root / "artifacts/gated_generation_received"
    check_hashes(prior_generator)
    require(load(prior_generator, "input_manifest.json")["generator"] == identity["generator"], "Generator receipt mismatch")
    model_manifest = root / "models/instruction_model_manifest.json"
    model_manifest_bytes_verified = model_manifest.exists()
    if model_manifest_bytes_verified:
        require(sha256(model_manifest) == identity["model_manifest_sha256"], "Generator manifest checksum mismatch")
        require(json.loads(model_manifest.read_text(encoding="utf-8")) == identity["generator"], "Generator manifest mismatch")
    for name, digest in identity["code_sha256"].items():
        require(sha256(root / name) == digest, f"Code mismatch: {name}")
    require(sha256(root / "docs/XFEVER_GENERATION.md") == identity["protocol_sha256"], "Protocol mismatch")
    rows = load(folder, "predictions.json")
    lookup = {(r["file"], r["row"]): r for r in rows}
    expected = {(name, i) for name in FILES for i in range(600)}
    require(len(rows) == len(lookup) == 6600 and set(lookup) == expected, "Record cohort mismatch")
    selected = set(ids)
    shared = {}
    explanation_count = 0
    for name in FILES:
        language = name.split("/")[0]
        for i, original in enumerate(sets[name]):
            r = lookup[name, i]
            require(r["claim_id"] == original["id"] and r["english_page"] == sets[FILES[0]][i]["page"], "Source row mismatch")
            require(r["language"] == language and r["true_label"] == LABEL_MAP[original["label"]], "Language/label mismatch")
            origin = "original" if language == "en" else ("upstream_human" if ".human." in name else "upstream_machine")
            require(r["translation_origin"] == origin, "Translation-origin mismatch")
            for field, limit in (("claim", 192), ("evidence", 512)):
                original_count = r[f"original_{field}_tokens"]
                shown_count = r[f"shown_{field}_tokens"]
                require(type(original_count) is int and original_count > 0, "Invalid reported source token count")
                require(shown_count == min(original_count, limit), "Reported token clipping mismatch")
                require(r[field + "_clipped"] == (original_count > limit), "Clipping flag mismatch")
                # All received texts are unclipped; verify their exact source content.
                require(not r[field + "_clipped"] and r["shown_" + field] == original[field], "Unexpected clipped or altered source text")
            verdict = r["generated_verdict"]
            require(verdict in LABELS and r["raw_candidate_label"] == verdict, "Invalid verdict")
            prompt = qwen_prompt(verdict_question(r["shown_claim"], r["shown_evidence"]))
            require(prompt == r["verdict_prompt"], "Verdict prompt mismatch")
            selected_row = str(original["id"]) in selected
            require(r["explanation_selected"] == selected_row, "Explanation selection mismatch")
            reasons = []
            numeric = None
            if selected_row:
                explanation_count += 1
                require(isinstance(r["generated_explanation"], str), "Missing explanation result")
                prompt = qwen_prompt(explanation_question(r["shown_claim"], r["shown_evidence"], verdict, language))
                require(r["explanation_prompt"] == prompt, "Explanation prompt mismatch")
                numeric = numeric_provenance_v2(r["generated_explanation"] or "", r["shown_claim"],
                                               [{"id": "premise", "text": r["shown_evidence"]}])
                if not r["generated_explanation"]:
                    reasons.append("empty_explanation")
                if numeric["absent_from_inputs"]:
                    reasons.append("numeric_value_absent")
            else:
                require(r["generated_explanation"] is None and r["explanation_prompt"] is None
                        and r["explanation_prompt_tokens"] is None, "Unexpected explanation outside subset")
            require(r["numeric_provenance"] == numeric and r["reasons"] == reasons, "Numeric policy mismatch")
            require(r["explanation_guarded_label"] == (verdict if selected_row and not reasons else None), "Guard decision mismatch")
            for kind in ("verdict", "explanation") if selected_row else ("verdict",):
                tokens = r[kind + "_prompt_tokens"]
                require(type(tokens) is int and 0 < tokens <= 1024, "Reported prompt budget mismatch")
                key = kind + ":" + hashlib.sha256(r[kind + "_prompt"].encode("utf-8")).hexdigest()
                answer = (r["generated_" + kind], tokens)
                if key in shared:
                    require(shared[key] == answer, "Shared prompt responses inconsistent")
                shared[key] = answer
    saved = load(folder, "summary.json")
    require(saved["no_target_fitting"] is True and saved["explanation_language_verified"] is False
            and saved["explanation_correctness_verified"] is False, "Scope flag mismatch")
    require(saved["unique_live_answers_cached"] == len(shared), "Unique prompt receipt mismatch")
    summaries = []
    english = {i: lookup[FILES[0], i]["raw_candidate_label"] for i in range(600)}
    saved_by_name = {r["file"]: r for r in saved["files"]}
    require(len(saved["files"]) == 11 and set(saved_by_name) == set(FILES), "Summary cohort mismatch")
    arrays = {}
    mapping = {label: i for i, label in enumerate(LABELS)}
    truth = np.array([mapping[LABEL_MAP[r["label"]]] for r in sets[FILES[0]]])
    for name in FILES:
        subset = [lookup[name, i] for i in range(600)]
        explained = [r for r in subset if r["explanation_selected"]]
        report = {"file": name, "rows": 600, "verdict_metrics": summarize(subset, "raw_candidate_label"),
            "verdict_disagreement_with_english": sum(r["raw_candidate_label"] != english[r["row"]] for r in subset) / 600,
            "claim_clipped": 0, "evidence_clipped": 0, "explained_rows": len(explained),
            "explained_unique_claims": len({r["claim_id"] for r in explained}),
            "explanation_subset_raw_verdict_metrics": summarize(explained, "raw_candidate_label"),
            "explanation_subset_guarded_metrics": summarize(explained, "explanation_guarded_label"),
            "numeric_rejections": sum("numeric_value_absent" in r["reasons"] for r in explained)}
        compare(saved_by_name[name], report)
        summaries.append(report)
        arrays[name] = np.array([mapping[r["raw_candidate_label"]] for r in subset])
        require(math.isclose(float(matrix_score(np.bincount(truth * 3 + arrays[name], minlength=9))[0]),
                             report["verdict_metrics"]["macro_f1_all_claims_abstentions_as_errors"], abs_tol=1e-12), "Confusion score mismatch")
    y, baseline, positions, membership, groups, nli_identity = verify_nli(root, "artifacts/xfever_received", "docs/XFEVER_NLI_STRESS.md")
    y2, multilingual, _, _, groups2, multilingual_identity = verify_nli(root, "artifacts/xfever_multilingual_received", "docs/XFEVER_MULTILINGUAL_CONTROL.md")
    require(np.array_equal(truth, y) and np.array_equal(truth, y2) and groups == groups2, "NLI cohort alignment mismatch")
    require(nli_identity["files"] == multilingual_identity["files"] == hashes, "NLI source hash mismatch")
    rng = np.random.default_rng(369)
    bootstrap = rng.multinomial(groups, np.full(groups, 1 / groups), size=2000)
    swaps = rng.integers(0, 2, size=(10000, groups), dtype=np.int8)
    translation_results = []
    for name in FILES[1:]:
        translation_results.append({"file": name, **paired_result(truth, arrays[FILES[0]], arrays[name], membership, groups, bootstrap, swaps)})
    for r, adjusted in zip(translation_results, holm_adjust([r["group_swap_two_sided_p"] for r in translation_results]), strict=True):
        r["holm_p_ten_language_comparisons"] = float(adjusted)
    model_results = []
    for model_name, model in (("english_nli", baseline), ("multilingual_nli", multilingual)):
        for name in FILES:
            model_results.append({"baseline": model_name, "file": name,
                                  **paired_result(truth, model[name], arrays[name], membership, groups, bootstrap, swaps)})
    for r, adjusted in zip(model_results, holm_adjust([r["group_swap_two_sided_p"] for r in model_results]), strict=True):
        r["holm_p_twenty_two_model_comparisons"] = float(adjusted)
    out = root / "artifacts/xfever_generation_verification"
    out.mkdir(exist_ok=True)
    write_json_atomic(out / "verified_summary.json", summaries)
    write_json_atomic(out / "statistics.json", {"groups": groups, "group_rule": "connected English claim ID or English page",
        "seed": 369, "bootstrap_samples": 2000, "group_swap_samples": 10000,
        "translation_comparisons": translation_results, "exploratory_model_comparisons": model_results,
        "packages": {n: version(n) for n in ("numpy", "scikit-learn")}})
    write_json_atomic(out / "verification_checks.json", {"verdict_records": 6600, "explanation_records": explanation_count,
        "unique_claim_ids": len({r["id"] for r in sets[FILES[0]]}), "groups": groups, "unique_prompt_keys": len(shared),
        "source_texts_unchanged": True, "clipped_claims": 0, "clipped_evidence": 0,
        "neural_inference_rerun": False, "tokenizer_counts_recomputed": False, "sqlite_cache_inspected": False,
        "local_model_manifest_bytes_verified": model_manifest_bytes_verified})
    model_note = ("The local model-manifest byte hash also matches; model weights were not read by this verifier."
                  if model_manifest_bytes_verified else "Local model/manifest bytes are unavailable here; the new model-manifest byte hash was not independently checked.")
    lower_languages = sum(r["macro_f1_difference"] < 0 and r["holm_p_ten_language_comparisons"] < .05 for r in translation_results)
    multilingual_lower = sum(r["baseline"] == "multilingual_nli" and r["macro_f1_difference"] < 0
                             and r["holm_p_twenty_two_model_comparisons"] < .05 for r in model_results)
    lines = ["# Multilingual generation verification and grouped comparisons", "",
        f"Verified all 6600 verdict records and {explanation_count} explanation records: output/data/model-receipt/code/protocol identity, aligned IDs/labels, exact source texts, prompt construction, reported token budgets, fixed explanation subset, shared response consistency, numeric decisions, metrics and counts. No input text was clipped or altered. Generator identity declarations match the prior verified controller receipt. {model_note} The receipt's {len(shared)} cached answers match distinct prompt keys; model inference, tokenizer counts and SQLite bytes were not independently rerun/inspected.", "",
        f"English verdict accuracy is 71.0% and macro F1 is 0.6963. Japanese accuracy is 55.0% for machine translation and 62.0% for upstream human translation. After grouped swaps and Holm correction, {lower_languages}/10 translated sets have lower macro F1 than English; the two French differences are not confirmed. In the exploratory model comparisons, no Qwen gain over multilingual NLI is confirmed, while Qwen is lower in {multilingual_lower}/11 sets. These results support reporting language limitations rather than general multilingual superiority.", "",
        "| File | Accuracy | Macro F1 | Disagreement with English | Subset numeric rejects |", "|---|---:|---:|---:|---:|"]
    for r in summaries:
        m = r["verdict_metrics"]
        lines.append(f"| {r['file']} | {m['covered_accuracy']:.1%} | {m['macro_f1_all_claims_abstentions_as_errors']:.4f} | {r['verdict_disagreement_with_english']:.1%} | {r['numeric_rejections']}/{r['explained_rows']} |")
    lines += ["", f"Paired analysis uses {groups} connected claim/page groups, 2000 group bootstrap samples and 10000 group swaps, seed 369. Language comparisons are target minus English macro F1. Holm corrects the ten comparisons. Intervals are marginal, not simultaneous. Groups limit duplicate dependence, but do not remove pretraining overlap or prior benchmark exposure.", "",
        "| Translation file | Macro F1 difference from English | Marginal 95% interval | Holm p |", "|---|---:|---|---:|"]
    for r in translation_results:
        lo, hi = r["marginal_group_bootstrap_95_interval"]
        lines.append(f"| {r['file']} | {r['macro_f1_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {r['holm_p_ten_language_comparisons']:.4f} |")
    lines += ["", "The saved English and multilingual NLI exports also passed score/decision/metric verification. Model comparisons below were added after observing generator outputs and are exploratory. Differences are Qwen minus the baseline, with Holm correction across all 22 comparisons. Models differ in training, size, decoding and token budgets. This does not isolate architecture or compute efficiency. No probabilities are available from the generator, so calibration/Brier claims are omitted.", "",
        "| File | NLI baseline | Qwen macro F1 difference | Marginal 95% interval | Holm p |", "|---|---|---:|---|---:|"]
    for r in model_results:
        lo, hi = r["marginal_group_bootstrap_95_interval"]
        lines.append(f"| {r['file']} | {r['baseline']} | {r['macro_f1_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {r['holm_p_twenty_two_model_comparisons']:.4f} |")
    lines += ["", "Numeric guards apply only to the preselected 60-ID explanation subset in each file. Rejection counts do not measure explanation truth or language compliance. Explanations are not independently verified. This is supplied-evidence generation, without retrieval, a multilingual completeness gate or source authentication. Upstream human translations require no new project annotation. Preserve every result; do not tune prompts or select languages from these observed scores. Full generation robustness and final reproducibility checks remain open.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json_atomic(out / "audit_manifest.json", {"source_output_manifest_sha256": sha256(folder / "output_manifest.json"),
        "verifier_sha256": sha256(Path(__file__)), "baseline_output_manifests": {n: sha256(root / "artifacts" / n / "output_manifest.json")
            for n in ("xfever_received", "xfever_multilingual_received")},
        "files": {p.name: sha256(p) for p in out.iterdir() if p.name != "audit_manifest.json"}})
    print(f"Verified 6600 verdicts and {explanation_count} explanations; {groups} groups.")
    print(json.dumps({"translation_comparisons": translation_results, "model_comparisons": model_results}, indent=2))


if __name__ == "__main__":
    main()
