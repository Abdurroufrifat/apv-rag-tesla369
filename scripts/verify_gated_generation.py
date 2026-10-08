"""Replay live controller records and reuse matching explanation diagnostics."""
import contextlib
import hashlib
import io
import json
import math
import re
from pathlib import Path

import numpy as np

from apv_rag.gated_generation_flow import execute_generation
from apv_rag.splits import sha256, write_json_atomic
from run_gated_generation import qwen_prompt
from run_sentence_rag_scifact import summarize
from verify_integrated_gate import check_hashes, load, main as verify_gate, require


def explanation_diagnostics(rows, originals):
    result = {}
    for cohort in ("scifact", "climate_retrieved", "climate_supplied"):
        result[cohort] = {}
        for policy in ("no_gate", "nli", "embedding", "combined"):
            answered = [r for r in rows if r["cohort"] == cohort and r["policy"] == policy and r["candidate_label"] is not None]
            sentence_entailments = []
            sentence_contradictions = []
            all_above_07 = 0
            matched = 0
            for r in answered:
                original = originals[(cohort, str(r["claim_id"]))]
                if r["generated_explanation"] != original["generated_explanation"] or r["evidence"] != original["evidence"]:
                    continue
                audits = original["explanation_nli_audit"]
                sentences = re.split(r"(?<=[.!?])\s+", r["generated_explanation"])
                require([a["sentence"] for a in audits] == sentences, "Explanation audit text mismatch")
                maxima = []
                for a in audits:
                    scores = np.asarray(a["scores_cen"], dtype=float)
                    require(scores.shape == (len(r["evidence"]), 3), "Explanation NLI shape mismatch")
                    require(np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(), "Invalid explanation NLI scores")
                    require(np.allclose(scores.sum(1), 1, rtol=0, atol=1e-6), "Explanation NLI normalization mismatch")
                    maximum = float(scores[:, 1].max())
                    require(math.isclose(maximum, a["max_entailment"], abs_tol=1e-10, rel_tol=0), "Explanation NLI maximum mismatch")
                    maxima.append(maximum)
                    sentence_entailments.append(maximum)
                    sentence_contradictions.append(float(scores[:, 0].max()))
                matched += 1
                all_above_07 += int(bool(maxima) and all(v >= .7 for v in maxima))
            result[cohort][policy] = {
                "answered_claims": len(answered), "matching_cached_audits": matched,
                "sentences_audited": len(sentence_entailments),
                "sentence_max_entailment_at_least_07": sum(v >= .7 for v in sentence_entailments),
                "sentence_max_contradiction_at_least_07": sum(v >= .7 for v in sentence_contradictions),
                "explanations_all_sentence_max_entailment_at_least_07": all_above_07,
                "mean_sentence_max_entailment": float(np.mean(sentence_entailments)) if sentence_entailments else None,
                "scope": "reused NLI diagnostic on byte-matching explanation/context; not human grounding labels",
            }
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    with contextlib.redirect_stdout(io.StringIO()):
        verify_gate()
    folder = root / "artifacts/gated_generation_received"
    check_hashes(folder)
    identity = load(folder, "input_manifest.json")
    require(identity["backend"] == "live", "Expected live outputs")
    require(identity["policies"] == ["no_gate", "nli", "embedding", "combined"], "Expected all policies")
    require(identity["settings"] == {"threshold": .5, "seed": 369, "threads": 4, "dtype": "float32",
            "max_input_tokens": 1024, "max_new_tokens": 128, "do_sample": False, "num_beams": 1,
            "gate_features": "verified frozen-context cache"}, "Unexpected settings")
    for name, digest in identity["code_sha256"].items():
        require(sha256(root / name) == digest, f"Code mismatch: {name}")
    require(sha256(root / "docs/GATED_GENERATION.md") == identity["protocol_sha256"], "Protocol mismatch")
    gate_folder = root / "artifacts/integrated_gate_received"
    require(sha256(gate_folder / "output_manifest.json") == identity["replay_output_manifest_sha256"], "Feature/replay identity mismatch")
    gate_rows = {(r["cohort"], str(r["claim_id"])): r for r in load(gate_folder, "predictions.json")}
    originals = {}
    for cohort, dirname in {"scifact": "sentence_rag_received", "climate_retrieved": "climate_rag_received",
                            "climate_supplied": "climate_supplied_received"}.items():
        source = root / "artifacts" / dirname
        require(load(source, "input_manifest.json")["models"]["generator"] == identity["generator"], "Generator receipt mismatch")
        for r in load(source, "predictions.json"):
            originals[(cohort, str(r["claim_id"]))] = r
    rows = load(folder, "predictions.json")
    keys = [(r["cohort"], str(r["claim_id"]), r["policy"]) for r in rows]
    expected = {(c, i, p) for c, i in originals for p in identity["policies"]}
    require(len(rows) == len(set(keys)) == len(expected) == 3600 and set(keys) == expected, "Cohort/record IDs mismatch")
    answers = {}
    matched_prior_labels = 0
    baseline_text_matches = 0
    baseline_verdict_matches = 0
    early_rejections = 0
    for r in rows:
        key = (r["cohort"], str(r["claim_id"]))
        original = originals[key]
        prior = gate_rows[key]
        probability = None if r["policy"] == "no_gate" else prior["gate_probabilities"][r["policy"]]

        def generate(kind, question):
            prompt = qwen_prompt(question)
            require(r[kind + "_prompt"] == prompt, "Generated prompt/context mismatch")
            count = r[kind + "_prompt_tokens"]
            require(type(count) is int and 0 < count <= 1024, "Invalid reported token count")
            response = {"answer": r["generated_" + kind], "prompt": prompt, "prompt_tokens": count}
            answer_key = kind + ":" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if answer_key in answers:
                require(answers[answer_key] == response, "Shared prompt cache has inconsistent answers")
            answers[answer_key] = response
            return response

        replay = execute_generation(original["shown_claim"], original["evidence"], generate, probability)
        require(set(r) == set(replay) | {"cohort", "claim_id", "policy", "backend", "true_label"}, "Output schema mismatch")
        for field, value in replay.items():
            require(r[field] == value, f"Controller field mismatch: {field}")
        require(r["backend"] == "live" and r["true_label"] == original["true_label"], "Backend/target mismatch")
        matched_prior_labels += int(r["candidate_label"] == prior[r["policy"] + "_label"])
        if "gate_below_threshold" in r["reasons"]:
            require(not r["generation_requests"] and r["generated_verdict"] is None and r["generated_explanation"] is None,
                    "Gate reject reached generation")
            early_rejections += 1
        if r["policy"] == "no_gate":
            baseline_verdict_matches += int(r["generated_verdict"] == original["generated_verdict"])
            baseline_text_matches += int(r["generated_explanation"] == original["generated_explanation"])
    saved = load(folder, "summary.json")
    summary = {}
    for cohort in ("scifact", "climate_retrieved", "climate_supplied"):
        summary[cohort] = {}
        for policy in identity["policies"]:
            subset = [r for r in rows if r["cohort"] == cohort and r["policy"] == policy]
            result = {
                "metrics": summarize(subset, "candidate_label"),
                "gate_rejected_before_generation": sum("gate_below_threshold" in r["reasons"] for r in subset),
                "policy_verdict_requests": sum("verdict" in r["generation_requests"] for r in subset),
                "policy_explanation_requests": sum("explanation" in r["generation_requests"] for r in subset),
            }
            for metric, value in result["metrics"].items():
                target = saved["cohorts"][cohort][policy]["metrics"][metric]
                require(value is target if value is None else math.isclose(value, target, abs_tol=1e-12, rel_tol=0), "Metric mismatch")
            for field in ("gate_rejected_before_generation", "policy_verdict_requests", "policy_explanation_requests"):
                require(result[field] == saved["cohorts"][cohort][policy][field], "Request count mismatch")
            summary[cohort][policy] = result
    require(saved["backend"] == "live" and saved["unique_live_answers_cached"] == len(answers) == 1800, "Unique prompt receipt mismatch")
    diagnostics = explanation_diagnostics(rows, originals)
    out = root / "artifacts/gated_generation_verification"
    out.mkdir(exist_ok=True)
    write_json_atomic(out / "verified_summary.json", summary)
    write_json_atomic(out / "explanation_diagnostics.json", diagnostics)
    checks = {"records": len(rows), "claims": len(originals), "early_gate_rejections": early_rejections,
              "unique_prompt_keys": len(answers), "candidate_labels_matching_prior_replay": matched_prior_labels,
              "baseline_verdicts_matching_original": baseline_verdict_matches,
              "baseline_explanations_matching_original": baseline_text_matches,
              "neural_inference_independently_rerun": False, "sqlite_cache_independently_inspected": False}
    write_json_atomic(out / "verification_checks.json", checks)
    lines = ["# Gate-before-generation live output verification", "",
             f"All {len(rows)} records from {len(originals)} claims passed output/source/code/protocol identity, controller execution replay, context/prompt, reported token-budget, shared-answer consistency, numeric policy and metric/count checks. All {early_rejections} early gate rejects have zero generation requests. The receipt reports 1800 unique cached live answers, consistent with 1800 distinct prompt keys in these records. SQLite bytes and neural inference were not independently inspected or rerun; these outputs establish export consistency, not independent proof of a model invocation.", "",
             f"Candidate decisions match the previous policy replay in {matched_prior_labels}/3600 records. No-gate verdicts match earlier saved generation in {baseline_verdict_matches}/900 cases; explanations match in {baseline_text_matches}/900. This confirms controller reproducibility on frozen contexts and observed cohorts. It is not a new independent accuracy result or a general sufficiency improvement.", "",
             "| Cohort | Policy | Early gate rejects | Explanation requests | Answered accuracy | Coverage |", "|---|---|---:|---:|---:|---:|"]
    for cohort, policies in summary.items():
        for policy, v in policies.items():
            lines.append(f"| {cohort} | {policy} | {v['gate_rejected_before_generation']} | {v['policy_explanation_requests']} | {v['metrics']['covered_accuracy']:.1%} | {v['metrics']['coverage']:.1%} |")
    lines += ["", "The gate controls requests before generation; numeric rejection follows generation. Individual-policy request reductions are logical counts. The all-policy run includes the no-gate control, so all 900 claims still produce a verdict/explanation shared across passing policies. No latency, memory, energy or monetary saving was measured.", "",
              "## Explanation diagnostic audit", "",
              "Reuse prior sentence-level NLI scores only where live explanation and context exactly match the source export. Scores use evidence as premise and explanation sentence as hypothesis. The 0.7 threshold is a descriptive machine diagnostic, not a validated grounding criterion. Maximum entailment does not verify factual correctness, relation correctness, source authenticity or argument quality. Neutral metacommentary can also score low. Reference verdict labels do not establish explanation correctness.", "",
              "| Cohort | Policy | Answered claims with matching audits | Sentences | Sentences with max entailment >= 0.7 | All-sentence explanations >= 0.7 |", "|---|---|---:|---:|---:|---:|"]
    for cohort, policies in diagnostics.items():
        for policy, v in policies.items():
            lines.append(f"| {cohort} | {policy} | {v['matching_cached_audits']}/{v['answered_claims']} | {v['sentences_audited']} | {v['sentence_max_entailment_at_least_07']} | {v['explanations_all_sentence_max_entailment_at_least_07']} |")
    lines += ["", "Keep the mixed gate results and threshold 0.5 unchanged. Source authentication, fresh retrieval integration, multilingual generation, explanation-quality validation and the final reproducibility audit remain open. No manuscript or GitHub publication.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json_atomic(out / "audit_manifest.json", {"source_output_manifest_sha256": sha256(folder / "output_manifest.json"),
                       "verifier_sha256": sha256(Path(__file__)), "files": {p.name: sha256(p) for p in out.iterdir() if p.name != "audit_manifest.json"}})
    print(json.dumps({"checks": checks, "explanation_diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
