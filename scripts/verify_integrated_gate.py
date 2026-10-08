"""Non-neural replay and descriptive matched-coverage gate audit."""
import json
import math
from pathlib import Path

import numpy as np

from apv_rag.generative_rag import LABELS
from apv_rag.integrated_gate import apply_gate, collapse_context, probability_complete
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2
from apv_rag.splits import sha256, write_json_atomic
from run_semantic_sufficiency import aggregate_features
from run_sentence_rag_scifact import summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def check_hashes(folder):
    hashes = load(folder, "output_manifest.json")
    for name, digest in hashes.items():
        require(Path(name).name == name, "Unsafe manifest path")
        require(sha256(folder / name) == digest, f"Checksum mismatch: {folder.name}/{name}")


def keyed(rows):
    result = {(r["cohort"], str(r["claim_id"])): r for r in rows}
    require(len(result) == len(rows), "Duplicate output IDs")
    return result


def score_rules(row, scores, cosines):
    label_column = {"Refuted": 0, "Supported": 1, "Not Enough Evidence": 2}
    column = label_column[row["no_gate_label"]]
    return {
        **row["gate_probabilities"],
        "cosine_max": float(max(cosines)),
        "verdict_nli_mean": float(np.asarray(scores)[:, column].mean()),
        "verdict_nli_max": float(np.asarray(scores)[:, column].max()),
    }


def selectivity(rows, cache):
    eligible = [r for r in rows if r["no_gate_label"] is not None]
    require(bool(eligible), "No common eligible predictions")
    rankings = {}
    for r in eligible:
        c = cache[f"{r['cohort']}:{r['claim_id']}"]
        for rule, score in score_rules(r, c["scores_cen"], c["cosines"]).items():
            rankings.setdefault(rule, []).append((score, str(r["claim_id"]), r))
    for rule in rankings:
        rankings[rule].sort(key=lambda x: (-x[0], x[1]))
    base_accuracy = sum(r["no_gate_label"] == r["true_label"] for r in eligible) / len(eligible)
    result = {
        "cohort_size": len(rows),
        "numeric_eligible": len(eligible),
        "random_selection_expected_accuracy": base_accuracy,
        "aurc_common_eligible": {},
        "matched_coverage": {},
    }
    for rule, ranking in rankings.items():
        mistakes = np.array([r["no_gate_label"] != r["true_label"] for _, _, r in ranking])
        risk = np.cumsum(mistakes) / np.arange(1, len(mistakes) + 1)
        result["aurc_common_eligible"][rule] = float(risk.mean())
    for gate in ("nli", "embedding", "combined"):
        accepted = [r for r in rows if r[gate + "_label"] is not None]
        k = len(accepted)
        require(k > 0, "No accepted predictions for comparison")
        actual = sum(r[gate + "_label"] == r["true_label"] for r in accepted) / k
        accuracies = {
            rule: sum(r["no_gate_label"] == r["true_label"] for _, _, r in ranking[:k]) / k
            for rule, ranking in rankings.items()
        }
        require(math.isclose(actual, accuracies[gate], abs_tol=1e-12), "Gate/top-k disagreement")
        result["matched_coverage"][gate] = {
            "accepted": k,
            "coverage_all_claims": k / len(rows),
            "fixed_threshold_gate_accuracy": actual,
            "top_k_accuracy": accuracies,
        }
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/integrated_gate_received"
    check_hashes(folder)
    identity = load(folder, "input_manifest.json")
    for name, digest in identity["code_sha256"].items():
        require(sha256(root / name) == digest, f"Code mismatch: {name}")
    require(sha256(root / "docs/INTEGRATED_GATE_REPLAY.md") == identity["protocol_sha256"], "Protocol mismatch")
    require(identity["settings"] == {"seed": 369, "threads": 4, "dtype": "float32", "nli_tokens": 256,
            "embedding_tokens": 384, "gate_threshold": .5, "copy_count": 5}, "Unexpected run settings")
    learned = root / "artifacts/semantic_sufficiency_received"
    check_hashes(learned)
    require(sha256(learned / "models.json") == identity["learned_models_sha256"], "Classifier mismatch")
    require(load(learned, "input_manifest.json")["models"] == identity["feature_models"], "Feature-model identity mismatch")
    models = load(learned, "models.json")
    require(set(models) == {"nli", "embedding", "combined"}, "Unexpected classifier variants")
    rows = load(folder, "predictions.json")
    output = keyed(rows)
    copies = keyed(load(folder, "copy_audit.json"))
    cache = load(folder, "feature_cache.json")
    expected_keys = set()
    copy_effects = {name: [] for name in models}
    flips = {name: 0 for name in models}
    sources = {"scifact": "sentence_rag_received", "climate_retrieved": "climate_rag_received",
               "climate_supplied": "climate_supplied_received"}
    for cohort, dirname in sources.items():
        source = root / "artifacts" / dirname
        check_hashes(source)
        require(sha256(source / "predictions.json") == identity["sources"][cohort], "Generation source mismatch")
        original = load(source, "predictions.json")
        require(len(original) == len({str(r['claim_id']) for r in original}) == 300, "Unexpected source cohort")
        for old in original:
            key = (cohort, str(old["claim_id"]))
            cache_key = f"{cohort}:{old['claim_id']}"
            expected_keys.add(key)
            r = output[key]
            evidence = collapse_context(old["evidence"])
            require(r["true_label"] == old["true_label"] in LABELS, "Verdict label mismatch")
            require(r["context_families"] == [str(e.get("family_id", e["id"])) for e in evidence], "Context mismatch")
            c = cache[cache_key]
            scores = np.asarray(c["scores_cen"], dtype=float)
            cosines = np.asarray(c["cosines"], dtype=float)
            require(scores.shape == (len(evidence), 3) and cosines.shape == (len(evidence),), "Cache shape mismatch")
            require(np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(), "Invalid NLI scores")
            require(np.allclose(scores.sum(1), 1, rtol=0, atol=1e-6), "NLI normalization mismatch")
            require(np.isfinite(cosines).all() and (abs(cosines) <= 1.00001).all(), "Invalid cosine scores")
            features = aggregate_features(scores, cosines)
            require(np.allclose(features, r["features"], rtol=0, atol=1e-12), "Feature aggregation mismatch")
            numeric = numeric_provenance_v2(old["generated_explanation"] or "", old["shown_claim"], old["evidence"])
            reasons = ["numeric_value_absent"] if numeric["absent_from_inputs"] else []
            if not old["raw_candidate_label"] or not old["generated_explanation"]:
                reasons.append("invalid_or_empty_generation")
            require(reasons == r["numeric_reasons"], "Numeric policy mismatch")
            require(r["no_gate_label"] == (old["raw_candidate_label"] if not reasons else None), "No-gate policy mismatch")
            synthetic = [dict(evidence[0], id=f"copy{i}", family_id=evidence[0].get("family_id", evidence[0]["id"])) for i in range(5)]
            require(collapse_context(evidence + synthetic) == evidence, "Synthetic collapse changed context")
            require(copies[key]["collapsed_context_unchanged"] is True, "Copy audit mismatch")
            duplicated = aggregate_features(c["scores_cen"] + [c["scores_cen"][0]] * 5,
                                            c["cosines"] + [c["cosines"][0]] * 5)
            for name, model in models.items():
                probability = probability_complete(features, model)
                require(math.isclose(probability, r["gate_probabilities"][name], abs_tol=1e-10, rel_tol=0), "Gate probability mismatch")
                require(r[name + "_label"] == apply_gate(old["raw_candidate_label"], reasons, probability), "Gate decision mismatch")
                duplicate_probability = probability_complete(duplicated, model)
                shift = duplicate_probability - probability
                require(math.isclose(shift, copies[key]["without_collapse_probability_shift"][name], abs_tol=1e-10, rel_tol=0), "Copy effect mismatch")
                copy_effects[name].append(abs(shift))
                flips[name] += int((probability >= .5) != (duplicate_probability >= .5))
    require(set(output) == set(copies) == expected_keys and len(rows) == 900, "Output cohort mismatch")
    require(set(cache) == {f"{c}:{i}" for c, i in expected_keys}, "Feature cache IDs mismatch")
    metrics = {}
    audit = {}
    saved = load(folder, "metrics.json")
    for cohort in sources:
        subset = [r for r in rows if r["cohort"] == cohort]
        metrics[cohort] = {}
        for field in ("no_gate_label", "nli_label", "embedding_label", "combined_label"):
            replay = summarize(subset, field)
            require(set(replay) == set(saved[cohort][field]), "Metric schema mismatch")
            for name, value in replay.items():
                target = saved[cohort][field][name]
                require(value is target if value is None else math.isclose(value, target, abs_tol=1e-12, rel_tol=0), "Metric replay mismatch")
            metrics[cohort][field] = replay
        audit[cohort] = selectivity(subset, cache)
    copy_summary = {name: {"mean_absolute_shift_without_collapse": float(np.mean(values)),
                   "max_absolute_shift_without_collapse": float(max(values)),
                   "threshold_flips_without_collapse": flips[name]} for name, values in copy_effects.items()}
    out = root / "artifacts/integrated_gate_verification"
    out.mkdir(exist_ok=True)
    write_json_atomic(out / "verified_metrics.json", metrics)
    write_json_atomic(out / "selectivity_audit.json", audit)
    write_json_atomic(out / "copy_summary.json", copy_summary)
    lines = ["# Integrated gate verification and selective prediction audit", "",
             "All 900 records passed file/code/protocol/source identity, cache-shape, feature, serialized classifier, numeric policy, decision, synthetic-copy and metric replay checks. Neural inference was not independently rerun. Model-file hashes are checked against the saved training receipt, not local model bytes.", "",
             "At the fixed 0.5 threshold, NLI-gated answered accuracy is 64.7% on SciFact at 22.7% coverage and 53.1% on retrieved climate at 16.3% coverage. These exceed the simple cosine/verdict-NLI ranking baselines at the same accepted counts. The supplied-climate result reverses: 32.8% at 19.3% coverage, below the simple verdict-NLI mean ranking's 53.4% at the same count. Embedding and combined gates reduce answered accuracy relative to the common no-gate policy in all three cohorts. This is mixed selective-prediction evidence, not a general sufficiency improvement. Preserve all variants and results; do not choose a deployment gate from these observations.", "",
             "The replay can only retain a saved verdict or abstain. It cannot increase all-claim accuracy when abstentions count as errors. Evaluate selective accuracy with coverage and matched-coverage baselines.", "",
             "| Cohort | Policy | Coverage | All-claim accuracy | Answered accuracy |", "|---|---|---:|---:|---:|"]
    for cohort, policies in metrics.items():
        for field, m in policies.items():
            lines.append(f"| {cohort} | {field.removesuffix('_label')} | {m['coverage']:.1%} | {m['accuracy_all_claims_abstentions_as_errors']:.1%} | {m['covered_accuracy']:.1%} |")
    lines += ["", "Common eligible pool: predictions that pass the shared numeric/generation policy. Ranking scores use no reference labels. Baselines were introduced after observing the gate results, so this comparison is exploratory. At each fixed gate's accepted count, compare every score's top-k accuracy; these counts are diagnostics, not tuned deployment thresholds. Ties use claim IDs as strings. Random-selection expected accuracy equals the eligible pool accuracy.", "",
              "| Cohort | Fixed gate count | NLI gate ranking | Embedding gate ranking | Combined gate ranking | Cosine max | Verdict NLI mean | Verdict NLI max |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    rules = ("nli", "embedding", "combined", "cosine_max", "verdict_nli_mean", "verdict_nli_max")
    for cohort, a in audit.items():
        for gate, values in a["matched_coverage"].items():
            cells = " | ".join(f"{values['top_k_accuracy'][rule]:.1%}" for rule in rules)
            lines.append(f"| {cohort} | {gate}: {values['accepted']} | {cells} |")
    lines += ["", "AURC is the average empirical error rate over all nonempty ranked prefixes of the common eligible pool; lower is better. It is descriptive and depends on ranking ties. No confidence intervals, threshold selection or claims of safe abstention are made.", "",
              "| Cohort | NLI gate | Embedding gate | Combined gate | Cosine max | Verdict NLI mean | Verdict NLI max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for cohort, a in audit.items():
        cells = " | ".join(f"{a['aurc_common_eligible'][rule]:.4f}" for rule in rules)
        lines.append(f"| {cohort} | {cells} |")
    lines += ["", "All 900 collapsed contexts remain identical after adding five copies of their first source. Without collapse, mean/std feature aggregation changes and can cross the gate threshold. This is deterministic context handling; it does not establish source authenticity, independence, explanation quality or generation robustness.", "",
              "| Gate | Mean absolute probability shift | Maximum shift | Threshold flips without collapse |", "|---|---:|---:|---:|"]
    for name, a in copy_summary.items():
        lines.append(f"| {name} | {a['mean_absolute_shift_without_collapse']:.4f} | {a['max_absolute_shift_without_collapse']:.4f} | {a['threshold_flips_without_collapse']} / 900 |")
    lines += ["", "Keep the original fixed policies and their results. Do not tune the 0.5 threshold on these observed cohorts. Constructed completeness performance does not establish real passage sufficiency. Full generation integration, source authentication, multilingual generation and final reproducibility checks remain open.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json_atomic(out / "audit_manifest.json", {"source_output_manifest_sha256": sha256(folder / "output_manifest.json"),
                       "verifier_sha256": sha256(Path(__file__)), "files": {p.name: sha256(p) for p in out.iterdir() if p.name != "audit_manifest.json"}})
    print("Verified 900 records; neural inference not rerun.")
    print(json.dumps({"selectivity": audit, "copy_effects": copy_summary}, indent=2))


if __name__ == "__main__":
    main()
