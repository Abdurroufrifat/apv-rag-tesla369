"""Exploratory grouped paired statistics for expanded retrieved features."""

import json
from pathlib import Path

import numpy as np

from apv_rag.evidence_baseline import LABELS
from apv_rag.nli_comparison import SEEDS
from apv_rag.paired_statistics import holm_adjust, macro_score
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = root / "artifacts/retrieved_feature_received"
    source = inputs / "predictions.json"
    manifest = json.loads((inputs / "output_manifest.json").read_text())
    if sha256(source) != manifest["predictions.json"]:
        raise ValueError("received prediction checksum mismatch")
    rows = json.loads(source.read_text())
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    ids = json.loads((split / "validation_indices.json").read_text())
    groups = json.loads((split / "group_assignments.json").read_text())
    lookup = {
        (r["method"], r["representation"], r["seed"], r["rule"], r["upstream_index"]): r
        for r in rows
    }
    expected = {
        (m, rep, s, rule, i)
        for m in ("bm25_nli", "dense_nli")
        for rep in ("base_7", "expanded_18")
        for s in SEEDS
        for rule in ("argmax", "prior_adjusted")
        for i in ids
    }
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("prediction pairing incomplete or duplicated")
    group_names = sorted({groups[str(i)] for i in ids})
    positions = [
        np.asarray([j for j, i in enumerate(ids) if groups[str(i)] == g]) for g in group_names
    ]
    group_number = {name: j for j, name in enumerate(group_names)}
    membership = np.asarray([group_number[groups[str(i)]] for i in ids])
    mapping = {label: i for i, label in enumerate(LABELS)}
    truth_labels = [
        lookup[("bm25_nli", "base_7", SEEDS[0], "argmax", i)]["true_label"] for i in ids
    ]
    truth = np.asarray([mapping[label] for label in truth_labels])
    output = root / "artifacts/retrieved_feature_statistics"
    if output.exists():
        raise FileExistsError("refusing to overwrite completed analysis")
    results = []
    for method in ("bm25_nli", "dense_nli"):
        for rule in ("argmax", "prior_adjusted"):
            matrices = {}
            for representation in ("base_7", "expanded_18"):
                matrices[representation] = []
                for seed in SEEDS:
                    selected = [lookup[(method, representation, seed, rule, i)] for i in ids]
                    if [r["true_label"] for r in selected] != truth_labels:
                        raise ValueError("true labels differ across comparisons")
                    matrices[representation].append(
                        [mapping[r["predicted_label"]] for r in selected]
                    )
            a, b = np.asarray(matrices["base_7"]), np.asarray(matrices["expanded_18"])

            def difference(y, left, right):
                return float(
                    np.mean(
                        [
                            macro_score(y, after) - macro_score(y, before)
                            for before, after in zip(left, right, strict=True)
                        ]
                    )
                )

            observed = difference(truth, a, b)
            rng = np.random.default_rng(369)
            effects = []
            for _ in range(2000):
                draw = rng.integers(len(group_names), size=len(group_names))
                ix = np.concatenate([positions[g] for g in draw])
                effects.append(difference(truth[ix], a[:, ix], b[:, ix]))
            extreme = 0
            for _ in range(10000):
                swap = rng.integers(2, size=len(group_names)).astype(bool)[membership][None, :]
                effect = difference(truth, np.where(swap, b, a), np.where(swap, a, b))
                extreme += abs(effect) >= abs(observed) - 1e-12
            results.append(
                {
                    "method": method,
                    "rule": rule,
                    "base_mean_macro_f1": float(np.mean([macro_score(truth, row) for row in a])),
                    "expanded_mean_macro_f1": float(
                        np.mean([macro_score(truth, row) for row in b])
                    ),
                    "mean_seed_macro_f1_gain": observed,
                    "group_bootstrap_95_interval": np.percentile(effects, [2.5, 97.5]).tolist(),
                    "randomization_two_sided_p": (extreme + 1) / 10001,
                }
            )
            print(f"Completed statistics: {method} {rule}", flush=True)
    for row, adjusted in zip(
        results, holm_adjust([r["randomization_two_sided_p"] for r in results]), strict=True
    ):
        row["holm_adjusted_p_four_comparisons"] = adjusted
    report = {
        "status": "exploratory internal-validation statistics; not new confirmation",
        "claims": len(ids),
        "connected_groups": len(group_names),
        "seeds": list(SEEDS),
        "bootstrap_samples": 2000,
        "randomization_samples": 10000,
        "random_seed": 369,
        "pairing": "same group sampling/swaps across both representations and all seeds",
        "multiplicity": "Holm across both methods and both rules (four comparisons)",
        "official_dev_records_read": 0,
        "results": results,
        "limitations": [
            "feature design followed observed development errors",
            "Holm correction does not account for all earlier development searches",
            "bootstrap intervals are marginal, not simultaneous",
            "oracle excerpt corpus; no full RAG or leaderboard claim",
        ],
        "input_sha256": {
            "predictions.json": sha256(source),
            "validation_indices.json": sha256(split / "validation_indices.json"),
            "group_assignments.json": sha256(split / "group_assignments.json"),
        },
    }
    output.mkdir()
    write_json_atomic(output / "statistics.json", report)
    lines = [
        "# Retrieved-feature paired statistics",
        "",
        "Exploratory internal-validation results; not new held-out confirmation.",
        "",
        f"{len(ids)} claims in {len(group_names)} connected groups; five seeds.",
        "2000 group-bootstrap draws; 10000 two-sided paired group randomizations.",
        "Holm correction covers both methods and both decision rules.",
        "",
        "| Method | Rule | Macro-F1 gain | Marginal 95% interval | Holm p |",
        "|---|---|---:|---|---:|",
    ]
    for r in results:
        lower, upper = r["group_bootstrap_95_interval"]
        lines.append(
            f"| {r['method']} | {r['rule']} | {r['mean_seed_macro_f1_gain']:+.4f} | "
            f"[{lower:+.4f}, {upper:+.4f}] | {r['holm_adjusted_p_four_comparisons']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Correction covers these four comparisons, not all earlier feature searches.",
            "Feature design and repeated validation were adaptive. Separate new evaluation data",
            "is required to confirm a revised method. Earlier held-out results remain unchanged.",
            "",
        ]
    )
    (output / "STATISTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
