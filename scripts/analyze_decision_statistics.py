"""Grouped paired bootstrap and randomization of saved decision predictions."""

import json
from pathlib import Path

import numpy as np

from apv_rag.evidence_baseline import LABELS
from apv_rag.nli_comparison import SEEDS
from apv_rag.paired_statistics import holm_adjust, macro_score
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts/cached_decision_comparison/decision_predictions.json"
    rows = json.loads(source.read_text())
    groups_path = root / "data/processed/averitec/phase2b_split_v0_1/group_assignments.json"
    groups = json.loads(groups_path.read_text())
    mapping = {label: i for i, label in enumerate(LABELS)}
    output = root / "artifacts/cached_decision_statistics"
    if output.exists():
        raise FileExistsError("refusing to overwrite completed statistical analysis")
    results = []
    for method in ("bm25_nli", "dense_nli"):
        subset = [r for r in rows if r["method"] == method]
        lookup = {(r["seed"], r["upstream_index"]): r for r in subset}
        ids = sorted({r["upstream_index"] for r in subset})
        if len(lookup) != len(ids) * len(SEEDS):
            raise ValueError("prediction pairing is incomplete or duplicated")
        truth = np.asarray([mapping[lookup[(SEEDS[0], i)]["true_label"]] for i in ids])
        if any(
            lookup[(seed, i)]["true_label"] != lookup[(SEEDS[0], i)]["true_label"]
            for seed in SEEDS
            for i in ids
        ):
            raise ValueError("true labels differ between seeds")
        original = np.asarray(
            [[mapping[lookup[(seed, i)]["predicted_label"]] for i in ids] for seed in SEEDS]
        )
        selected = np.asarray(
            [[mapping[lookup[(seed, i)]["selected_label"]] for i in ids] for seed in SEEDS]
        )
        group_names = sorted({groups[str(i)] for i in ids})
        group_positions = [
            np.asarray([j for j, i in enumerate(ids) if groups[str(i)] == name])
            for name in group_names
        ]
        group_index = {name: j for j, name in enumerate(group_names)}
        membership = np.asarray([group_index[groups[str(i)]] for i in ids])

        def difference(y, a, b):
            return float(
                np.mean([macro_score(y, b[j]) - macro_score(y, a[j]) for j in range(len(SEEDS))])
            )

        observed = difference(truth, original, selected)
        rng = np.random.default_rng(369)
        effects = []
        for _ in range(2000):
            sample = rng.integers(0, len(group_names), size=len(group_names))
            ix = np.concatenate([group_positions[i] for i in sample])
            effects.append(difference(truth[ix], original[:, ix], selected[:, ix]))
        extreme = 0
        for _ in range(10000):
            swap = rng.integers(0, 2, size=len(group_names)).astype(bool)[membership]
            a = np.where(swap[None, :], selected, original)
            b = np.where(swap[None, :], original, selected)
            extreme += abs(difference(truth, a, b)) >= abs(observed) - 1e-12
        results.append(
            {
                "method": method,
                "claims": len(ids),
                "groups": len(group_names),
                "mean_seed_macro_f1_gain": observed,
                "group_bootstrap_95_interval": np.percentile(effects, [2.5, 97.5]).tolist(),
                "randomization_two_sided_p": (extreme + 1) / 10001,
            }
        )
    adjusted = holm_adjust([r["randomization_two_sided_p"] for r in results])
    for result, value in zip(results, adjusted, strict=True):
        result["holm_adjusted_p"] = value
    output.mkdir(parents=True)
    report = {
        "status": "exploratory development analysis; not final-test confirmation",
        "unit": "frozen connected provenance/claim group",
        "seeds": list(SEEDS),
        "bootstrap_repetitions": 2000,
        "randomization_repetitions": 10000,
        "random_seed": 369,
        "official_dev_records_used": 0,
        "results": results,
        "input_sha256": {source.name: sha256(source), groups_path.name: sha256(groups_path)},
    }
    write_json_atomic(output / "decision_statistics.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
