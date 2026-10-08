"""Verify XFEVER exports and run the ten prespecified grouped comparisons."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from apv_rag.direct_nli import TARGET_LABELS
from apv_rag.metrics import multiclass_brier
from apv_rag.paired_statistics import holm_adjust
from apv_rag.splits import sha256, write_json_atomic
from apv_rag.xfever import LABEL_MAP, validate_parallel


def check(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            check(a[k], b[k])
    else:
        assert np.isclose(a, b, atol=1e-12, rtol=1e-10)


def score(y, pred):
    m = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    d = m.sum(0) + m.sum(1)
    return float(np.divide(2 * m.diagonal(), d, out=np.zeros(3), where=d != 0).mean())


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = root / "artifacts/xfever_received"
    manifest = json.loads((inputs / "output_manifest.json").read_text())
    for name in ("predictions.json", "stress_summary.json", "input_manifest.json"):
        assert sha256(inputs / name) == manifest[name], name
    identity = json.loads((inputs / "input_manifest.json").read_text())
    for name, digest in identity["code_sha256"].items():
        assert sha256(root / name) == digest, name
    assert sha256(root / "docs/XFEVER_NLI_STRESS.md") == identity["protocol_sha256"]
    source = root / "data/external/xfever/zenodo_8206962/evaluation_inputs_v1"
    sets = {}
    for name, digest in identity["files"].items():
        assert sha256(source / name) == digest
        sets[name] = [json.loads(line) for line in (source / name).read_text().splitlines()]
    count = validate_parallel(sets)
    reference = sets["en/test.6h.jsonl"]
    parent = list(range(count))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, r in enumerate(reference):
        for key in [("id", r["id"]), ("page", r["page"])]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = np.array([find(i) for i in range(count)])
    names = sorted(set(groups))
    positions = [np.flatnonzero(groups == g) for g in names]
    membership = np.array([names.index(g) for g in groups])
    rows = json.loads((inputs / "predictions.json").read_text())
    lookup = {(r["file"], r["row"]): r for r in rows}
    expected = {(name, i) for name in sets for i in range(count)}
    assert len(rows) == len(expected) and set(lookup) == expected
    mapping = {label: i for i, label in enumerate(TARGET_LABELS)}
    truth_labels = [LABEL_MAP[r["label"]] for r in reference]
    truth = np.array([mapping[label] for label in truth_labels])
    summary = json.loads((inputs / "stress_summary.json").read_text())
    assert summary["rows_per_file"] == count
    assert summary["unique_claim_ids"] == len({r["id"] for r in reference})
    arrays = {}
    for name in sets:
        selected = [lookup[name, i] for i in range(count)]
        assert [r["claim_id"] for r in selected] == [r["id"] for r in reference]
        assert [r["english_page"] for r in selected] == [r["page"] for r in reference]
        assert [r["true_label"] for r in selected] == truth_labels
        p = np.asarray([r["probabilities"] for r in selected])
        assert (
            p.shape == (count, 3)
            and np.isfinite(p).all()
            and (p >= 0).all()
            and (p <= 1).all()
            and np.allclose(p.sum(1), 1)
        )
        pred = np.array([mapping[r["predicted_label"]] for r in selected])
        assert np.array_equal(pred, p.argmax(1))
        arrays[name] = pred
        report = next(r for r in summary["results"] if r["file"] == name)
        labels = np.array(TARGET_LABELS)[pred]
        check(
            report["macro_f1"],
            f1_score(
                truth_labels, labels, labels=list(TARGET_LABELS), average="macro", zero_division=0
            ),
        )
        check(report["accuracy"], accuracy_score(truth_labels, labels))
        check(report["brier"], multiclass_brier(truth_labels, p, TARGET_LABELS))
        check(
            report["per_class"],
            classification_report(
                truth_labels, labels, labels=list(TARGET_LABELS), output_dict=True, zero_division=0
            ),
        )
    english = arrays["en/test.6h.jsonl"]
    for r in summary["results"]:
        check(
            r["prediction_disagreement_with_english"], float(np.mean(arrays[r["file"]] != english))
        )
    results = []
    for name in sorted(sets):
        if name == "en/test.6h.jsonl":
            continue
        target = arrays[name]
        observed = score(truth, target) - score(truth, english)
        rng = np.random.default_rng(369)
        effects = []
        for _ in range(2000):
            ix = np.concatenate([positions[g] for g in rng.integers(len(names), size=len(names))])
            effects.append(score(truth[ix], target[ix]) - score(truth[ix], english[ix]))
        extreme = 0
        for _ in range(10000):
            swap = rng.integers(2, size=len(names)).astype(bool)[membership]
            effect = score(truth, np.where(swap, english, target)) - score(
                truth, np.where(swap, target, english)
            )
            extreme += abs(effect) >= abs(observed) - 1e-12
        results.append(
            {
                "file": name,
                "macro_f1": score(truth, target),
                "gain_over_english": observed,
                "group_bootstrap_95_interval": np.quantile(effects, [0.025, 0.975]).tolist(),
                "randomization_two_sided_p": (extreme + 1) / 10001,
            }
        )
    for r, adjusted in zip(
        results, holm_adjust([r["randomization_two_sided_p"] for r in results]), strict=True
    ):
        r["holm_adjusted_p_ten_comparisons"] = adjusted
    output = root / "artifacts/xfever_statistics"
    output.mkdir(exist_ok=True)
    write_json_atomic(
        output / "statistics.json",
        {
            "verified_predictions": len(rows),
            "rows_per_file": count,
            "groups": len(names),
            "group_rule": "connected English claim IDs or English page identifiers",
            "english_macro_f1": score(truth, english),
            "results": results,
            "bootstrap_samples": 2000,
            "randomization_samples": 10000,
            "seed": 369,
            "verification_limit": "decisions and scores reproduced; NLI inference not rerun",
            "input_sha256": {
                n: sha256(inputs / n)
                for n in ("predictions.json", "input_manifest.json", "stress_summary.json")
            },
        },
    )
    lines = [
        "# XFEVER English-model transfer stress",
        "",
        (
            f"{len(rows)} predictions verified; {count} aligned rows per file; "
            f"{len(names)} connected groups."
        ),
        f"English macro-F1: {score(truth, english):.4f}.",
        "",
        (
            "| Target file | Macro-F1 | Difference from English | Marginal 95% group "
            "interval | Holm p |"
        ),
        "|---|---:|---:|---|---:|",
    ]
    for r in results:
        lo, hi = r["group_bootstrap_95_interval"]
        lines.append(
            f"| {r['file']} | {r['macro_f1']:.4f} | {r['gain_over_english']:+.4f} | "
            f"[{lo:+.4f}, {hi:+.4f}] | {r['holm_adjusted_p_ten_comparisons']:.4f} |"
        )
    lines += [
        "",
        (
            "This tests an English NLI model on supplied translated evidence, not a "
            "multilingual model or retrieval system. Translations share source claims"
            " and groups; language files are not independent datasets. Repeated "
            "evidence rows remain included. No target fitting was performed. Human "
            "translation is upstream benchmark data, not new project annotation."
        ),
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
