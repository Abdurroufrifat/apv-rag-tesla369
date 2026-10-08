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


def verify(root, folder, protocol):
    inputs = root / folder
    manifest = json.loads((inputs / "output_manifest.json").read_text())
    for name in ("predictions.json", "stress_summary.json", "input_manifest.json"):
        assert sha256(inputs / name) == manifest[name], name
    identity = json.loads((inputs / "input_manifest.json").read_text())
    for name, digest in identity["code_sha256"].items():
        assert sha256(root / name) == digest, name
    assert sha256(root / protocol) == identity["protocol_sha256"]
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
    return truth, arrays, positions, membership, len(names), identity


def main():
    root = Path(__file__).resolve().parents[1]
    truth, base, positions, membership, groups, original_identity = verify(
        root, "artifacts/xfever_received", "docs/XFEVER_NLI_STRESS.md"
    )
    truth_new, control, _, _, _, identity = verify(
        root, "artifacts/xfever_multilingual_received", "docs/XFEVER_MULTILINGUAL_CONTROL.md"
    )
    assert np.array_equal(truth, truth_new)
    assert identity["files"] == original_identity["files"]
    assert identity["inference_dtype"] == "float32"
    assert identity["seed"] == 369 and identity["threads"] == 4
    assert identity["max_pair_tokens"] == 256
    results = []
    for name in sorted(base):
        old, new = base[name], control[name]
        observed = score(truth, new) - score(truth, old)
        rng = np.random.default_rng(369)
        effects = []
        for _ in range(2000):
            ix = np.concatenate([positions[g] for g in rng.integers(groups, size=groups)])
            effects.append(score(truth[ix], new[ix]) - score(truth[ix], old[ix]))
        extreme = 0
        for _ in range(10000):
            swap = rng.integers(2, size=groups).astype(bool)[membership]
            effect = score(truth, np.where(swap, old, new)) - score(truth, np.where(swap, new, old))
            extreme += abs(effect) >= abs(observed) - 1e-12
        results.append(
            {
                "file": name,
                "english_model_macro_f1": score(truth, old),
                "multilingual_model_macro_f1": score(truth, new),
                "gain": observed,
                "group_bootstrap_95_interval": np.quantile(effects, [0.025, 0.975]).tolist(),
                "randomization_two_sided_p": (extreme + 1) / 10001,
            }
        )
    for row, adjusted in zip(
        results, holm_adjust([row["randomization_two_sided_p"] for row in results]), strict=True
    ):
        row["holm_adjusted_p_eleven_comparisons"] = adjusted
    output = root / "artifacts/xfever_multilingual_statistics"
    output.mkdir(exist_ok=True)
    report = {
        "scope": "exploratory post-English-results model comparison",
        "verified_predictions": 13200,
        "rows_per_file": len(truth),
        "groups": groups,
        "group_rule": "connected English claim IDs or English page identifiers",
        "results": results,
        "bootstrap_samples": 2000,
        "randomization_samples": 10000,
        "seed": 369,
        "verification_limit": (
            "predictions and metrics verified; model inference not rerun; "
            "local model weights unavailable"
        ),
        "limitations": [
            "FEVER-related model training overlap cannot be ruled out",
            "supplied evidence only, no retrieval or generative RAG",
            "model comparison mixes training and architecture differences",
        ],
        "input_sha256": {
            folder: {
                n: sha256(root / "artifacts" / folder / n)
                for n in (
                    "predictions.json",
                    "input_manifest.json",
                    "stress_summary.json",
                    "output_manifest.json",
                )
            }
            for folder in ("xfever_received", "xfever_multilingual_received")
        },
    }
    write_json_atomic(output / "statistics.json", report)
    lines = [
        "# Exploratory XFEVER multilingual model comparison",
        "",
        f"13,200 predictions verified; {len(truth)} rows per file; {groups} connected groups.",
        "",
        (
            "| File | English model F1 | Multilingual model F1 | Gain | "
            "Marginal 95% interval | Holm p |"
        ),
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in results:
        lo, hi = row["group_bootstrap_95_interval"]
        lines.append(
            f"| {row['file']} | {row['english_model_macro_f1']:.4f} | "
            f"{row['multilingual_model_macro_f1']:.4f} | {row['gain']:+.4f} | "
            f"[{lo:+.4f}, {hi:+.4f}] | {row['holm_adjusted_p_eleven_comparisons']:.4f} |"
        )
    lines += [
        "",
        report["scope"] + ".",
        report["verification_limit"] + ".",
        "Intervals are marginal, not simultaneous. Holm correction covers eleven comparisons.",
        *report["limitations"],
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json_atomic(
        output / "analysis_manifest.json",
        {
            "code_sha256": sha256(Path(__file__)),
            "outputs": {n: sha256(output / n) for n in ("statistics.json", "RESULTS.md")},
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
