"""Verify received binary proxy predictions and summarize internal validation."""

import json
from pathlib import Path

import numpy as np
from run_sufficiency_proxy import metrics

from apv_rag.nli_comparison import SEEDS
from apv_rag.splits import sha256, write_json_atomic
from apv_rag.sufficiency_proxy import proxy_labels


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/sufficiency_proxy_received"
    manifest = json.loads((folder / "output_manifest.json").read_text())
    for name in ("input_manifest.json", "predictions.json", "proxy_summary.json"):
        assert sha256(folder / name) == manifest[name], name
    identity = json.loads((folder / "input_manifest.json").read_text())
    for name, digest in identity["code_sha256"].items():
        assert sha256(root / name) == digest, name
    assert sha256(root / "docs/SUFFICIENCY_PROXY.md") == identity["protocol_sha256"]
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    source = root / "data/external/averitec/official_7c62d1e/train.json"
    assert sha256(source) == identity["inputs_sha256"]["source"]
    for name in ("train_indices.json", "validation_indices.json", "group_assignments.json"):
        assert sha256(split / name) == identity["inputs_sha256"][name], name
    records = json.loads(source.read_text())
    train_ids = json.loads((split / "train_indices.json").read_text())
    ids = json.loads((split / "validation_indices.json").read_text())
    groups = json.loads((split / "group_assignments.json").read_text())
    assert not {groups[str(i)] for i in ids} & {groups[str(i)] for i in train_ids}
    train_y = proxy_labels([records[i]["label"] for i in train_ids])
    truth = proxy_labels([records[i]["label"] for i in ids])
    summary = json.loads((folder / "proxy_summary.json").read_text())
    assert summary["official_dev_records_used"] == 0
    assert summary["train_rows"] == len(train_ids) and summary["validation_rows"] == len(ids)
    assert np.isclose(summary["training_positive_fraction"], train_y.mean())
    assert np.isclose(summary["validation_positive_fraction"], truth.mean())
    rows = json.loads((folder / "predictions.json").read_text())
    lookup = {(r["method"], r["representation"], r["seed"], r["claim_index"]): r for r in rows}
    keys = [
        (m, rep, seed)
        for m in ("bm25_nli", "dense_nli")
        for rep in ("base7", "expanded18")
        for seed in SEEDS
    ]
    expected = {(*key, index) for key in keys for index in ids}
    assert len(rows) == len(expected) and set(lookup) == expected
    reports = {(r["method"], r["representation"], r["seed"]): r for r in summary["results"]}
    assert len(summary["results"]) == len(keys) and set(reports) == set(keys)
    output_rows = []
    baseline = metrics(truth, np.full(len(ids), train_y.mean()), 0.5)
    for key in keys:
        report = reports[key]
        selected = [lookup[(*key, index)] for index in ids]
        assert [r["true_proxy_label"] for r in selected] == truth.tolist()
        assert [r["group"] for r in selected] == [str(groups[str(i)]) for i in ids]
        p = np.array([r["probability_non_nei"] for r in selected])
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        threshold = report["training_oof_selected_threshold"]
        grid = {float(k): v for k, v in report["training_oof_grid"].items()}
        assert set(grid) == {i / 10 for i in range(1, 10)}
        assert threshold == min(grid, key=lambda t: (-grid[t], abs(t - 0.5), t))
        assert all(r["threshold"] == threshold for r in selected)
        assert [r["predicted_proxy_label"] for r in selected] == (p >= threshold).astype(
            int
        ).tolist()
        for name, computed in (
            ("validation_metrics", metrics(truth, p, threshold)),
            ("fixed_half_metrics", metrics(truth, p, 0.5)),
            ("training_prior_baseline", baseline),
        ):
            assert report[name].keys() == computed.keys()
            for metric, value in computed.items():
                assert np.isclose(value, report[name][metric], atol=1e-12, rtol=1e-10)
        width = 7 if key[1] == "base7" else 18
        for field in ("model_coefficients", "scaler_mean", "scaler_scale"):
            a = np.asarray(report[field])
            assert a.shape == (width,) and np.isfinite(a).all()
        assert (np.asarray(report["scaler_scale"]) > 0).all()
        assert np.asarray(report["model_intercept"]).shape == (1,)
    for method in ("bm25_nli", "dense_nli"):
        for rep in ("base7", "expanded18"):
            subset = [reports[(method, rep, seed)]["validation_metrics"] for seed in SEEDS]
            means = {name: float(np.mean([x[name] for x in subset])) for name in baseline}
            output_rows.append({"method": method, "representation": rep, "mean_metrics": means})
    output = root / "artifacts/sufficiency_proxy_statistics"
    output.mkdir(exist_ok=True)
    result = {
        "verified_predictions": len(rows),
        "validation_rows": len(ids),
        "validation_groups": len({groups[str(i)] for i in ids}),
        "non_nei_rows": int(truth.sum()),
        "nei_rows": int((truth == 0).sum()),
        "training_prior_baseline": baseline,
        "results": output_rows,
        "scope": "exploratory benchmark-label proxy, not retrieved-evidence sufficiency",
        "verification_limit": (
            "metrics and decisions reproduced; raw OOF probabilities, fitting "
            "and feature matrices not independently replayed"
        ),
        "input_sha256": {n: sha256(folder / n) for n in manifest},
    }
    write_json_atomic(output / "verification.json", result)
    lines = [
        "# Learned NEI proxy verification",
        "",
        f"{len(rows):,} predictions verified on {len(ids)} validation claims "
        f"({result['validation_groups']} groups). NEI: {result['nei_rows']}; "
        f"non-NEI: {result['non_nei_rows']}.",
        "",
        "| Features | Macro-F1 | Balanced accuracy | AUROC | Positive AP | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in output_rows + [
        {"method": "training prior", "representation": "baseline", "mean_metrics": baseline}
    ]:
        m = row["mean_metrics"]
        lines.append(
            f"| {row['method']} {row['representation']} | {m['macro_f1']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | {m['roc_auc']:.4f} | "
            f"{m['positive_average_precision']:.4f} | {m['brier']:.4f} |"
        )
    lines += [
        "",
        "Means across five training seeds; these are not five independent datasets.",
        "Higher F1/AUC/AP is better; lower Brier is better. Positive AP must be interpreted "
        "against the high non-NEI prevalence. Class weighting does not yield "
        "calibrated probabilities.",
        result["scope"] + ".",
        result["verification_limit"] + ".",
        "No statistical superiority or external generalization is established here. "
        "Do not deploy this as an evidence-sufficiency gate without downstream evaluation.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json_atomic(
        output / "analysis_manifest.json",
        {
            "code_sha256": sha256(Path(__file__)),
            "outputs": {n: sha256(output / n) for n in ("verification.json", "RESULTS.md")},
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
