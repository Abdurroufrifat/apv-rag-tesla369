"""Build manuscript tables and static figures from archived experiment summaries."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/results_package"


def main():
    OUT.mkdir(exist_ok=True)
    sources = {}

    def read(name):
        p = ROOT / "artifacts" / name
        sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())

    internal = read("retrieved_feature_received/retrieved_feature_summary.json")
    held = read("heldout_received/heldout_summary.json")["primary_independent"]
    statistics = read("retrieved_feature_statistics/statistics.json")
    ablation = read("ablation_received/ablation_summary.json")
    selection = read("expanded_training_rule_received/selection_summary.json")
    tables = {}
    groups = {}
    for r in internal["results"]:
        key = (r["method"], r["representation"], r["rule"])
        groups.setdefault(key, []).append(r)
    rows = []
    for (method, representation, rule), records in sorted(groups.items()):
        assert len(records) == 5 and len({r["seed"] for r in records}) == 5
        rows.append(
            dict(
                method=method,
                representation=representation,
                rule=rule,
                macro_f1=np.mean([r["metrics"]["macro_f1"] for r in records]),
                brier=np.mean([r["metrics"]["multiclass_brier"] for r in records]),
            )
        )
    tables["internal_validation"] = rows
    tables["heldout"] = [
        dict(
            method=r["method"],
            argmax_macro_f1=np.mean([x["argmax"]["macro_f1"] for x in r["seed_metrics"]]),
            selected_macro_f1=np.mean([x["selected"]["macro_f1"] for x in r["seed_metrics"]]),
            gain=r["mean_seed_macro_f1_gain"],
            ci_low=r["group_bootstrap_95_interval"][0],
            ci_high=r["group_bootstrap_95_interval"][1],
            holm_p=r["holm_adjusted_p"],
        )
        for r in held["results"]
    ]
    tables["expanded_feature_statistics"] = [
        dict(
            method=r["method"],
            rule=r["rule"],
            gain=r["mean_seed_macro_f1_gain"],
            ci_low=r["group_bootstrap_95_interval"][0],
            ci_high=r["group_bootstrap_95_interval"][1],
            holm_p=r["holm_adjusted_p_four_comparisons"],
        )
        for r in statistics["results"]
    ]
    variants = sorted({r["variant"] for r in ablation["results"]})
    tables["own_excerpt_ablations"] = [
        dict(
            variant=v,
            rule=rule,
            macro_f1=np.mean(
                [
                    r["metrics"]["macro_f1"]
                    for r in ablation["results"]
                    if r["variant"] == v and r["rule"] == rule
                ]
            ),
        )
        for v in variants
        for rule in sorted({r["rule"] for r in ablation["results"]})
    ]
    tables["training_rule_selection"] = [
        dict(
            method=m,
            exponent=r["selected_exponent"],
            training_oof_macro_f1=r["mean_training_oof_macro_f1"][str(r["selected_exponent"])],
        )
        for m, r in selection["results"].items()
    ]
    report = [
        "# Verified experiment results",
        "",
        (
            "Macro-F1 values are means over five seeds, not pooled predictions. Internal "
            "validation contains 609 claims in 375 connected groups. Primary held-out "
            "evaluation contains 461 claims in 367 groups; 39 train-linked official "
            "development claims were excluded."
        ),
        "",
        (
            "Expanded features and ablations were developed after earlier results were "
            "observed. Their internal validation results are exploratory. The held-out "
            "table concerns the earlier seven-feature model and prior adjustment, not the"
            " expanded model. Neither held-out gain is significant after Holm correction."
        ),
        "",
    ]
    for name, data in tables.items():
        with (OUT / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        keys = list(data[0])
        report.extend(
            [
                f"## {name.replace('_', ' ')}",
                "",
                "| " + " | ".join(keys) + " |",
                "| " + " | ".join(["---"] * len(keys)) + " |",
            ]
        )
        for row in data:
            report.append(
                "| "
                + " | ".join(
                    f"{row[k]:.4f}" if isinstance(row[k], (float, np.floating)) else str(row[k])
                    for k in keys
                )
                + " |"
            )
        report.append("")
    report.extend(
        [
            "## Interpretation limits",
            "",
            (
                "Retrieval uses benchmark-annotated answer excerpts in split-local corpora. "
                "These are not open-web retrieval results or official AVeriTeC scores. Source"
                " and excerpt-length features are proxies; they do not establish evidence "
                "sufficiency. Inner calibration uses stratified folds. Training rule "
                "selection uses grouped outer folds with fixed cached retrieval features."
            ),
            "",
            (
                "Bootstrap intervals are marginal 95% connected-group intervals. Internal "
                "statistics use Holm correction over four comparisons; this does not cover "
                "all earlier adaptive development searches. Held-out statistics use the "
                "original protocol correction. No new independent benchmark evaluation is "
                "included."
            ),
            "",
            (
                "Figures: internal_validation.png/pdf compares seven and eighteen features. "
                "heldout_gain.png/pdf shows the earlier selected-rule gain over argmax. "
                "expanded_feature_gain.png/pdf shows eighteen-minus-seven-feature internal "
                "gains. All plotted values are also in the CSV tables."
            ),
        ]
    )
    (OUT / "RESULTS.md").write_text("\n".join(report) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    reps = sorted({r["representation"] for r in rows})
    rules = sorted({r["rule"] for r in rows})
    for ax, method in zip(axes, ["bm25_nli", "dense_nli"], strict=True):
        for j, rep in enumerate(reps):
            vals = [
                next(
                    r["macro_f1"]
                    for r in rows
                    if r["method"] == method and r["representation"] == rep and r["rule"] == rule
                )
                for rule in rules
            ]
            ax.bar(
                np.arange(len(rules)) + (j - 0.5) * 0.35,
                vals,
                0.35,
                label=rep,
                color=["#376795", "#b97b29"][j],
                hatch=["", "//"][j],
            )
        ax.set_xticks(range(len(rules)), rules)
        ax.set_ylim(0, 0.45)
        ax.set_title(method.replace("_nli", "").upper())
        ax.set_ylabel("Mean macro-F1")
        ax.legend(frameon=False)
    fig.suptitle("Internal validation: 609 claims; exploratory")
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"internal_validation.{ext}", dpi=200)
    plt.close(fig)
    for name, title in [
        ("heldout", "Earlier seven-feature model: 461 independent claims"),
        ("expanded_feature_statistics", "Expanded minus seven features: internal validation"),
    ]:
        data = tables[name]
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        labels = [
            r["method"].replace("_nli", "").upper() + (" / " + r["rule"] if "rule" in r else "")
            for r in data
        ]
        x = np.array([r["gain"] for r in data])
        low = np.array([r["ci_low"] for r in data])
        high = np.array([r["ci_high"] for r in data])
        ax.errorbar(
            x, range(len(data)), xerr=[x - low, high - x], fmt="o", capsize=4, color="#376795"
        )
        ax.axvline(0, color="0.35", linestyle="--")
        ax.set_yticks(range(len(data)), labels)
        ax.invert_yaxis()
        ax.set_xlabel("Mean macro-F1 difference (marginal 95% group bootstrap interval)")
        ax.set_title(title)
        filename = "heldout_gain" if name == "heldout" else "expanded_feature_gain"
        for ext in ["png", "pdf"]:
            fig.savefig(OUT / f"{filename}.{ext}", dpi=200)
        plt.close(fig)
    sources["scripts/build_results_package.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (OUT / "source_manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
    (OUT / "output_manifest.json").write_text(
        json.dumps(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in OUT.iterdir()
                if p.name != "output_manifest.json"
            },
            indent=2,
        )
        + "\n"
    )
    print(OUT / "RESULTS.md")


if __name__ == "__main__":
    main()
