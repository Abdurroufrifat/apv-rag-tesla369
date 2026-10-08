"""Verify archived SciFact predictions and run the planned grouped comparisons."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from apv_rag.decision_rules import decision_indices
from apv_rag.evidence_baseline import LABELS
from apv_rag.nli_comparison import SEEDS
from apv_rag.paired_statistics import holm_adjust
from apv_rag.scifact import connected_groups, target_label
from apv_rag.splits import sha256, write_json_atomic


def target_score(truth, pred):
    matrix = np.bincount(truth * 4 + pred, minlength=16).reshape(4, 4)
    denom = matrix.sum(0)[:3] + matrix.sum(1)[:3]
    return float(
        np.divide(2 * matrix.diagonal()[:3], denom, out=np.zeros(3), where=denom != 0).mean()
    )


def check(a, b):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("Metric keys differ")
        for key in a:
            check(a[key], b[key])
    elif not np.isclose(a, b, atol=1e-12, rtol=1e-10):
        raise ValueError(f"Metric mismatch: {a}, {b}")


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = root / "artifacts/scifact_received"
    manifest = json.loads((inputs / "output_manifest.json").read_text())
    for name in ("predictions.json", "transfer_summary.json", "input_manifest.json"):
        if sha256(inputs / name) != manifest[name]:
            raise ValueError(f"Upload checksum mismatch: {name}")
    summary = json.loads((inputs / "transfer_summary.json").read_text())
    identity = json.loads((inputs / "input_manifest.json").read_text())
    for name, digest in identity["target_inputs"].items():
        if sha256(root / "data/external/scifact/sealed_v1" / name) != digest:
            raise ValueError("Target data mismatch")
    if sha256(root / "scripts/run_scifact_transfer.py") != identity["runner_sha256"]:
        raise ValueError("Runner differs from recorded version")
    for name, digest in identity["code_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Code mismatch: {name}")
    if sha256(root / "docs/SCIFACT_TRANSFER_PROTOCOL.md") != identity["protocol_sha256"]:
        raise ValueError("Transfer design mismatch")
    if identity["seeds"] != list(SEEDS) or identity["exponent"] != 1:
        raise ValueError("Fixed settings changed")
    claims = [
        json.loads(line)
        for line in (root / "data/external/scifact/sealed_v1/claims_dev.jsonl")
        .read_text()
        .splitlines()
    ]
    source = root / "data/external/averitec/official_7c62d1e"
    from apv_rag.scifact import normalized_claim

    observed = json.loads((source / "train.json").read_text()) + json.loads(
        (source / "dev.json").read_text()
    )
    seen = {normalized_claim(r["claim"]) for r in observed}
    overlap = [r["id"] for r in claims if normalized_claim(r["claim"]) in seen]
    remaining = [r for r in claims if r["id"] not in set(overlap)]
    conflicting = [r["id"] for r in remaining if target_label(r) is None]
    evaluated = [r for r in remaining if target_label(r) is not None]
    if (
        overlap != summary["exact_overlap_excluded"]
        or conflicting != summary["conflicting_labels_excluded"]
    ):
        raise ValueError("Exclusion audit mismatch")
    groups = np.asarray(connected_groups(evaluated))
    ids = [r["id"] for r in evaluated]
    labels = [target_label(r) for r in evaluated]
    mapping = {label: i for i, label in enumerate(LABELS)}
    truth = np.array([mapping[label] for label in labels])
    rows = json.loads((inputs / "predictions.json").read_text())
    lookup = {(r["representation"], r["seed"], r["rule"], r["claim_id"]): r for r in rows}
    expected = {
        (rep, seed, rule, i)
        for rep in ("base_7", "expanded_18")
        for seed in SEEDS
        for rule in ("argmax", "prior_adjusted")
        for i in ids
    }
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Incomplete or duplicate predictions")
    if summary["claims"] != len(ids) or summary["groups"] != len(set(groups)):
        raise ValueError("Population counts differ")
    matrices, table = {}, []
    for rep in ("base_7", "expanded_18"):
        for rule, alpha in [("argmax", 0), ("prior_adjusted", 1)]:
            matrices[rep, rule] = []
            scores = []
            for seed in SEEDS:
                selected = [lookup[rep, seed, rule, i] for i in ids]
                if [r["true_label"] for r in selected] != labels or [
                    r["group"] for r in selected
                ] != groups.tolist():
                    raise ValueError("Truth or group alignment differs")
                p = np.asarray([r["probabilities"] for r in selected])
                if (
                    p.shape != (len(ids), 4)
                    or not np.isfinite(p).all()
                    or (p < 0).any()
                    or (p > 1).any()
                    or not np.allclose(p.sum(1), 1)
                ):
                    raise ValueError("Invalid probability rows")
                pred = np.array([mapping[r["predicted_label"]] for r in selected])
                if not np.array_equal(
                    pred, decision_indices(p, identity["training_priors"], alpha)
                ):
                    raise ValueError("Decision rule mismatch")
                other = [
                    lookup[rep, seed, "prior_adjusted" if rule == "argmax" else "argmax", i][
                        "probabilities"
                    ]
                    for i in ids
                ]
                if not np.array_equal(p, other):
                    raise ValueError("Rules use different probabilities")
                report = next(
                    r
                    for r in summary["results"]
                    if r["representation"] == rep and r["rule"] == rule and r["seed"] == seed
                )
                pred_labels = np.array(LABELS)[pred]
                check(
                    report["macro_f1"],
                    f1_score(
                        labels,
                        pred_labels,
                        labels=list(LABELS[:3]),
                        average="macro",
                        zero_division=0,
                    ),
                )
                check(report["accuracy"], accuracy_score(labels, pred_labels))
                check(report["out_of_target_rate"], float(np.mean(pred == 3)))
                check(
                    report["per_class"],
                    classification_report(
                        labels,
                        pred_labels,
                        labels=list(LABELS[:3]),
                        output_dict=True,
                        zero_division=0,
                    ),
                )
                matrices[rep, rule].append(pred)
                scores.append(report["macro_f1"])
            matrices[rep, rule] = np.asarray(matrices[rep, rule])
            table.append(
                {"representation": rep, "rule": rule, "mean_macro_f1": float(np.mean(scores))}
            )
    names = sorted(set(groups))
    positions = [np.flatnonzero(groups == g) for g in names]
    membership = np.array([names.index(g) for g in groups])
    results = []
    for rule in ("argmax", "prior_adjusted"):
        a, b = matrices["base_7", rule], matrices["expanded_18", rule]

        def difference(y, left, right):
            return float(
                np.mean(
                    [
                        target_score(y, after) - target_score(y, before)
                        for before, after in zip(left, right, strict=True)
                    ]
                )
            )

        observed = difference(truth, a, b)
        rng = np.random.default_rng(369)
        bootstrap = []
        for _ in range(2000):
            ix = np.concatenate([positions[g] for g in rng.integers(len(names), size=len(names))])
            bootstrap.append(difference(truth[ix], a[:, ix], b[:, ix]))
        extreme = 0
        for _ in range(10000):
            swap = rng.integers(2, size=len(names)).astype(bool)[membership][None, :]
            effect = difference(truth, np.where(swap, b, a), np.where(swap, a, b))
            extreme += abs(effect) >= abs(observed) - 1e-12
        results.append(
            {
                "rule": rule,
                "gain": observed,
                "group_bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                "randomization_two_sided_p": (extreme + 1) / 10001,
            }
        )
    for r, p in zip(
        results, holm_adjust([r["randomization_two_sided_p"] for r in results]), strict=True
    ):
        r["holm_adjusted_p_two_comparisons"] = p
    output = root / "artifacts/scifact_statistics"
    output.mkdir(exist_ok=True)
    write_json_atomic(
        output / "statistics.json",
        {
            "claims": len(ids),
            "groups": len(names),
            "verified_prediction_rows": len(rows),
            "mean_scores": table,
            "results": results,
            "bootstrap_samples": 2000,
            "randomization_samples": 10000,
            "seed": 369,
            "verification_limit": (
                "summary metrics and decisions verified; "
                "target inference and training fits not rerun"
            ),
            "input_sha256": {
                n: sha256(inputs / n)
                for n in ("predictions.json", "transfer_summary.json", "input_manifest.json")
            },
        },
    )
    lines = [
        "# SciFact transfer results",
        "",
        (
            f"{len(ids)} claims; {len(names)} document-connected groups. "
            f"All {len(rows)} prediction rows and reported metrics verified."
        ),
        "",
        "| Features | Rule | Mean target macro-F1 |",
        "|---|---|---:|",
    ]
    lines += [f"| {r['representation']} | {r['rule']} | {r['mean_macro_f1']:.4f} |" for r in table]
    lines += [
        "",
        "| Rule | Expanded minus base | Marginal 95% group interval | Holm p |",
        "|---|---:|---|---:|",
    ]
    lines += [
        (
            f"| {r['rule']} | {r['gain']:+.4f} | "
            f"[{r['group_bootstrap_95_interval'][0]:+.4f}, "
            f"{r['group_bootstrap_95_interval'][1]:+.4f}] | "
            f"{r['holm_adjusted_p_two_comparisons']:.4f} |"
        )
        for r in results
    ]
    lines += [
        "",
        (
            "This is classifier transfer with BM25 abstract retrieval, not the official "
            "joint SciFact metric. Source-host features identify the dataset host. "
            "Abstract premises are truncated at 256 NLI pair tokens. No target fitting "
            "was performed. Metrics and decisions were reproduced from exports; training "
            "fits and target NLI inference were not rerun."
        ),
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
