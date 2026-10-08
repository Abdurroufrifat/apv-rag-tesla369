"""Check completed held-out outputs without loading neural models."""

import json
from pathlib import Path

import numpy as np

from apv_rag.decision_rules import decision_indices
from apv_rag.evidence_baseline import LABELS
from apv_rag.metrics import classification_metrics
from apv_rag.splits import sha256


def main(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    output = root / "artifacts/heldout_excerpt_evaluation"
    hashes = json.loads((output / "output_manifest.json").read_text())
    required = {
        "input_manifest.json",
        "predictions.json",
        "overlap_audit.json",
        "heldout_summary.json",
        "dev_bm25_nli_features.npy",
        "dev_dense_nli_features.npy",
    }
    if not required.issubset(hashes):
        raise ValueError("incomplete output manifest")
    for name, expected in hashes.items():
        if Path(name).name != name or sha256(output / name) != expected:
            raise ValueError(f"output checksum mismatch: {name}")
    protocol_path = root / "artifacts/frozen_heldout_protocol/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    identity = json.loads((output / "input_manifest.json").read_text())
    if identity["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("protocol identity mismatch")
    if identity["evaluator_sha256"] != sha256(root / "src/apv_rag/heldout.py"):
        raise ValueError("evaluator identity mismatch")
    if identity["grouping_sha256"] != sha256(root / "src/apv_rag/splits.py"):
        raise ValueError("grouping identity mismatch")
    rows = json.loads((output / "predictions.json").read_text())
    count = protocol["evaluation_source"]["count"]
    expected_keys = {
        (m, s, i) for m in protocol["methods"] for s in protocol["seeds"] for i in range(count)
    }
    if (
        len(rows) != len(expected_keys)
        or {(r["method"], r["seed"], r["dev_index"]) for r in rows} != expected_keys
    ):
        raise ValueError("prediction pairing incomplete or duplicated")
    truths = {}
    for row in rows:
        i = row["dev_index"]
        if (
            row["true_label"] not in LABELS
            or truths.setdefault(i, row["true_label"]) != row["true_label"]
        ):
            raise ValueError("inconsistent true labels")
        p = np.asarray([row["probabilities"]])
        selected = decision_indices(
            p, protocol["training_priors"], protocol["methods"][row["method"]]["decision_exponent"]
        )
        if (
            row["selected_label"] != LABELS[int(selected[0])]
            or row["argmax_label"] != LABELS[int(p.argmax())]
        ):
            raise ValueError("saved decisions differ from frozen rules")
    audit = json.loads((output / "overlap_audit.json").read_text())
    independent = audit["independent_ids"]
    excluded = [r["dev_index"] for r in audit["excluded"]]
    if set(independent) & set(excluded) or sorted(independent + excluded) != list(range(count)):
        raise ValueError("overlap audit does not partition official dev")
    summary = json.loads((output / "heldout_summary.json").read_text())
    if (
        summary["primary_independent"]["claims"] != len(independent)
        or summary["secondary_all_dev"]["claims"] != count
    ):
        raise ValueError("report population counts differ from audit")
    lookup = {(r["method"], r["seed"], r["dev_index"]): r for r in rows}
    for population, ids in (
        ("primary_independent", independent),
        ("secondary_all_dev", list(range(count))),
    ):
        if not ids:
            if summary[population]["status"] != "unavailable: no independent claims":
                raise ValueError("empty population incorrectly reported as evaluated")
            continue
        results = summary[population]["results"]
        if len(results) != len(protocol["methods"]) or {r["method"] for r in results} != set(
            protocol["methods"]
        ):
            raise ValueError("report method coverage mismatch")
        for result in results:
            gains = []
            metrics = result["seed_metrics"]
            if len(metrics) != len(protocol["seeds"]) or {r["seed"] for r in metrics} != set(
                protocol["seeds"]
            ):
                raise ValueError("report seed coverage mismatch")
            for row in metrics:
                saved = [lookup[(result["method"], row["seed"], i)] for i in ids]
                truth = [r["true_label"] for r in saved]
                p = np.asarray([r["probabilities"] for r in saved])
                calculated = {}
                for rule, key in (("argmax", "argmax_label"), ("selected", "selected_label")):
                    calculated[rule] = classification_metrics(
                        truth,
                        [r[key] for r in saved],
                        p,
                        LABELS,
                        ece_bins=15,
                        coverages=[0.5, 0.8, 1],
                    )
                calculated["selected"]["expected_calibration_error"] = calculated["argmax"][
                    "expected_calibration_error"
                ]
                calculated["selected"].pop("risk_coverage")
                if any(calculated[rule] != row[rule] for rule in calculated):
                    raise ValueError("report metrics differ from predictions")
                gains.append(calculated["selected"]["macro_f1"] - calculated["argmax"]["macro_f1"])
            if not np.isclose(np.mean(gains), result["mean_seed_macro_f1_gain"]):
                raise ValueError("report effect differs from predictions")
    print("Held-out output integrity and frozen decision checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
