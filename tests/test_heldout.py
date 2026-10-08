import runpy
from pathlib import Path

import numpy as np
import pytest

from apv_rag import heldout
from apv_rag.evidence_baseline import LABELS
from apv_rag.splits import sha256, write_json_atomic


def test_overlap_audit_excludes_transitive_links_to_fitted_records():
    train = [
        {"claim": "A", "fact_checking_article": "https://x/a"},
        {"claim": "B", "fact_checking_article": "https://x/b"},
    ]
    dev = [
        {"claim": "a", "fact_checking_article": "https://x/c"},
        {"claim": "C", "fact_checking_article": "https://x/c"},
        {"claim": "B"},
        {"claim": ""},
    ]
    audit = heldout.overlap_audit(train, [0], dev)
    assert audit["independent_ids"] == [2]
    assert [r["dev_index"] for r in audit["excluded"]] == [0, 1, 3]
    assert audit["groups"]["0"] == audit["groups"]["1"]


def test_features_reject_wrong_shape_and_nan():
    with pytest.raises(ValueError):
        heldout.check_features(np.zeros((2, 6)), 2)
    with pytest.raises(ValueError):
        heldout.check_features(np.full((2, 7), np.nan), 2)


def test_empty_population_cannot_claim_confirmation():
    assert heldout.population_report([], [], {}, {}) == {
        "status": "unavailable: no independent claims",
        "claims": 0,
    }


def test_grouped_report_preserves_probabilities_and_pairing():
    protocol = {
        "methods": {"bm25_nli": {"decision_exponent": 1}, "dense_nli": {"decision_exponent": 1}},
        "seeds": [369, 1369],
        "training_priors": [0.1, 0.6, 0.15, 0.15],
        "statistics": {"seed": 369, "bootstrap_samples": 20, "randomization_samples": 20},
    }
    p = np.asarray(
        [[0.3, 0.4, 0.15, 0.15], [0.1, 0.7, 0.1, 0.1], [0.1, 0.4, 0.4, 0.1], [0.1, 0.4, 0.1, 0.4]]
    )
    values = {m: {s: p.copy() for s in protocol["seeds"]} for m in protocol["methods"]}
    report = heldout.population_report(
        list(range(4)), LABELS, values, protocol, {str(i): str(i // 2) for i in range(4)}
    )
    assert report["groups"] == 2
    assert all(r["mean_seed_macro_f1_gain"] > 0 for r in report["results"])
    for result in report["results"]:
        for row in result["seed_metrics"]:
            assert row["selected"]["multiclass_brier"] == row["argmax"]["multiclass_brier"]
    np.testing.assert_array_equal(values["bm25_nli"][369], p)


def test_cache_receipt_rejects_changed_bytes(tmp_path):
    path = tmp_path / "features.npy"
    path.write_bytes(b"original")
    heldout.cache_receipt(path, save=True)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache checksum"):
        heldout.cache_receipt(path)


def test_preflight_rejects_modified_code_before_dev_read(tmp_path):
    frozen = tmp_path / "artifacts/frozen_heldout_protocol"
    frozen.mkdir(parents=True)
    code = tmp_path / "code.py"
    code.write_text("original")
    protocol = {
        "code_sha256": {"code.py": sha256(code)},
        "decision_summary_sha256": "unused",
        "train_source_sha256": "unused",
        "train_indices_sha256": "unused",
    }
    write_json_atomic(frozen / "protocol.json", protocol)
    write_json_atomic(
        frozen / "protocol_checksum.json", {"protocol.json": sha256(frozen / "protocol.json")}
    )
    code.write_text("changed")
    with pytest.raises(ValueError, match="code.py"):
        heldout.preflight(tmp_path)


def test_output_validator_recomputes_metrics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    validate = runpy.run_path(str(root / "scripts/validate_heldout_evaluation.py"))["main"]
    protocol = {
        "methods": {"bm25_nli": {"decision_exponent": 1}, "dense_nli": {"decision_exponent": 1}},
        "seeds": [369],
        "training_priors": [0.25] * 4,
        "evaluation_source": {"count": 4},
        "statistics": {"seed": 369, "bootstrap_samples": 5, "randomization_samples": 5},
    }
    frozen = tmp_path / "artifacts/frozen_heldout_protocol"
    frozen.mkdir(parents=True)
    write_json_atomic(frozen / "protocol.json", protocol)
    output = tmp_path / "artifacts/heldout_excerpt_evaluation"
    output.mkdir()
    code_dir = tmp_path / "src/apv_rag"
    code_dir.mkdir(parents=True)
    for name in ("heldout.py", "splits.py"):
        (code_dir / name).write_bytes((root / "src/apv_rag" / name).read_bytes())
    write_json_atomic(
        output / "input_manifest.json",
        {
            "protocol_sha256": sha256(frozen / "protocol.json"),
            "evaluator_sha256": sha256(code_dir / "heldout.py"),
            "grouping_sha256": sha256(code_dir / "splits.py"),
        },
    )
    p = np.eye(4)
    groups = {str(i): str(i) for i in range(4)}
    values = {m: {369: p} for m in protocol["methods"]}
    report = heldout.population_report(list(range(4)), LABELS, values, protocol, groups)
    write_json_atomic(
        output / "heldout_summary.json",
        {"primary_independent": report, "secondary_all_dev": report},
    )
    write_json_atomic(
        output / "overlap_audit.json",
        {"independent_ids": list(range(4)), "excluded": [], "groups": groups},
    )
    rows = [
        {
            "method": m,
            "seed": 369,
            "dev_index": i,
            "true_label": LABELS[i],
            "argmax_label": LABELS[i],
            "selected_label": LABELS[i],
            "probabilities": p[i].tolist(),
        }
        for m in protocol["methods"]
        for i in range(4)
    ]
    write_json_atomic(output / "predictions.json", rows)
    for m in protocol["methods"]:
        np.save(output / f"dev_{m}_features.npy", np.zeros((4, 7)))

    def manifest():
        write_json_atomic(
            output / "output_manifest.json",
            {
                path.name: sha256(path)
                for path in output.iterdir()
                if path.name != "output_manifest.json"
            },
        )

    manifest()
    assert validate(tmp_path) == 0
    report["results"][0]["seed_metrics"][0]["argmax"]["macro_f1"] = 0.123
    write_json_atomic(
        output / "heldout_summary.json",
        {"primary_independent": report, "secondary_all_dev": report},
    )
    manifest()
    with pytest.raises(ValueError, match="metrics differ"):
        validate(tmp_path)
