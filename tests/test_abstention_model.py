from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apv_rag.abstention_model import (
    abstention_scores,
    bootstrap_aurc_intervals,
    risk_coverage_curve,
    run_phase2g,
    select_abstention_score,
    threshold_policy,
    validate_phase2g_run,
)
from apv_rag.evidence_baseline import LABELS
from apv_rag.splits import sha256, write_json_atomic


def test_abstention_scores_use_maximum_margin_and_normalized_entropy():
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.2, 0.1, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    scores = abstention_scores(probabilities)

    np.testing.assert_allclose(scores["maximum_probability"], [1.0, 0.7, 0.25])
    np.testing.assert_allclose(scores["probability_margin"], [1.0, 0.5, 0.0])
    assert scores["normalized_entropy"][0] == 1.0
    assert scores["normalized_entropy"][2] == 0.0


def test_risk_coverage_curve_orders_highest_score_first_and_computes_aurc():
    curve = risk_coverage_curve(
        ["Supported", "Supported", "Refuted"],
        ["Supported", "Refuted", "Refuted"],
        np.asarray([0.9, 0.8, 0.1]),
    )

    np.testing.assert_allclose([row["risk"] for row in curve["points"]], [0.0, 0.5, 1 / 3])
    assert curve["aurc"] == np.mean([0.0, 0.5, 1 / 3])


def test_score_selection_uses_lowest_aurc_then_configured_order():
    candidates = {
        "maximum_probability": {"aurc": 0.20},
        "probability_margin": {"aurc": 0.15},
        "normalized_entropy": {"aurc": 0.15},
    }

    assert select_abstention_score(candidates)["name"] == "probability_margin"


def test_threshold_policy_retains_requested_top_coverage_and_marks_abstentions():
    policy = threshold_policy(
        ["Supported", "Refuted", "Supported"],
        ["Supported", "Refuted", "Refuted"],
        np.asarray([0.9, 0.2, 0.8]),
        [2 / 3, 1.0],
    )

    two_thirds = policy["0.666667"]
    assert two_thirds["threshold"] == 0.8
    assert two_thirds["decisions"] == ["machine_candidate", "abstain", "machine_candidate"]
    assert two_thirds["selective_accuracy"] == 0.5
    assert policy["1.000000"]["abstained"] == 0


def test_bootstrap_intervals_are_seeded_and_contain_the_point_estimate():
    truth = ["Supported", "Supported", "Refuted", "Refuted"]
    predicted = ["Supported", "Refuted", "Refuted", "Refuted"]
    scores = {
        "maximum_probability": np.asarray([0.9, 0.6, 0.8, 0.7]),
        "probability_margin": np.asarray([0.8, 0.2, 0.7, 0.5]),
    }

    first = bootstrap_aurc_intervals(truth, predicted, scores, repetitions=50, seed=369)
    second = bootstrap_aurc_intervals(truth, predicted, scores, repetitions=50, seed=369)

    assert first == second
    for name, interval in first.items():
        point = risk_coverage_curve(truth, predicted, scores[name])["aurc"]
        assert interval["lower"] <= point <= interval["upper"]


def test_phase2g_artifact_is_self_contained_and_detects_tampering(tmp_path: Path):
    config_dir = tmp_path / "config"
    source_dir = tmp_path / "artifacts/phase2f_provenance_model/source"
    config_dir.mkdir()
    source_dir.mkdir(parents=True)
    (config_dir / "abstention_model.yaml").write_text(
        """protocol_version: '0.1'
random_seed: 369
bootstrap_repetitions: 50
target_coverages: [0.5, 1.0]
output_directory: artifacts/phase2g_abstention_model
""",
        encoding="utf-8",
    )
    rows = []
    for index in range(12):
        truth = LABELS[index % len(LABELS)]
        probabilities = {label: 0.1 for label in LABELS}
        probabilities[truth] = 0.7
        rows.append(
            {
                "upstream_index": index,
                "true_label": truth,
                "predicted_label": truth,
                "probabilities": probabilities,
                "confidence": 0.7,
            }
        )
    prediction_path = source_dir / "validation_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    write_json_atomic(
        source_dir / "run_manifest.json",
        {
            "official_dev_records_used": 0,
            "outputs": {
                "validation_predictions.jsonl": {
                    "sha256": sha256(prediction_path),
                    "bytes": prediction_path.stat().st_size,
                }
            },
        },
    )

    run_dir = run_phase2g(tmp_path, source_run_dir=source_dir, run_id="synthetic")
    prediction_path.write_text("tampered", encoding="utf-8")

    assert (run_dir / "input_snapshot/validation_predictions.jsonl").is_file()
    assert validate_phase2g_run(run_dir) == []
    (run_dir / "selective_predictions.jsonl").write_text("tampered", encoding="utf-8")
    assert "output integrity failure: selective_predictions.jsonl" in validate_phase2g_run(run_dir)
