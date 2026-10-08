from __future__ import annotations

import json
from pathlib import Path

import pytest

from apv_rag.evidence_baseline import (
    assert_prediction_alignment,
    load_split_records,
    render_claim_only,
    render_claim_plus_evidence,
    run_baseline,
    select_setting,
    validate_run,
)
from apv_rag.splits import sha256, write_json_atomic


def sample_record() -> dict:
    return {
        "claim": "Claim text",
        "label": "Supported",
        "justification": "FORBIDDEN JUSTIFICATION",
        "speaker": "FORBIDDEN SPEAKER",
        "reporting_source": "FORBIDDEN REPORTER",
        "fact_checking_article": "https://forbidden.example/fact-check",
        "questions": [
            {
                "question": "Evidence question?",
                "answers": [
                    {
                        "answer_type": "Extractive",
                        "answer": "Evidence answer",
                        "source_medium": "Web text",
                        "source_url": "https://forbidden.example/source",
                    }
                ],
            }
        ],
    }


def test_claim_only_returns_the_claim_without_label():
    text = render_claim_only(sample_record())

    assert text == "Claim text"
    assert "Supported" not in text


def test_claim_plus_evidence_keeps_permitted_evidence_fields_only():
    text = render_claim_plus_evidence(sample_record())

    assert text == (
        "CLAIM: Claim text\n"
        "QUESTION: Evidence question?\n"
        "ANSWER_TYPE: Extractive\n"
        "ANSWER: Evidence answer\n"
        "SOURCE_MEDIUM: Web text"
    )
    for forbidden in (
        "Supported",
        "FORBIDDEN JUSTIFICATION",
        "FORBIDDEN SPEAKER",
        "FORBIDDEN REPORTER",
        "https://forbidden.example",
    ):
        assert forbidden not in text


def test_claim_plus_evidence_keeps_answer_when_upstream_question_is_blank():
    record = sample_record()
    record["questions"][0]["question"] = ""

    text = render_claim_plus_evidence(record)

    assert "QUESTION:" not in text
    assert "ANSWER: Evidence answer" in text


def test_load_split_records_rejects_duplicate_indices():
    records = [sample_record(), {**sample_record(), "claim": "Second claim"}]

    with pytest.raises(ValueError, match="unique"):
        load_split_records(records, [0, 0])


def test_load_split_records_rejects_empty_claim():
    records = [{**sample_record(), "claim": "   "}]

    with pytest.raises(ValueError, match="non-empty claim"):
        load_split_records(records, [0])


def test_select_setting_uses_macro_f1_then_lower_c():
    candidates = [
        {"variant": "claim_plus_evidence", "c": 4.0, "metrics": {"macro_f1": 0.61}},
        {"variant": "claim_plus_evidence", "c": 0.25, "metrics": {"macro_f1": 0.61}},
    ]

    assert select_setting(candidates)["c"] == 0.25


def test_prediction_indices_must_match_validation_indices():
    with pytest.raises(ValueError, match="prediction indices"):
        assert_prediction_alignment([2, 4], [{"upstream_index": 2}])


def test_run_baseline_writes_aligned_machine_only_artifacts(tmp_path: Path):
    project = tmp_path
    source_dir = project / "data/external/averitec/official_7c62d1e"
    split_dir = project / "data/processed/averitec/phase2b_split_v0_1"
    config_dir = project / "config"
    source_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    config_dir.mkdir()
    labels = [
        "Supported",
        "Refuted",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
    ]
    records = []
    for label_index, label in enumerate(labels):
        for item in range(8):
            token = f"labeltoken{label_index}"
            records.append(
                {
                    **sample_record(),
                    "claim": f"{token} claim {item}",
                    "label": label,
                    "questions": [
                        {
                            "question": f"What evidence for {token}?",
                            "answers": [
                                {
                                    "answer_type": "Extractive",
                                    "answer": f"Evidence says {token}",
                                    "source_medium": "Web text",
                                    "source_url": "https://excluded.example",
                                }
                            ],
                        }
                    ],
                }
            )
    source_path = source_dir / "train.json"
    source_path.write_text(json.dumps(records), encoding="utf-8")
    train_indices = [i for i in range(len(records)) if i % 8 < 6]
    validation_indices = [i for i in range(len(records)) if i % 8 >= 6]
    write_json_atomic(split_dir / "train_indices.json", train_indices)
    write_json_atomic(split_dir / "validation_indices.json", validation_indices)
    write_json_atomic(
        split_dir / "split_manifest.json",
        {
            "source": {"path": "data/external/averitec/official_7c62d1e/train.json"},
            "counts": {"train": 24, "validation": 8},
        },
    )
    (config_dir / "evidence_baseline.yaml").write_text(
        """protocol_version: '0.1'
random_seed: 369
variants: [claim_only, claim_plus_evidence]
regularization_c: [0.25]
tfidf: {ngram_range: [1, 2], min_df: 1, max_features: 5000}
calibration_cv: 3
ece_bins: 5
target_coverages: [0.5, 1.0]
output_directory: artifacts/phase2c_evidence_baseline
""",
        encoding="utf-8",
    )

    run_dir = run_baseline(
        project,
        run_id="synthetic",
        expected_source_sha256=sha256(source_path),
    )

    assert {path.name for path in run_dir.iterdir()} == {
        "metrics.json",
        "model_selection.json",
        "run_manifest.json",
        "validation_predictions.jsonl",
    }
    predictions = [
        json.loads(line)
        for line in (run_dir / "validation_predictions.jsonl").read_text().splitlines()
    ]
    assert [row["upstream_index"] for row in predictions] == validation_indices
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["official_dev_records_used"] == 0
    assert "dev.json" not in json.dumps(manifest)
    assert validate_run(run_dir) == []


def test_validator_rejects_prediction_index_mismatch(tmp_path: Path):
    project = tmp_path
    run_dir = project / "artifacts/phase2c_evidence_baseline/test-run"
    split_dir = project / "data/processed/averitec/phase2b_split_v0_1"
    run_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    write_json_atomic(split_dir / "validation_indices.json", [2, 4])
    prediction_path = run_dir / "validation_predictions.jsonl"
    prediction_path.write_text(
        json.dumps(
            {
                "upstream_index": 2,
                "true_label": "Supported",
                "predicted_label": "Supported",
                "probabilities": {
                    "Supported": 1.0,
                    "Refuted": 0.0,
                    "Not Enough Evidence": 0.0,
                    "Conflicting Evidence/Cherrypicking": 0.0,
                },
                "confidence": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json_atomic(run_dir / "metrics.json", {})
    write_json_atomic(run_dir / "model_selection.json", {})
    write_json_atomic(
        run_dir / "run_manifest.json",
        {
            "labels": [
                "Supported",
                "Refuted",
                "Not Enough Evidence",
                "Conflicting Evidence/Cherrypicking",
            ],
            "official_dev_records_used": 0,
            "inputs": {},
            "outputs": {
                "validation_predictions.jsonl": {"sha256": sha256(prediction_path)},
                "metrics.json": {"sha256": sha256(run_dir / "metrics.json")},
                "model_selection.json": {
                    "sha256": sha256(run_dir / "model_selection.json")
                },
            },
        },
    )

    errors = validate_run(run_dir)

    assert "prediction indices do not match validation indices" in errors


def test_validator_rejects_changed_input_file(tmp_path: Path):
    project = tmp_path
    run_dir = project / "artifacts/phase2c_evidence_baseline/test-run"
    config_dir = project / "config"
    run_dir.mkdir(parents=True)
    config_dir.mkdir()
    config_path = config_dir / "evidence_baseline.yaml"
    config_path.write_text("changed: true\n", encoding="utf-8")
    write_json_atomic(
        run_dir / "run_manifest.json",
        {
            "labels": list(
                (
                    "Supported",
                    "Refuted",
                    "Not Enough Evidence",
                    "Conflicting Evidence/Cherrypicking",
                )
            ),
            "official_dev_records_used": 0,
            "inputs": {
                "evidence_baseline.yaml": {
                    "path": "config/evidence_baseline.yaml",
                    "sha256": "0" * 64,
                }
            },
            "outputs": {},
        },
    )

    errors = validate_run(run_dir)

    assert "input SHA-256 mismatch: evidence_baseline.yaml" in errors
