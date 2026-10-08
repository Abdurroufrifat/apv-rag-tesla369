from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from apv_rag.disagreement_model import (
    DISAGREEMENT_FEATURES,
    build_phase2e_pipeline,
    extract_disagreement_features,
    run_phase2e,
    select_phase2e_setting,
    validate_phase2e_run,
)
from apv_rag.splits import sha256, write_json_atomic


def _config() -> dict:
    return {
        "random_seed": 369,
        "calibration_cv": 2,
        "word_tfidf": {"ngram_range": [1, 2], "min_df": 1, "max_features": 1000},
        "char_tfidf": {"ngram_range": [3, 5], "min_df": 1, "max_features": 1000},
    }


def _record() -> dict:
    return {
        "claim": "A claim that must not enter the structural features",
        "label": "Conflicting Evidence/Cherrypicking",
        "justification": "forbidden label rationale",
        "speaker": "forbidden speaker",
        "fact_checking_article": "https://forbidden.example/article",
        "questions": [
            {
                "question": "Did it happen?",
                "answers": [
                    {
                        "answer": "Yes",
                        "answer_type": "Boolean",
                        "source_medium": "Web text",
                        "source_url": "https://one.example",
                    },
                    {
                        "answer": "No",
                        "answer_type": "Boolean",
                        "source_medium": "PDF",
                        "source_url": "https://two.example",
                    },
                ],
            },
            {
                "question": "What was reported?",
                "answers": [
                    {
                        "answer": "It did not happen",
                        "answer_type": "Extractive",
                        "source_medium": "Web text",
                        "source_url": "https://three.example",
                    }
                ],
            },
        ],
    }


def test_features_detect_boolean_and_negation_disagreement():
    values = dict(
        zip(DISAGREEMENT_FEATURES, extract_disagreement_features(_record()), strict=False)
    )

    assert values["question_count"] == 2.0
    assert values["answer_count"] == 3.0
    assert values["boolean_yes_no_mixed"] == 1.0
    assert values["negation_mixed"] == 1.0
    assert values["source_medium_diversity"] == 2.0 / 3.0
    assert values["multi_answer_question_fraction"] == 0.5


def test_structural_features_ignore_labels_metadata_and_urls():
    original = _record()
    changed = copy.deepcopy(original)
    changed["label"] = "Supported"
    changed["justification"] = "completely different"
    changed["speaker"] = "different"
    changed["fact_checking_article"] = "https://different.example"
    for question in changed["questions"]:
        for answer in question["answers"]:
            answer["source_url"] = "https://changed.example"

    np.testing.assert_allclose(
        extract_disagreement_features(original),
        extract_disagreement_features(changed),
    )


def test_selection_uses_conflict_f1_after_macro_f1():
    conflict = "Conflicting Evidence/Cherrypicking"
    candidates = [
        {
            "representation": "text_only",
            "c": 1.0,
            "metrics": {
                "macro_f1": 0.5,
                "balanced_accuracy": 0.6,
                "per_class": {conflict: {"f1": 0.1}},
            },
        },
        {
            "representation": "text_plus_disagreement",
            "c": 1.0,
            "metrics": {
                "macro_f1": 0.5,
                "balanced_accuracy": 0.55,
                "per_class": {conflict: {"f1": 0.3}},
            },
        },
    ]

    assert select_phase2e_setting(candidates)["representation"] == ("text_plus_disagreement")


def test_combined_pipeline_contains_text_and_structural_branches():
    pipeline = build_phase2e_pipeline(_config(), "text_plus_disagreement", 1.0)

    features = pipeline.named_steps["features"]
    assert [name for name, _ in features.transformer_list] == ["text", "disagreement"]
    assert pipeline.named_steps["classifier"].estimator.class_weight == "balanced"


def test_phase2e_run_preserves_validation_indices_and_validates(tmp_path: Path):
    labels = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    records = []
    for label_index, label in enumerate(labels):
        for item in range(6):
            record = _record()
            record["claim"] = f"labeltoken{label_index} claim {item}"
            record["label"] = label
            records.append(record)
    source_dir = tmp_path / "data/external/averitec/official_7c62d1e"
    split_dir = tmp_path / "data/processed/averitec/phase2b_split_v0_1"
    config_dir = tmp_path / "config"
    source_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    config_dir.mkdir()
    source = source_dir / "train.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    train_indices = [index for index in range(24) if index % 6 < 4]
    validation_indices = [index for index in range(24) if index % 6 >= 4]
    write_json_atomic(split_dir / "train_indices.json", train_indices)
    write_json_atomic(split_dir / "validation_indices.json", validation_indices)
    write_json_atomic(
        split_dir / "split_manifest.json",
        {"source": {"path": "data\\external\\averitec\\official_7c62d1e\\train.json"}},
    )
    (config_dir / "disagreement_model.yaml").write_text(
        """protocol_version: '0.1'
random_seed: 369
representations: [text_only, disagreement_only, text_plus_disagreement]
regularization_c: [1.0]
word_tfidf: {ngram_range: [1, 2], min_df: 1, max_features: 1000}
char_tfidf: {ngram_range: [3, 5], min_df: 1, max_features: 1000}
calibration_cv: 2
ece_bins: 5
target_coverages: [0.5, 1.0]
output_directory: artifacts/phase2e_disagreement_model
""",
        encoding="utf-8",
    )

    run_dir = run_phase2e(tmp_path, run_id="synthetic", expected_source_sha256=sha256(source))

    predictions = [
        json.loads(line)
        for line in (run_dir / "validation_predictions.jsonl").read_text().splitlines()
    ]
    assert [row["upstream_index"] for row in predictions] == validation_indices
    assert validate_phase2e_run(run_dir) == []

    manifest_path = split_dir / "split_manifest.json"
    split_manifest = json.loads(manifest_path.read_text())
    split_manifest["source"]["path"] = "data/external/averitec/official_7c62d1e/train.json"
    write_json_atomic(manifest_path, split_manifest)
    (split_dir / "train_indices.json").write_text(
        json.dumps(train_indices, indent=4) + "\r\n", encoding="utf-8", newline=""
    )
    (split_dir / "validation_indices.json").write_text(
        json.dumps(validation_indices, indent=4) + "\r\n",
        encoding="utf-8",
        newline="",
    )
    assert validate_phase2e_run(run_dir) == []


def test_phase2e_validator_uses_preserved_split_not_current_project_split(tmp_path: Path):
    test_phase2e_run_preserves_validation_indices_and_validates(tmp_path)
    run_dir = tmp_path / "artifacts/phase2e_disagreement_model/synthetic"
    indices_path = tmp_path / "data/processed/averitec/phase2b_split_v0_1/train_indices.json"
    indices = json.loads(indices_path.read_text())
    write_json_atomic(indices_path, list(reversed(indices)))

    assert validate_phase2e_run(run_dir) == []


def test_phase2e_validator_rejects_changed_preserved_training_indices(tmp_path: Path):
    test_phase2e_run_preserves_validation_indices_and_validates(tmp_path)
    run_dir = tmp_path / "artifacts/phase2e_disagreement_model/synthetic"
    indices_path = run_dir / "input_snapshot/train_indices.json"
    indices = json.loads(indices_path.read_text())
    write_json_atomic(indices_path, list(reversed(indices)))

    assert "input SHA-256 mismatch: train_indices.json" in validate_phase2e_run(run_dir)
