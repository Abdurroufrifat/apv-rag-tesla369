from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from apv_rag.provenance_model import (
    PROVENANCE_FEATURES,
    build_phase2f_pipeline,
    extract_provenance_features,
    normalize_source_url,
    run_phase2f,
    select_phase2f_setting,
    validate_phase2f_run,
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
        "claim": "excluded from provenance features",
        "label": "Supported",
        "justification": "excluded",
        "questions": [
            {
                "question": "Q1",
                "answers": [
                    {
                        "answer": "The same answer",
                        "source_url": "https://www.example.com/report?id=7&utm_source=x",
                        "cached_source_url": "https://web.archive.org/web/20230101/https://example.com/report?id=7",
                    },
                    {
                        "answer": "The same answer.",
                        "source_url": "http://example.com/report?id=7",
                    },
                    {
                        "answer": "Independent answer",
                        "source_url": "https://independent.org/item",
                    },
                ],
            }
        ],
    }


def test_normalize_source_url_recovers_archive_target_and_removes_tracking():
    archived = (
        "https://web.archive.org/web/20230101120000/"
        "https://WWW.Example.com/news/?id=4&utm_source=social"
    )

    assert normalize_source_url(archived) == "example.com/news?id=4"


def test_features_collapse_repeated_domains_urls_and_answers():
    values = dict(zip(PROVENANCE_FEATURES, extract_provenance_features(_record()), strict=True))

    assert values["source_count"] == 3.0
    assert values["family_count"] == 2.0
    assert values["family_diversity"] == 2.0 / 3.0
    assert values["largest_family_fraction"] == 2.0 / 3.0
    assert values["duplicate_url_fraction"] == 1.0 / 3.0
    assert values["duplicate_answer_fraction"] == 1.0 / 3.0
    assert values["archive_fraction"] == 1.0 / 3.0


def test_provenance_features_ignore_labels_claims_and_justifications():
    original = _record()
    changed = copy.deepcopy(original)
    changed["claim"] = "different"
    changed["label"] = "Refuted"
    changed["justification"] = "different"

    np.testing.assert_allclose(
        extract_provenance_features(original), extract_provenance_features(changed)
    )


def test_selection_breaks_macro_f1_tie_with_conflict_f1():
    conflict = "Conflicting Evidence/Cherrypicking"
    candidates = [
        {
            "representation": "phase2e",
            "c": 1.0,
            "metrics": {
                "macro_f1": 0.5,
                "balanced_accuracy": 0.6,
                "per_class": {conflict: {"f1": 0.1}},
            },
        },
        {
            "representation": "phase2e_plus_provenance",
            "c": 1.0,
            "metrics": {
                "macro_f1": 0.5,
                "balanced_accuracy": 0.55,
                "per_class": {conflict: {"f1": 0.3}},
            },
        },
    ]

    assert select_phase2f_setting(candidates)["representation"] == ("phase2e_plus_provenance")


def test_combined_pipeline_contains_text_disagreement_and_provenance():
    pipeline = build_phase2f_pipeline(_config(), "phase2e_plus_provenance", 1.0)

    branches = pipeline.named_steps["features"].transformer_list
    assert [name for name, _ in branches] == ["phase2e", "provenance"]
    assert pipeline.named_steps["classifier"].estimator.class_weight == "balanced"


def test_phase2f_run_is_self_contained_and_validates(tmp_path: Path):
    labels = [
        "Supported",
        "Refuted",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
    ]
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
        {"source": {"path": "data/external/averitec/official_7c62d1e/train.json"}},
    )
    (config_dir / "provenance_model.yaml").write_text(
        """protocol_version: '0.1'
random_seed: 369
representations: [phase2e, provenance_only, phase2e_plus_provenance]
regularization_c: [1.0]
word_tfidf: {ngram_range: [1, 2], min_df: 1, max_features: 1000}
char_tfidf: {ngram_range: [3, 5], min_df: 1, max_features: 1000}
calibration_cv: 2
ece_bins: 5
target_coverages: [0.5, 1.0]
output_directory: artifacts/phase2f_provenance_model
""",
        encoding="utf-8",
    )

    run_dir = run_phase2f(tmp_path, run_id="synthetic", expected_source_sha256=sha256(source))
    write_json_atomic(split_dir / "train_indices.json", list(reversed(train_indices)))

    assert (run_dir / "input_snapshot/train_indices.json").is_file()
    assert validate_phase2f_run(run_dir) == []
