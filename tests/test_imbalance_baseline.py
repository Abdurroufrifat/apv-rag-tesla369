from __future__ import annotations

import json
from pathlib import Path

from apv_rag.imbalance_baseline import (
    build_phase2d_pipeline,
    run_phase2d,
    select_phase2d_setting,
    validate_phase2d_run,
)
from apv_rag.splits import sha256, write_json_atomic

LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


def _config() -> dict:
    return {
        "random_seed": 369,
        "calibration_cv": 3,
        "word_tfidf": {"ngram_range": [1, 2], "min_df": 1, "max_features": 5000},
        "char_tfidf": {"ngram_range": [3, 5], "min_df": 1, "max_features": 5000},
    }


def _record(label: str, token: str, item: int) -> dict:
    return {
        "claim": f"{token} claim {item}",
        "label": label,
        "justification": "excluded",
        "speaker": "excluded",
        "reporting_source": "excluded",
        "fact_checking_article": "https://excluded.example",
        "questions": [
            {
                "question": f"What evidence says {token}?",
                "answers": [
                    {
                        "answer": f"Evidence contains {token}",
                        "answer_type": "Extractive",
                        "source_medium": "Web text",
                        "source_url": "https://excluded.example/source",
                    }
                ],
            }
        ],
    }


def _make_project(root: Path) -> tuple[Path, list[int]]:
    source_dir = root / "data/external/averitec/official_7c62d1e"
    split_dir = root / "data/processed/averitec/phase2b_split_v0_1"
    config_dir = root / "config"
    source_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    config_dir.mkdir()
    records = [
        _record(label, f"labeltoken{label_index}", item)
        for label_index, label in enumerate(LABELS)
        for item in range(8)
    ]
    source_path = source_dir / "train.json"
    source_path.write_text(json.dumps(records), encoding="utf-8")
    train_indices = [index for index in range(32) if index % 8 < 6]
    validation_indices = [index for index in range(32) if index % 8 >= 6]
    write_json_atomic(split_dir / "train_indices.json", train_indices)
    write_json_atomic(split_dir / "validation_indices.json", validation_indices)
    write_json_atomic(
        split_dir / "split_manifest.json",
        {"source": {"path": "data/external/averitec/official_7c62d1e/train.json"}},
    )
    (config_dir / "imbalance_baseline.yaml").write_text(
        """protocol_version: '0.1'
random_seed: 369
representations: [word, char, hybrid]
regularization_c: [1.0]
word_tfidf: {ngram_range: [1, 2], min_df: 1, max_features: 5000}
char_tfidf: {ngram_range: [3, 5], min_df: 1, max_features: 5000}
calibration_cv: 3
ece_bins: 5
target_coverages: [0.5, 1.0]
output_directory: artifacts/phase2d_imbalance_baseline
""",
        encoding="utf-8",
    )
    return source_path, validation_indices


def test_hybrid_pipeline_uses_word_char_features_and_balanced_classes():
    pipeline = build_phase2d_pipeline(_config(), "hybrid", 1.0)

    features = pipeline.named_steps["features"]
    assert [name for name, _ in features.transformer_list] == ["word", "char"]
    assert pipeline.named_steps["classifier"].estimator.class_weight == "balanced"


def test_selection_breaks_macro_f1_tie_with_conflict_f1():
    candidates = [
        {
            "representation": "word",
            "c": 1.0,
            "metrics": {
                "macro_f1": 0.5,
                "balanced_accuracy": 0.6,
                "per_class": {LABELS[3]: {"f1": 0.1}},
            },
        },
        {
            "representation": "hybrid",
            "c": 1.0,
            "metrics": {
                "macro_f1": 0.5,
                "balanced_accuracy": 0.55,
                "per_class": {LABELS[3]: {"f1": 0.3}},
            },
        },
    ]

    assert select_phase2d_setting(candidates)["representation"] == "hybrid"


def test_phase2d_run_writes_figures_and_passes_integrity_validation(tmp_path: Path):
    source_path, validation_indices = _make_project(tmp_path)

    run_dir = run_phase2d(
        tmp_path,
        run_id="synthetic",
        expected_source_sha256=sha256(source_path),
    )

    assert {
        "confusion_matrix.png",
        "metrics.json",
        "model_selection.json",
        "representation_comparison.png",
        "run_manifest.json",
        "validation_predictions.jsonl",
    } == {path.name for path in run_dir.iterdir()}
    predictions = [
        json.loads(line)
        for line in (run_dir / "validation_predictions.jsonl").read_text().splitlines()
    ]
    assert [row["upstream_index"] for row in predictions] == validation_indices
    assert validate_phase2d_run(run_dir) == []


def test_phase2d_accepts_windows_style_source_path_in_split_manifest(tmp_path: Path):
    source_path, _ = _make_project(tmp_path)
    manifest_path = (
        tmp_path
        / "data/processed/averitec/phase2b_split_v0_1/split_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["path"] = (
        "data\\external\\averitec\\official_7c62d1e\\train.json"
    )
    write_json_atomic(manifest_path, manifest)

    run_dir = run_phase2d(
        tmp_path,
        run_id="windows-path",
        expected_source_sha256=sha256(source_path),
    )

    assert run_dir.name == "windows-path"


def test_phase2d_validator_recomputes_metrics_after_hash_is_updated(tmp_path: Path):
    source_path, _ = _make_project(tmp_path)
    run_dir = run_phase2d(
        tmp_path,
        run_id="synthetic",
        expected_source_sha256=sha256(source_path),
    )
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["macro_f1"] = 0.123456
    write_json_atomic(metrics_path, metrics)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["metrics.json"]["sha256"] = sha256(metrics_path)
    manifest["outputs"]["metrics.json"]["bytes"] = metrics_path.stat().st_size
    write_json_atomic(manifest_path, manifest)

    errors = validate_phase2d_run(run_dir)

    assert "recorded metrics do not match predictions" in errors
