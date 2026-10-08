from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from apv_rag.repetition_stress import (
    collapse_provenance_families,
    inject_derivative_copies,
    ribs_auc,
    ribs_statistics,
    run_phase2h,
    validate_phase2h_run,
)
from apv_rag.splits import sha256, write_json_atomic


def _record() -> dict:
    return {
        "claim": "A claim",
        "label": "Supported",
        "questions": [
            {
                "question": "What is the evidence?",
                "answers": [
                    {
                        "answer": "Repeated evidence",
                        "source_url": "https://www.example.org/page-a",
                    },
                    {
                        "answer": "Independent evidence",
                        "source_url": "https://independent.net/report",
                    },
                ],
            }
        ],
    }


def test_copy_injection_adds_requested_derivatives_without_mutating_source():
    original = _record()
    source = copy.deepcopy(original)

    perturbed = inject_derivative_copies(original, 5)

    assert original == source
    assert len(perturbed["questions"][0]["answers"]) == 7
    copies = perturbed["questions"][0]["answers"][2:]
    assert {row["source_url"] for row in copies} == {"https://www.example.org/page-a"}
    assert all(row["answer"] == "Repeated evidence" for row in copies)


def test_family_collapse_keeps_one_answer_per_domain_and_keeps_missing_urls_separate():
    record = _record()
    record["questions"][0]["answers"].extend(
        [
            {"answer": "Mirror", "source_url": "https://example.org/page-b"},
            {"answer": "Missing one", "source_url": ""},
            {"answer": "Missing two"},
        ]
    )

    collapsed = collapse_provenance_families(record)
    answers = collapsed["questions"][0]["answers"]

    assert [row["answer"] for row in answers] == [
        "Repeated evidence",
        "Independent evidence",
        "Missing one",
        "Missing two",
    ]


def test_family_collapse_makes_copy_injection_invariant():
    baseline = collapse_provenance_families(_record())
    stressed = collapse_provenance_families(inject_derivative_copies(_record(), 25))

    assert stressed == baseline


def test_ribs_statistics_measure_support_probability_shift():
    baseline = np.asarray([0.2, 0.7, 0.4])
    stressed = np.asarray([0.3, 0.5, 0.4])

    result = ribs_statistics(baseline, stressed)

    assert np.isclose(result["mean_signed_ribs"], np.mean([0.1, -0.2, 0.0]))
    assert np.isclose(result["mean_absolute_ribs"], np.mean([0.1, 0.2, 0.0]))
    assert np.isclose(result["maximum_absolute_ribs"], 0.2)


def test_ribs_auc_is_copy_count_normalized_trapezoid():
    counts = [0, 5, 10]
    values = [0.0, 0.1, 0.2]

    assert ribs_auc(counts, values) == 0.1


def test_phase2h_run_is_self_contained_and_validates(tmp_path: Path):
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
    (config_dir / "repetition_stress.yaml").write_text(
        """protocol_version: '0.1'
random_seed: 369
copy_counts: [0, 1]
regularization_c: 1.0
word_tfidf: {ngram_range: [1, 2], min_df: 1, max_features: 1000}
char_tfidf: {ngram_range: [3, 5], min_df: 1, max_features: 1000}
calibration_cv: 2
bootstrap_repetitions: 20
output_directory: artifacts/phase2h_repetition_stress
""",
        encoding="utf-8",
    )

    run_dir = run_phase2h(tmp_path, run_id="synthetic", expected_source_sha256=sha256(source))
    write_json_atomic(split_dir / "validation_indices.json", list(reversed(validation_indices)))

    assert (run_dir / "input_snapshot/validation_indices.json").is_file()
    assert validate_phase2h_run(run_dir) == []
