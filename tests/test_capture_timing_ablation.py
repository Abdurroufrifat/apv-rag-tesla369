from __future__ import annotations

import copy

import numpy as np

from apv_rag.capture_timing_ablation import capture_features, build_timing_pipeline


def _record():
    return {
        "claim": "A claim",
        "claim_date": "25-8-2020",
        "label": "Supported",
        "justification": "not a feature",
        "questions": [{"question": "Where?", "answers": [
            {"answer": "Yes", "cached_source_url": "https://web.archive.org/web/20200824/https://a.org"},
            {"answer": "No", "cached_source_url": "https://web.archive.org/web/20200826/https://b.org"},
            {"answer": "Maybe", "cached_source_url": "https://web.archive.org/web/20201390/https://c.org"},
            {"answer": "Other", "source_url": "https://web.archive.org/web/20200827/https://d.org"},
        ]}],
    }


def test_capture_timing_uses_parseable_wayback_snapshot_not_publication_date():
    values = capture_features(_record())
    assert len(values) == 5
    np.testing.assert_allclose(values[:3], [1, 3 / 4, 2 / 3])
    np.testing.assert_allclose(values[3], 1 / 365.25)
    np.testing.assert_allclose(values[4], 1 / 365.25)


def test_missing_claim_date_and_non_archive_sources_have_no_invented_age():
    row = _record()
    row["claim_date"] = None
    assert capture_features(row).tolist() == [0, 0.75, 0, 0, 0]
    row["questions"][0]["answers"] = [{"source_url": "https://example.org/news"}]
    assert capture_features(row).tolist() == [0, 0, 0, 0, 0]


def test_capture_features_exclude_gold_fields_and_augment_existing_baseline():
    row = _record()
    other = copy.deepcopy(row)
    other["label"] = "Refuted"
    other["justification"] = "different"
    np.testing.assert_array_equal(capture_features(row), capture_features(other))
    model = build_timing_pipeline({
        "random_seed": 369,
        "calibration_cv": 2,
        "word_tfidf": {"ngram_range": [1, 2], "min_df": 1, "max_features": 1000},
        "char_tfidf": {"ngram_range": [3, 5], "min_df": 1, "max_features": 1000},
    })
    assert [name for name, _ in model.named_steps["features"].transformer_list] == [
        "phase2e", "capture_timing"
    ]
