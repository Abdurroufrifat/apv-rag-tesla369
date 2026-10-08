from __future__ import annotations

import copy

import numpy as np

from apv_rag.paraphrase_stress import (
    paraphrase_claim,
    paraphrase_record,
    paraphrase_statistics,
)


def test_mild_and_strong_paraphrases_are_deterministic_and_conservative():
    claim = "The report says the event happened because officials found evidence."

    mild = paraphrase_claim(claim, "mild")
    strong = paraphrase_claim(claim, "strong")

    assert mild == paraphrase_claim(claim, "mild")
    assert strong == paraphrase_claim(claim, "strong")
    assert mild != claim
    assert strong != mild
    assert "officials" in strong


def test_paraphrase_preserves_numbers_named_tokens_and_terminal_punctuation():
    claim = "Tesla reportedly visited New York in 1927."
    changed = paraphrase_claim(claim, "strong")

    assert "Tesla" in changed
    assert "New York" in changed
    assert "1927" in changed
    assert changed.endswith(".")


def test_record_paraphrase_changes_only_claim():
    record = {
        "claim": "The report says the event happened.",
        "label": "Supported",
        "questions": [{"question": "Q", "answers": [{"answer": "A"}]}],
    }
    original = copy.deepcopy(record)

    changed = paraphrase_record(record, "mild")

    assert record == original
    assert changed["claim"] != original["claim"]
    assert changed["label"] == original["label"]
    assert changed["questions"] == original["questions"]


def test_statistics_report_all_records_and_changed_subset_separately():
    truth = ["Supported", "Refuted", "Supported", "Refuted"]
    clean = ["Supported", "Refuted", "Supported", "Refuted"]
    noisy = ["Refuted", "Refuted", "Refuted", "Refuted"]
    mask = np.asarray([True, False, True, False])

    result = paraphrase_statistics(
        truth,
        clean,
        noisy,
        np.asarray([0.9, 0.8, 0.7, 0.6]),
        np.asarray([0.6, 0.8, 0.5, 0.6]),
        mask,
    )

    assert result["transformation_coverage"] == 0.5
    assert result["all_records"]["prediction_flip_rate"] == 0.5
    assert result["changed_records"]["prediction_flip_rate"] == 1.0
    assert result["changed_records"]["count"] == 2
