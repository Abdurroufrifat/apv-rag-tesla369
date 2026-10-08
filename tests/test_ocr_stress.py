from __future__ import annotations

import copy

import numpy as np

from apv_rag.ocr_stress import corrupt_evidence_record, corrupt_text, robustness_statistics


def test_zero_rate_preserves_text_and_nonzero_rate_is_deterministic():
    text = "Archival evidence 1927"

    assert corrupt_text(text, 0.0, seed=369) == text
    first = corrupt_text(text, 0.2, seed=369)
    second = corrupt_text(text, 0.2, seed=369)

    assert first == second
    assert first != text
    assert len(first) == len(text)


def test_corruption_changes_exact_rounded_number_of_alphanumeric_characters():
    original = "abcdefghij"
    corrupted = corrupt_text(original, 0.2, seed=369)

    assert sum(a != b for a, b in zip(original, corrupted, strict=True)) == 2


def test_record_corruption_changes_answers_but_preserves_claim_label_and_source():
    record = {
        "claim": "Keep this claim",
        "label": "Supported",
        "questions": [
            {
                "question": "Keep generated question",
                "answers": [
                    {
                        "answer": "Corrupt this evidence passage",
                        "source_url": "https://example.org/source",
                    }
                ],
            }
        ],
    }
    original = copy.deepcopy(record)

    changed = corrupt_evidence_record(record, 0.2, seed=369)

    assert record == original
    assert changed["claim"] == original["claim"]
    assert changed["label"] == original["label"]
    assert changed["questions"][0]["question"] == original["questions"][0]["question"]
    assert changed["questions"][0]["answers"][0]["source_url"] == "https://example.org/source"
    assert changed["questions"][0]["answers"][0]["answer"] != "Corrupt this evidence passage"


def test_robustness_statistics_report_macro_f1_loss_flips_and_confidence_shift():
    truth = ["Supported", "Refuted", "Supported", "Refuted"]
    clean_pred = ["Supported", "Refuted", "Supported", "Refuted"]
    noisy_pred = ["Refuted", "Refuted", "Supported", "Supported"]
    clean_confidence = np.asarray([0.9, 0.8, 0.7, 0.6])
    noisy_confidence = np.asarray([0.6, 0.7, 0.6, 0.5])

    result = robustness_statistics(
        truth, clean_pred, noisy_pred, clean_confidence, noisy_confidence
    )

    assert result["clean_macro_f1"] == 1.0
    assert result["noisy_macro_f1"] == 0.5
    assert result["macro_f1_change"] == -0.5
    assert result["prediction_flip_rate"] == 0.5
    assert np.isclose(result["mean_confidence_change"], -0.15)
