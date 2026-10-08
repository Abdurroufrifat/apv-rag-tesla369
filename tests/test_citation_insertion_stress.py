from __future__ import annotations

import copy

import numpy as np
import pytest

from apv_rag.citation_insertion_stress import (
    citation_insertion_statistics,
    insert_fabricated_citations,
)


def _record():
    return {
        "claim": "A report says the bridge opened in 1927.",
        "label": "Supported",
        "questions": [
            {
                "question": "When did the bridge open?",
                "answers": [
                    {
                        "answer": "The bridge opened in 1927.",
                        "answer_type": "Extractive",
                        "source_url": "https://example.org/real",
                        "source_medium": "Web text",
                    }
                ],
            }
        ],
    }


def test_insertion_is_deterministic_and_changes_only_evidence_answers():
    record = _record()
    original = copy.deepcopy(record)

    changed, audit = insert_fabricated_citations(record, 2, upstream_index=41)

    assert record == original
    assert changed["claim"] == original["claim"]
    assert changed["label"] == original["label"]
    assert changed["questions"][0]["question"] == original["questions"][0]["question"]
    assert changed["questions"][0]["answers"][0] == original["questions"][0]["answers"][0]
    assert len(changed["questions"][0]["answers"]) == 3
    assert len(audit) == 2
    assert changed == insert_fabricated_citations(record, 2, upstream_index=41)[0]


def test_inserted_citations_use_reserved_domains_and_explicit_audit_markers():
    changed, audit = insert_fabricated_citations(_record(), 3, upstream_index=7)
    inserted = changed["questions"][0]["answers"][1:]

    assert [row["source_url"] for row in inserted] == [
        "https://citation-7-1.invalid/source",
        "https://citation-7-2.invalid/source",
        "https://citation-7-3.invalid/source",
    ]
    assert all(row["synthetic_test_citation"] is True for row in inserted)
    assert all(row["source_medium"] == "Synthetic stress-test citation" for row in inserted)
    assert all(entry["synthetic_test_citation"] is True for entry in audit)


def test_zero_insertions_return_an_unchanged_copy_and_negative_counts_fail():
    record = _record()
    unchanged, audit = insert_fabricated_citations(record, 0, upstream_index=3)

    assert unchanged == record
    assert unchanged is not record
    assert audit == []
    with pytest.raises(ValueError, match="non-negative"):
        insert_fabricated_citations(record, -1, upstream_index=3)


def test_insertion_creates_a_question_when_record_has_no_questions():
    record = {"claim": "A claim without evidence.", "label": "Not Enough Evidence", "questions": []}

    changed, audit = insert_fabricated_citations(record, 1, upstream_index=9)

    assert len(changed["questions"]) == 1
    assert changed["questions"][0]["question"] == "Synthetic citation stress-test prompt"
    assert len(changed["questions"][0]["answers"]) == 1
    assert len(audit) == 1


def test_statistics_measure_accuracy_confidence_and_support_probability_shift():
    result = citation_insertion_statistics(
        ["Supported", "Refuted"],
        ["Supported", "Refuted"],
        ["Supported", "Supported"],
        np.asarray([0.7, 0.2]),
        np.asarray([0.8, 0.6]),
        np.asarray([0.7, 0.8]),
        np.asarray([0.8, 0.7]),
    )

    assert result["prediction_flip_rate"] == 0.5
    assert result["mean_support_probability_shift"] == pytest.approx(0.25)
    assert result["mean_confidence_change"] == pytest.approx(0.0)
    assert result["confidence_increase_rate"] == 0.5
