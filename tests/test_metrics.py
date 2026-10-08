from __future__ import annotations

import numpy as np
import pytest

from apv_rag.metrics import (
    classification_metrics,
    expected_calibration_error,
    multiclass_brier,
    risk_coverage,
)


def test_multiclass_brier_is_zero_for_perfect_predictions():
    labels = ["A", "B"]
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert multiclass_brier(["A", "B"], probabilities, labels) == 0.0


def test_expected_calibration_error_uses_confidence_bins():
    probabilities = np.array([[0.8, 0.2], [0.6, 0.4]])

    value = expected_calibration_error(
        ["A", "B"], ["A", "A"], probabilities, bins=5
    )

    assert value == pytest.approx(0.4)


def test_risk_coverage_keeps_highest_confidence_first():
    output = risk_coverage(
        ["A", "B"],
        ["A", "A"],
        np.array([0.9, 0.4]),
        [0.5, 1.0],
    )

    assert output["0.500000"]["selected"] == 1
    assert output["0.500000"]["accuracy"] == 1.0
    assert output["1.000000"]["risk"] == 0.5


def test_classification_metrics_reports_all_requested_scores():
    output = classification_metrics(
        ["A", "A", "B", "B"],
        ["A", "B", "B", "B"],
        np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]]),
        ["A", "B"],
        ece_bins=2,
        coverages=[0.5, 1.0],
    )

    assert output["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert output["balanced_accuracy"] == 0.75
    assert set(output["per_class"]) == {"A", "B"}
    assert "multiclass_brier" in output
    assert "expected_calibration_error" in output
    assert set(output["risk_coverage"]) == {"0.500000", "1.000000"}


def test_metrics_reject_probability_rows_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to one"):
        multiclass_brier(["A"], np.array([[0.8, 0.8]]), ["A", "B"])
