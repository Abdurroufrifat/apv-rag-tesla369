"""Evaluation metrics for calibrated multiclass claim verification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score


def _validate_inputs(
    y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str]
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(y_true), len(labels)):
        raise ValueError("probability matrix shape does not match records and labels")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("probabilities must be finite values between zero and one")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    unknown = set(y_true) - set(labels)
    if unknown:
        raise ValueError(f"unknown true labels: {sorted(unknown)}")
    return matrix


def multiclass_brier(
    y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str]
) -> float:
    """Return the mean squared distance from one-hot outcomes."""

    matrix = _validate_inputs(y_true, probabilities, labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    targets = np.zeros_like(matrix)
    for row, label in enumerate(y_true):
        targets[row, label_to_index[label]] = 1.0
    return float(np.mean(np.sum((matrix - targets) ** 2, axis=1)))


def expected_calibration_error(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    """Compute top-label equal-width expected calibration error."""

    if bins < 1:
        raise ValueError("bins must be positive")
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(y_true) or len(y_pred) != len(y_true):
        raise ValueError("prediction lengths do not match probability rows")
    confidences = matrix.max(axis=1)
    correct = np.asarray(
        [truth == prediction for truth, prediction in zip(y_true, y_pred, strict=True)]
    )
    total = len(y_true)
    if total == 0:
        raise ValueError("at least one prediction is required")
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences >= lower) & (
            confidences <= upper if index == bins - 1 else confidences < upper
        )
        count = int(mask.sum())
        if count:
            value += (count / total) * abs(float(correct[mask].mean() - confidences[mask].mean()))
    return float(value)


def risk_coverage(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: np.ndarray,
    coverages: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    """Evaluate accuracy and error risk after retaining most-confident predictions."""

    confidence_array = np.asarray(confidences, dtype=float)
    if confidence_array.shape != (len(y_true),) or len(y_pred) != len(y_true):
        raise ValueError("prediction lengths do not match confidences")
    if len(y_true) == 0:
        raise ValueError("at least one prediction is required")
    order = np.argsort(-confidence_array, kind="stable")
    correct = np.asarray(
        [truth == prediction for truth, prediction in zip(y_true, y_pred, strict=True)]
    )
    output: dict[str, dict[str, float | int]] = {}
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverage values must be in (0, 1]")
        selected = min(len(y_true), max(1, math.ceil(float(coverage) * len(y_true))))
        accuracy = float(correct[order[:selected]].mean())
        output[f"{float(coverage):.6f}"] = {
            "selected": selected,
            "realized_coverage": selected / len(y_true),
            "accuracy": accuracy,
            "risk": 1.0 - accuracy,
            "minimum_confidence": float(confidence_array[order[selected - 1]]),
        }
    return output


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
    *,
    ece_bins: int,
    coverages: Sequence[float],
) -> dict[str, Any]:
    """Compute the fixed Phase 2C classification and calibration metrics."""

    matrix = _validate_inputs(y_true, probabilities, labels)
    if len(y_pred) != len(y_true) or set(y_pred) - set(labels):
        raise ValueError("predicted labels do not align with the fixed labels")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    confidences = matrix.max(axis=1)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "multiclass_brier": multiclass_brier(y_true, matrix, labels),
        "expected_calibration_error": expected_calibration_error(
            y_true, y_pred, matrix, bins=ece_bins
        ),
        "risk_coverage": risk_coverage(y_true, y_pred, confidences, coverages),
    }
