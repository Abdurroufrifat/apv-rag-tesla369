"""Untuned three-label baseline from mean retrieved-premise NLI probabilities."""

import numpy as np

TARGET_LABELS = ("Supported", "Refuted", "Not Enough Evidence")


def direct_probabilities(scores):
    if not scores:
        return np.array([0.0, 0.0, 1.0])
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Invalid NLI probability matrix")
    if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(1), 1):
        raise ValueError("Invalid NLI probabilities")
    # Original model order: contradiction, entailment, neutral.
    return values.mean(0)[[1, 0, 2]]
