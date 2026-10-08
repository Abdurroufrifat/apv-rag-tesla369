"""Binary benchmark-label proxy; not a retrieved-evidence sufficiency label."""

import numpy as np

from apv_rag.evidence_baseline import LABELS


def proxy_labels(labels):
    if any(label not in LABELS for label in labels):
        raise ValueError("Unknown benchmark label")
    return np.array([label != "Not Enough Evidence" for label in labels], dtype=int)


def select_threshold(truth, probabilities):
    from sklearn.metrics import f1_score

    truth, probabilities = np.asarray(truth), np.asarray(probabilities)
    if probabilities.shape != truth.shape or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability vector")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Invalid probability range")
    grid = np.arange(1, 10) / 10
    scores = [
        f1_score(truth, probabilities >= t, labels=[0, 1], average="macro", zero_division=0)
        for t in grid
    ]
    index = min(range(len(grid)), key=lambda i: (-scores[i], abs(grid[i] - 0.5), grid[i]))
    return float(grid[index]), dict(zip(map(str, grid), map(float, scores), strict=True))
