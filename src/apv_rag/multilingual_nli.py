"""Normalize a model's declared label order to contradiction/entailment/neutral."""

import numpy as np


def cen_order(mapping):
    labels = {int(k): str(v).lower() for k, v in mapping.items()}
    if set(labels) != {0, 1, 2} or set(labels.values()) != {
        "contradiction", "entailment", "neutral"
    }:
        raise ValueError("Unexpected NLI mapping")
    return [next(k for k, v in labels.items() if v == label)
            for label in ("contradiction", "entailment", "neutral")]



def normalize_model_scores(scores):
    """Correct only small softmax rounding drift; reject invalid inference."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"Model probability output is nonfinite or malformed: {values.tolist()}")
    totals = values.sum(axis=1)
    if (values < 0).any() or (values > 1).any() or not np.allclose(
        totals, 1.0, rtol=0, atol=0.001
    ):
        raise ValueError(
            f"Model probability output is invalid: "
            f"values={values.tolist()}, sums={totals.tolist()}"
        )
    return (values / totals[:, None]).tolist()
