"""Paired multiclass effect sizes and multiplicity correction."""

import numpy as np


def macro_score(truth, predicted, classes=4):
    matrix = np.bincount(truth * classes + predicted, minlength=classes**2)
    matrix = matrix.reshape(classes, classes)
    denominator = matrix.sum(0) + matrix.sum(1)
    return float(
        np.divide(
            2 * np.diag(matrix), denominator, out=np.zeros(classes), where=denominator != 0
        ).mean()
    )


def holm_adjust(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a finite vector in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()
