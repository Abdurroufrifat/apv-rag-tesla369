"""Exploratory mean masking of existing gate features; no fitting or metadata invention."""
import numpy as np

from apv_rag.integrated_gate import probability_complete

VARIANTS = ('full', 'without_nli', 'without_embedding', 'without_both')
LABELS = ('Supported', 'Refuted', 'Not Enough Evidence')


def masked_probability(features, model, variant):
    if variant not in VARIANTS or model['columns'] != list(range(13)):
        raise ValueError('Expected a declared variant and the frozen combined 13-feature head')
    values = np.asarray(features, dtype=float).copy()
    means = np.asarray(model['mean'], dtype=float)
    if values.shape != (13,) or means.shape != (13,) or not np.isfinite(values).all() or not np.isfinite(means).all():
        raise ValueError('Finite 13-feature vectors and training means required')
    if variant in ('without_nli', 'without_both'):
        values[:9] = means[:9]
    if variant in ('without_embedding', 'without_both'):
        values[9:] = means[9:]
    return probability_complete(values, model)


def summarize(predictions, truth):
    if not truth or len(predictions) != len(truth) or any(t not in LABELS for t in truth) or any(p not in (*LABELS, None) for p in predictions):
        raise ValueError('Aligned nonempty predictions and benchmark truth required')
    accepted = sum(p is not None for p in predictions)
    correct = sum(p == t for p, t in zip(predictions, truth, strict=True))
    return {'queries': len(truth), 'accepted': accepted, 'correct': correct,
            'coverage': accepted / len(truth), 'all_query_accuracy': correct / len(truth),
            'accepted_accuracy': correct / accepted if accepted else None}
