"""Training-only selection of class-prior-adjusted verdict decisions."""

import numpy as np


def decision_indices(probabilities, priors, exponent):
    p, prior = np.asarray(probabilities, dtype=float), np.asarray(priors, dtype=float)
    if (
        p.ndim != 2
        or prior.shape != (p.shape[1],)
        or not np.isfinite(p).all()
        or not np.isfinite(prior).all()
        or (prior <= 0).any()
        or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1)
        or not np.isclose(prior.sum(), 1)
        or not 0 <= exponent <= 1
    ):
        raise ValueError("invalid probabilities, priors, or exponent")
    return (p / prior[None, :] ** exponent).argmax(axis=1)
