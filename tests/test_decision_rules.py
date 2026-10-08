import numpy as np
import pytest

from apv_rag.decision_rules import decision_indices


def test_zero_exponent_preserves_probability_argmax():
    p = np.asarray([[0.1, 0.6, 0.2, 0.1]])
    assert decision_indices(p, [0.25, 0.5, 0.15, 0.1], 0).tolist() == [1]
    assert p.tolist() == [[0.1, 0.6, 0.2, 0.1]]


def test_prior_adjustment_changes_decision_without_changing_probabilities():
    p = np.asarray([[0.1, 0.6, 0.2, 0.1]])
    assert decision_indices(p, [0.25, 0.6, 0.1, 0.05], 1).tolist() == [2]
    assert p[0, 1] == 0.6


def test_invalid_priors_are_rejected():
    with pytest.raises(ValueError):
        decision_indices(np.asarray([[0.5, 0.5]]), [1, 0], 1)
