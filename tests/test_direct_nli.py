import numpy as np
import pytest

from apv_rag.direct_nli import direct_probabilities


def test_mean_mapping_and_missing_evidence():
    assert np.allclose(direct_probabilities([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]), [0.5, 0.4, 0.1])
    assert np.array_equal(direct_probabilities([]), [0, 0, 1])


def test_reject_invalid_scores():
    for scores in [[[1, 1, 1]], [[float("nan"), 0, 1]], [[1, 0]]]:
        with pytest.raises(ValueError):
            direct_probabilities(scores)
