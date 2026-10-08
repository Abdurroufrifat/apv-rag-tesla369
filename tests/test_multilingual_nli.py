import numpy as np
import pytest

from apv_rag.direct_nli import direct_probabilities
from apv_rag.multilingual_nli import cen_order


def test_model_label_permutation_preserves_stance():
    scores = np.array([[0.8, 0.1, 0.1]])  # E/N/C order
    order = cen_order({0: "entailment", 1: "neutral", 2: "contradiction"})
    assert np.allclose(direct_probabilities(scores[:, order].tolist()), [0.8, 0.1, 0.1])
    with pytest.raises(ValueError):
        cen_order({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_low_precision_probability_rounding_and_invalid_outputs():
    from apv_rag.multilingual_nli import normalize_model_scores

    values = normalize_model_scores([[0.7998046875, 0.0999755859375, 0.0999755859375]])
    assert np.allclose(np.sum(values, axis=1), 1.0)
    for bad in ([[float('nan'), 0.5, 0.5]], [[0.2, 0.2, 0.2]], [[-0.1, 0.5, 0.6]]):
        with pytest.raises(ValueError, match='Model probability'):
            normalize_model_scores(bad)
