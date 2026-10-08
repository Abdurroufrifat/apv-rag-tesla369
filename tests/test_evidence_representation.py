import numpy as np
import pytest

from apv_rag.evidence_representation import FEATURE_NAMES, expanded_features
from apv_rag.nli_comparison import nli_features


def test_expansion_preserves_base_and_measures_overlap_and_source_counts():
    scores = [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]
    x = expanded_features(
        "alpha beta", ["alpha gamma", "beta"], ["https://a.org/1", "https://a.org/2"], scores
    )
    assert len(x) == len(FEATURE_NAMES) == 18
    np.testing.assert_allclose(x[:7], nli_features(scores))
    assert x[FEATURE_NAMES.index("claim_token_overlap_max")] == 0.5
    assert x[FEATURE_NAMES.index("unique_source_hosts")] == 1
    assert x[FEATURE_NAMES.index("contradiction_entailment_max_product")] == pytest.approx(0.64)
    assert expanded_features("claim", [], [], []).tolist() == [0] * 18


def test_expansion_rejects_unpaired_scores():
    with pytest.raises(ValueError):
        expanded_features("claim", ["text"], [], [[0.1, 0.8, 0.1]])
