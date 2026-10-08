import numpy as np
import pytest

from apv_rag.retrieved_representation import audit_features


def test_retrieval_positions_select_correct_texts_and_sources():
    docs = [
        {"text": "unrelated", "source_url": "https://a.org"},
        {"text": "claim words", "source_url": "https://b.org"},
    ]
    x = audit_features(
        "claim words", docs, {"document_ids": [1], "nli_probabilities": [[0.1, 0.8, 0.1]]}
    )
    assert x[13] == 1
    assert x[15] == 1
    assert np.isfinite(x).all()
    with pytest.raises(ValueError):
        audit_features("claim", docs, {"document_ids": [2], "nli_probabilities": [[0.1, 0.8, 0.1]]})
    with pytest.raises(ValueError):
        audit_features(
            "claim", docs, {"document_ids": [1, 1], "nli_probabilities": [[0.1, 0.8, 0.1]] * 2}
        )
