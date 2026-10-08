import pytest

from apv_rag.generative_interface import allowed_next_tokens


def test_constrained_label_trie_has_only_legal_continuations():
    labels = [[2, 3], [2, 4], [5]]
    assert allowed_next_tokens(labels, [], 1) == [2, 5]
    assert allowed_next_tokens(labels, [2], 1) == [3, 4]
    assert allowed_next_tokens(labels, [2, 3], 1) == [1]
    with pytest.raises(ValueError):
        allowed_next_tokens(labels, [7], 1)
