import pytest

from apv_rag.retrieval import BM25Index


def test_ranker_returns_matching_document_and_excludes_zero_score_documents():
    index = BM25Index(["Tesla worked on electricity", "Rice grows in water"])
    assert [row[0] for row in index.search("Tesla electricity", 5)] == [0]
    assert index.search("unknownword", 5) == []


def test_ranker_breaks_ties_by_document_index_and_handles_empty_documents():
    index = BM25Index(["bridge opened", "bridge opened", ""])
    results = index.search("bridge", 1)
    assert results[0][0] == 0
    assert results[0][1] > 0


def test_ranker_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        BM25Index(["text"], k1=0)
    with pytest.raises(ValueError):
        BM25Index(["text"], b=1.1)
    with pytest.raises(ValueError):
        BM25Index(["text"]).search("text", -1)
