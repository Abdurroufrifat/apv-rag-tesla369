from apv_rag.retrieval_diagnostics import ownership_metrics


def test_shared_excerpt_ownership_and_empty_claim_denominator():
    documents = [{"text": "shared", "source_url": "u"}, {"text": "other", "source_url": "v"}]
    records = [
        {"questions": [{"answers": [{"answer": "shared", "source_url": "u"}]}]},
        {"questions": []},
    ]
    result = ownership_metrics(records, documents, [[1, 0], []])
    assert result["claims_with_answer_excerpts"] == 1
    assert result["own_excerpt_hit_at_1"] == 0
    assert result["own_excerpt_hit_at_5"] == 1
    assert result["own_excerpt_mrr_at_5"] == 0.5
    assert result["empty_retrievals"] == 1
