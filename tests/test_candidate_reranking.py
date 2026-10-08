from apv_rag.candidate_reranking import rerank_candidates

def test_global_sentence_ranking_preserves_source_and_budget():
    corpus=[{'doc_id':10,'abstract':['irrelevant ocean words']},{'doc_id':20,'abstract':['solar temperature change','unrelated text']},{'doc_id':30,'abstract':['solar temperature change']}]
    result=rerank_candidates('solar temperature change',corpus,[(0,9),(1,4),(2,2)],top_k=2)
    assert [i for i,_ in result]==[1,2]
    assert [score for _,score in result]==[4,2]
    assert corpus[1]['abstract']==['solar temperature change','unrelated text']

def test_zero_overlap_uses_original_order():
    corpus=[{'abstract':['alpha']},{'abstract':['beta']}]
    assert rerank_candidates('missing',corpus,[(1,7),(0,6)],top_k=1)==[(1,7)]
