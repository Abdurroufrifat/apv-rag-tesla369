import pytest


def test_character_tokens_retrieve_chinese_and_japanese_without_spaces():
    from apv_rag.multilingual_retrieval import build_pool, retrieve
    for claim,texts in [
        ('东京首都',['东京是日本首都。','巴黎是法国首都。']),
        ('東京首都',['東京は日本の首都です。','パリはフランスの首都です。'])]:
        pool=build_pool(texts)
        rows=retrieve([{'row':0,'claim_id':1,'claim':claim}],pool,'unicode_words_cjk_1_2')
        assert rows[0]['evidence'][0]['text']==texts[0]
        assert rows[0]['evidence'][0]['score']>0


def test_pool_deduplicates_text_and_is_independent_of_source_row_order():
    from apv_rag.multilingual_retrieval import build_pool
    a=build_pool(['A is an island.',' a  is an island. ','B is a city.'])
    b=build_pool(['B is a city.',' a  is an island. ','A is an island.'])
    assert a==b
    assert len(a)==2
    assert all(set(d)=={'id','text'} for d in a)


def test_retrieval_refuses_labels_and_query_specific_evidence():
    from apv_rag.multilingual_retrieval import build_pool, retrieve
    pool=build_pool(['A is an island.'])
    query={'row':0,'claim_id':1,'claim':'A is an island.'}
    for extra in ({'label':'SUPPORTS'},{'evidence':'A is an island.'}):
        with pytest.raises(ValueError,match='claim-only'):
            retrieve([dict(query,**extra)],pool,'unicode_words_cjk_1_2')
    assert retrieve([query],pool,'unicode_words_cjk_1_2')[0]['evidence'][0]['text']=='A is an island.'


def test_score_keeps_nei_target_matching_separate_from_verifiable_recall():
    from apv_rag.multilingual_retrieval import score
    rows=[{'row':0,'claim_id':1,'claim':'A','evidence':[{'id':'P','text':'A','score':1.0}]},
          {'row':1,'claim_id':2,'claim':'B','evidence':[{'id':'Q','text':'B','score':1.0}]}]
    targets=[{'row':0,'claim_id':1,'target_id':'P','label':'SUPPORTS'},
             {'row':1,'claim_id':2,'target_id':'R','label':'NOT ENOUGH INFO'}]
    result=score(rows,targets)
    assert result['verifiable']['queries']==1
    assert result['verifiable']['target_hit_at_3']==1
    assert result['nei_descriptive_only']['target_hit_at_3']==0
    assert result['all_pairs']['target_hit_at_3']==0.5
