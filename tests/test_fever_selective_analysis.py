"""Controls must use scores, preserve abstentions and account for shared pages."""
import pytest


def test_selectors_do_not_use_gold_and_respect_accepted_population():
    from apv_rag.fever_selective_analysis import select_control
    rows = [{'claim_id':1,'candidate_label':'Supported','score':0.8},
            {'claim_id':2,'candidate_label':'Refuted','score':0.8},
            {'claim_id':3,'candidate_label':None,'score':None},
            {'claim_id':4,'candidate_label':'Supported','score':0.1}]
    cache = {'1':{'cosines':[0.2]},'2':{'cosines':[0.9]},'4':{'cosines':[0.3]}}
    assert select_control(rows,cache,2,'generated_label_nli') == [0,1]
    assert select_control(rows,cache,2,'mean_embedding_cosine') == [1,3]
    assert select_control(rows,cache,0,'generated_label_nli') == []
    with pytest.raises(ValueError):select_control(rows,cache,4,'generated_label_nli')


def test_shared_page_groups_are_transitive_and_group_test_does_not_treat_copies_as_independent():
    from apv_rag.fever_selective_analysis import page_groups, grouped_effect
    rows = [{'claim':'a','evidence':[{'id':'P'}]},
            {'claim':'b','evidence':[{'id':'P'},{'id':'Q'}]},
            {'claim':'c','evidence':[{'id':'Q'}]},
            {'claim':'d','evidence':[{'id':'R'}]}]
    groups = page_groups(rows)
    assert groups == [[0,1,2],[3]]
    result = grouped_effect([1,1,1,0],groups)
    assert result['all_claim_accuracy_difference'] == 0.75
    assert result['two_sided_group_swap_p'] == 1.0


def test_matched_analysis_counts_only_selected_correct_answers_and_preserves_labels():
    from apv_rag.fever_selective_analysis import analyze
    control = [{'claim_id':i,'claim':str(i),'evidence':[{'id':str(i)}],
                'candidate_label':label,'score':score} for i,label,score in
                [(1,'Supported',0.9),(2,'Supported',0.8),(3,'Refuted',0.2),(4,'Refuted',0.1)]]
    rows = []
    for r in control:
        for policy in ('no_gate','nli','embedding','combined'):
            rows.append(dict(r,policy=policy,candidate_label=r['candidate_label'] if policy=='no_gate' or r['claim_id'] in (1,3) else None))
    gold = [{'id':1,'label':'SUPPORTS'},{'id':2,'label':'REFUTES'},
            {'id':3,'label':'REFUTES'},{'id':4,'label':'SUPPORTS'}]
    cache={str(i):{'cosines':[float(i)/5]} for i in range(1,5)}
    result=analyze(rows,gold,cache)
    first=result['comparisons'][0]
    assert first['accepted_answers']==2
    assert first['gate_correct']==2
    assert first['control_correct']==1
    assert first['gate_accepted_accuracy']==1
    assert first['control_accepted_accuracy']==0.5
    assert first['gate_selected_ids']==[1,3]
    assert first['control_selected_ids']==[1,2]
    broken=[dict(r,candidate_label='Not Enough Evidence') if r['policy']=='nli' and r['claim_id']==1 else r for r in rows]
    with pytest.raises(ValueError,match='common answer'):analyze(broken,gold,cache)
