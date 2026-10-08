import pytest
from apv_rag.fresh_rag_cohort import build_cohort


def test_cohort_keeps_three_supported_classes_and_hides_gold():
    inputs=[{'claim_id':str(i),'record':{'claim':f'Claim {i}'}} for i in range(4)]
    gold=[{'claim_id':str(i),'label':label} for i,label in enumerate(('Supported','Refuted','Not Enough Evidence','Conflicting Evidence/Cherrypicking'))]
    claims,targets,excluded=build_cohort(inputs,gold)
    assert claims==[{'id':str(i),'claim':f'Claim {i}'} for i in range(3)]
    assert targets=={str(i):gold[i]['label'] for i in range(3)}
    assert excluded==['3']


def test_cohort_rejects_missing_or_duplicate_gold_ids():
    inputs=[{'claim_id':'1','record':{'claim':'Claim'}}]
    with pytest.raises(ValueError):build_cohort(inputs,[])
    with pytest.raises(ValueError):build_cohort(inputs,[{'claim_id':'1','label':'Supported'}]*2)
