import copy
from apv_rag.exact_copy_removal import remove_exact_source_copies


def test_removes_identical_source_answers_but_preserves_distinct_evidence():
    a={'answer':'First sentence','source_url':'https://example.org/a','answer_type':'Extractive'}
    b=dict(a,answer='Second sentence');c=dict(a,source_url='https://example.org/b')
    row={'claim':'Claim','questions':[{'question':'Question','answers':[a,copy.deepcopy(a),b,c]}]}
    original=copy.deepcopy(row)
    assert remove_exact_source_copies(row)['questions'][0]['answers']==[a,b,c]
    assert row==original


def test_missing_source_and_different_question_answers_are_preserved():
    a={'answer':'Sentence','source_url':'https://example.org/a'};unknown={'answer':'Sentence'}
    row={'questions':[{'answers':[a,unknown,copy.deepcopy(unknown)]},{'answers':[copy.deepcopy(a)]}]}
    assert remove_exact_source_copies(row)==row
