from apv_rag.fresh_copy_confirmation import select_fresh, model_input


def row(i, claim, pages):
    return {'claim_id':str(i), 'claim':claim, 'claim_label':'SUPPORTS',
            'evidences':[{'article':p, 'evidence':'A sentence.', 'evidence_label':'SUPPORTS', 'votes':['SUPPORTS']} for p in pages]}


def test_selection_excludes_transitive_page_overlap_and_is_label_blind():
    rows=[row(1,'Old claim',['A']),row(2,'Bridge claim',['A','B']),row(3,'New linked claim',['B']),row(4,'Fresh claim',['C']),row(5,'old claim',['D'])]
    selected, groups=select_fresh(rows, {'1'}, set(), {'old claim'})
    assert [r['claim_id'] for r in selected] == ['4']
    for r in rows:
        r['claim_label']='DISPUTED'
    again,_=select_fresh(rows, {'1'}, set(), {'old claim'})
    assert [r['claim_id'] for r in again] == ['4']
    assert groups == {'4':'4'}


def test_model_input_hides_all_annotations_and_preserves_sentence_order():
    raw=row(1,'A claim',['Example article'])
    data=model_input(raw)
    assert 'label' not in data and 'claim_label' not in data
    assert data['questions'][0]['answers'][0]['answer']=='A sentence.'
    assert 'SUPPORTS' not in str(data)
    assert raw['evidences'][0]['votes']==['SUPPORTS']
