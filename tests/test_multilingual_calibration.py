import pytest


def fixture():
    rows=[];gold=[]
    for i in range(10):
        for language in ('en','zh'):
            key=f'{language}:{i}'
            gold.append({'key':key,'claim_id':i,'label':'SUPPORTS'})
            for policy in ('no_gate','nli','embedding','combined'):
                accepted=not (policy=='embedding' and i==0)
                rows.append({'key':key,'file':language+'/test.jsonl','claim_id':i,'policy':policy,
                    'candidate_label':('Supported' if i%2 else 'Refuted') if accepted else None,
                    'uncalibrated_english_nli_label_score':(.9 if i%3 else .6) if accepted else None})
    return rows,gold


def test_all_translations_share_label_blind_fold_and_order_is_irrelevant():
    from apv_rag.multilingual_calibration import claim_folds
    rows,_=fixture();folds=claim_folds(rows)
    assert len(folds)==10 and set(folds.values())==set(range(5))
    assert claim_folds(list(reversed(rows)))==folds
    assert all(sum(f==v for v in folds.values())==2 for f in range(5))


def test_heldout_labels_never_change_their_calibrator_and_abstentions_stay_unavailable():
    from apv_rag.multilingual_calibration import claim_folds,fit_folds,predict_folds
    rows,gold=fixture();folds=claim_folds(rows);fits=fit_folds(rows,gold,folds)
    changed=[dict(g,label='REFUTES') if folds[str(g['claim_id'])]==0 else dict(g) for g in gold]
    refit=fit_folds(rows,changed,folds)
    assert fits['0']==refit['0']
    output=predict_folds(rows,folds,fits)
    assert len(output)==len(rows)
    assert all('true_label' not in r and 'correct' not in r for r in output)
    assert all(r['correctness_confidence'] is None for r in output if r['candidate_label'] is None)
    for i,r in enumerate(output):
        assert r['candidate_label']==rows[i]['candidate_label']
        assert r['claim_id'] not in fits[str(r['fold'])]['development_claim_ids']


def test_crossfit_rejects_missing_keys_and_conflicting_parallel_labels():
    from apv_rag.multilingual_calibration import claim_folds,fit_folds
    rows,gold=fixture();folds=claim_folds(rows)
    with pytest.raises(ValueError):fit_folds(rows,gold[:-1],folds)
    changed=[dict(g) for g in gold];changed[1]['label']='REFUTES'
    with pytest.raises(ValueError,match='Parallel labels'):fit_folds(rows,changed,folds)


def test_empty_training_fold_keeps_confidence_unavailable_and_constant_uses_training_only():
    from apv_rag.multilingual_calibration import claim_folds,fit_folds,predict_folds
    rows,gold=fixture();folds=claim_folds(rows);held=next(i for i,f in folds.items() if f==0)
    for row in rows:
        if row['policy']=='nli' and str(row['claim_id'])!=held:
            row['candidate_label']=None;row['uncalibrated_english_nli_label_score']=None
    fits=fit_folds(rows,gold,folds);predicted=predict_folds(rows,folds,fits)
    assert fits['0']['policies']['nli']['method']=='unavailable_no_accepted_development_answers'
    assert fits['0']['prevalence_controls']['nli']['probability'] is None
    assert all(r['correctness_confidence'] is None and r['prevalence_control_confidence'] is None
        for r in predicted if r['policy']=='nli' and str(r['claim_id'])==held)
    for fold in fits.values():
        for policy,c in fold['prevalence_controls'].items():
            n=c['accepted_development_answers']
            if n:assert c['probability']==(c['correct_development_answers']+1)/(n+2)
