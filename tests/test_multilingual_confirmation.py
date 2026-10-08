import pytest


def test_prior_inventory_paths_are_identical_on_windows_and_linux(monkeypatch):
    from pathlib import Path,PureWindowsPath
    from prepare_multilingual_confirmation import previous_inputs
    expected=previous_inputs()
    relative=Path.relative_to
    def windows_relative(path,*args,**kwargs):
        return PureWindowsPath(relative(path,*args,**kwargs))
    monkeypatch.setattr(Path,'relative_to',windows_relative)
    actual=previous_inputs()
    assert actual==expected
    assert all('\\' not in name for name in actual['files'])


def sets():
    return {f'{lang}/test.jsonl':[
        {'id':i,'claim':f'Claim {i}', 'page':f'Page {i}',
         'evidence':f'{lang} Evidence {i}','label':'SUPPORTS'} for i in range(8)]
        for lang in ('en','es','fr','id','ja','zh')}


def test_selection_is_label_blind_and_keeps_parallel_rows_together():
    from apv_rag.multilingual_confirmation import select_new
    original=sets(); selected=select_new(original,set(),set(),set(),set(),4)
    changed={k:[dict(r,label='REFUTES') for r in v] for k,v in original.items()}
    assert selected==select_new(changed,set(),set(),set(),set(),4)
    assert len({original['en/test.jsonl'][i]['id'] for i in selected})==4
    broken=sets();broken['ja/test.jsonl'][0]['id']=99
    with pytest.raises(ValueError,match='alignment'):select_new(broken,set(),set(),set(),set(),4)


def test_selection_excludes_used_ids_texts_pages_excerpts_and_shared_new_pages():
    from apv_rag.multilingual_confirmation import select_new,page_key
    from apv_rag.multilingual_retrieval import normalize,excerpt_id
    original=sets()
    for v in original.values():v[5]['page']=v[4]['page']
    chosen=select_new(original,{0},{normalize('Claim 1')},{page_key('Page 2')},
        {excerpt_id('zh Evidence 3')},3)
    assert {original['en/test.jsonl'][i]['id'] for i in chosen} <= {4,5,6,7}
    assert len({page_key(original['en/test.jsonl'][i]['page']) for i in chosen})==3
    with pytest.raises(ValueError,match='eligible'):select_new(original,set(range(8)),set(),set(),set(),1)


def test_frozen_calibrators_cannot_evaluate_development_claims_and_preserve_abstention():
    from apv_rag.multilingual_confirmation import fit_development,apply_frozen
    rows=[];gold=[]
    for i in range(6):
        gold.append({'key':str(i),'claim_id':i,'label':'SUPPORTS'})
        for policy in ('no_gate','nli','embedding','combined'):
            rows.append({'key':str(i),'file':'en/test.jsonl','claim_id':i,'policy':policy,
                'candidate_label':'Supported' if i%2 else 'Refuted',
                'uncalibrated_english_nli_label_score':.6 if i%2 else .9})
    fits=fit_development(rows,gold)
    with pytest.raises(ValueError,match='development'):apply_frozen(rows,fits)
    new=[dict(r,claim_id=r['claim_id']+100,key='new'+r['key']) for r in rows]
    new[0]['candidate_label']=None;new[0]['uncalibrated_english_nli_label_score']=None
    predictions=apply_frozen(new,fits)
    assert predictions[0]['correctness_confidence'] is None
    assert predictions[0]['prevalence_control_confidence'] is None
    assert all('label' not in r and 'correct' not in r for r in predictions)
    assert all(r['candidate_label']==new[i]['candidate_label'] for i,r in enumerate(predictions))


def test_confirmation_audit_replays_confidence_and_rejects_even_rehashed_corruption(tmp_path,monkeypatch):
    import contextlib
    import io
    import json
    import sqlite3
    import run_multilingual_confirmation as runner
    from apv_rag.multilingual_confirmation import FILES,fit_development,apply_frozen
    from apv_rag.multilingual_pipeline import prepare_context,multilingual_entry
    from apv_rag.fresh_pipeline import feature_entry
    from apv_rag.multilingual_retrieval import excerpt_id
    from apv_rag.splits import sha256,write_json_atomic
    from run_fresh_pipeline import response_key
    from run_gated_generation import qwen_prompt
    source=tmp_path/'source';source.mkdir();out=tmp_path/'run';out.mkdir()
    monkeypatch.setattr(runner,'OUT',source)
    monkeypatch.setattr(runner,'identity',lambda manifest,ref:{'test_identity':True})
    doc={'id':excerpt_id('Evidence supports the claim.'),'text':'Evidence supports the claim.','score':1.0}
    inputs=[{'key':name+':0','file':name,'row':0,'claim_id':100,'claim':'A claim.',
        'language':name.split('/')[0],'translation_origin':'original' if name.startswith('en/') else 'upstream_machine',
        'retrieved_evidence':[doc]} for name in FILES]
    def clip(text,limit):
        words=text.split();return ' '.join(words[:limit]),len(words),min(len(words),limit)
    prepared={r['key']:prepare_context(r,clip) for r in inputs}
    english={k:feature_entry(r['shown_claim'],r['evidence'],[[.1,.8,.1]],[.8]) for k,r in prepared.items()}
    multi={k:multilingual_entry(r,[[.1,.8,.1]]) for k,r in prepared.items()}
    model={'columns':list(range(13)),'mean':[0.]*13,'scale':[1.]*13,
        'coefficients':[[0.]*13],'intercept':[0.]}
    ref={'gate_models':{p:model for p in ('nli','embedding','combined')}};responses={}
    def generate(kind,question):
        prompt=qwen_prompt(question);key=response_key(kind,prompt)
        response={'answer':'Supported' if kind=='verdict' else 'Evidence supports the claim.',
            'prompt':prompt,'prompt_tokens':1}
        responses[key]=response|{'kind':kind,'origin':'current_run_live_or_resume'}
        return response
    with contextlib.redirect_stdout(io.StringIO()):rows,direct=runner.evaluate(inputs,prepared,english,multi,ref['gate_models'],generate)
    gold=[{k:r[k] for k in ('key','file','row','claim_id')}|{'label':'SUPPORTS','target_id':doc['id']} for r in inputs]
    development=[dict(r,claim_id=i,key=str(i)) for i in range(6) for r in rows[:4]]
    fits=fit_development(development,[{'key':str(i),'claim_id':i,'label':'SUPPORTS'} for i in range(6)])
    confidence=apply_frozen(rows,fits)
    with sqlite3.connect(out/'generation_cache.sqlite') as db:
        db.execute('CREATE TABLE answers (key TEXT PRIMARY KEY,response TEXT)')
        for key,r in responses.items():db.execute('INSERT INTO answers VALUES (?,?)',(key,json.dumps({k:r[k] for k in ('answer','prompt','prompt_tokens')})))
    for name,value in [('input_manifest.json',{'test_identity':True}),('prepared_contexts.json',prepared),
        ('feature_cache.json',english),('multilingual_feature_cache.json',multi),('responses_used.json',responses),
        ('predictions.json',rows),('direct_predictions.json',direct),('correctness_predictions.json',confidence)]:
        write_json_atomic(out/name,value)
    write_json_atomic(source/'gold.json',gold)
    write_json_atomic(out/'summary.json',runner.score(rows,direct,confidence,gold,responses))
    def receipts():
        write_json_atomic(out/'prediction_manifest.json',runner.binding(out))
        write_json_atomic(out/'scoring_manifest.json',{'gold_sha256':sha256(source/'gold.json'),
            'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),'confirmation_fitting':False})
        write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.name!='output_manifest.json'})
    receipts();result=runner.audit(out,{},inputs,ref,fits)
    assert result['queries']==6
    assert 'frozen_calibrator_metrics' in result['correctness_calibration']['pooled_descriptive']['no_gate']
    confidence[0]['correctness_confidence']=.001
    write_json_atomic(out/'correctness_predictions.json',confidence);receipts()
    with pytest.raises(ValueError,match='Frozen confidence'):runner.audit(out,{},inputs,ref,fits)
