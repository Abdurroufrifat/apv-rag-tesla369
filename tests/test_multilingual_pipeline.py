import pytest


def test_sample_keys_preserve_parallel_files_and_duplicate_claim_rows():
    from apv_rag.multilingual_pipeline import select_sample
    def row(i):return {'row':i,'claim_id':7,'claim':'A is an island.','evidence':[]}
    pools={'en/test.6h.jsonl':[row(0),row(1)],'zh/test.6h.human.jsonl':[row(0),row(1)]}
    selected=select_sample(pools,['7'])
    assert len(selected)==4
    assert len({r['key'] for r in selected})==4
    assert [r['row'] for r in selected]==[0,1,0,1]
    assert selected[-1]['translation_origin']=='upstream_human'
    pools['en/test.6h.jsonl'][0]['true_label']='Supported'
    with pytest.raises(ValueError,match='model-only'):select_sample(pools,['7'])


def test_prepared_context_records_clipping_and_keeps_source_identity():
    from apv_rag.multilingual_pipeline import prepare_context
    source={'claim':'A'*70,'retrieved_evidence':[{'id':'P','text':'B'*100,'score':1.0}]}
    def clip(text,n):return text[:n],len(text),min(n,len(text))
    result=prepare_context(source,clip)
    assert result['shown_claim']=='A'*64
    assert result['evidence'][0]['text']=='B'*96
    assert result['evidence'][0]['id']=='P'
    assert result['clipping']['claim']=={'original':70,'shown':64}
    assert result['clipping']['passages']==[{'id':'P','original':100,'shown':96}]


def test_direct_multilingual_baseline_is_bound_to_context_and_not_substituted_into_gate():
    from apv_rag.multilingual_pipeline import multilingual_entry, execute, direct_prediction
    from apv_rag.fresh_pipeline import feature_entry
    doc={'id':'P','text':'A is an island.','score':1.0}
    prepared={'claim':'A is an island.','shown_claim':'A is an island.',
              'retrieved_evidence':[doc],'evidence':[doc]}
    english=feature_entry(prepared['shown_claim'],[doc],[[0.7,0.2,0.1]],[0.8])
    multi=multilingual_entry(prepared,[[0.1,0.8,0.1]])
    assert direct_prediction(prepared,english,multi)['english_nli']['label']=='Refuted'
    assert direct_prediction(prepared,english,multi)['multilingual_nli']['label']=='Supported'
    def head(intercept):return {'columns':[1],'mean':[0],'scale':[1],'coefficients':[[1]],'intercept':[intercept]}
    heads={'nli':head(-0.5),'embedding':head(2),'combined':head(2)}
    requests=[]
    def generate(kind,question):
        requests.append(kind)
        return {'answer':'Supported' if kind=='verdict' else 'The evidence says A is an island.',
                'prompt':question,'prompt_tokens':20}
    rows=execute(prepared,english,multi,heads,generate)
    assert next(r for r in rows if r['policy']=='nli')['generation_requests']==[]
    assert all(r['correctness_confidence'] is None for r in rows)
    assert 'true_label' not in rows[0]
    broken=dict(multi,context_sha256='bad')
    requests.clear()
    with pytest.raises(ValueError,match='context'):execute(prepared,english,broken,heads,generate)
    assert requests==[]


def test_multilingual_probability_validation_and_empty_retrieval_behavior():
    from apv_rag.multilingual_pipeline import multilingual_entry, direct_prediction
    prepared={'shown_claim':'A','evidence':[{'id':'P','text':'A'}]}
    with pytest.raises(ValueError):multilingual_entry(prepared,[[0.2,0.2,0.2]])
    empty={'shown_claim':'A','evidence':[]}
    result=direct_prediction(empty,None,None)
    assert result['multilingual_nli']['label']=='Not Enough Evidence'
    assert result['multilingual_nli']['probabilities']==[0.0,0.0,1.0]


def saved_controller_fixture(tmp_path):
    import json
    import sqlite3
    from apv_rag.fresh_pipeline import feature_entry
    from apv_rag.multilingual_pipeline import prepare_context,multilingual_entry
    from apv_rag.splits import write_json_atomic
    from run_fresh_pipeline import response_key
    from run_gated_generation import qwen_prompt
    from run_multilingual_retrieved_pipeline import evaluate
    rows=[]
    for language in ('en','zh'):
        rows.append({'key':language+':0','file':language+'/test.6h.jsonl','row':0,'claim_id':7,
            'claim':'A is an island.','language':language,'translation_origin':'original' if language=='en' else 'upstream_machine',
            'retrieved_evidence':[{'id':'P','text':'A is an island.','score':1.0}]})
    rows.append(dict(rows[0],key='en:1',row=1,claim_id=8,retrieved_evidence=[]))
    prepared={r['key']:prepare_context(r,lambda t,n:(t,len(t),len(t))) for r in rows}
    en={k:feature_entry(r['shown_claim'],r['evidence'],[[.1,.8,.1]],[.8])
        for k,r in prepared.items() if r['evidence']}
    ml={k:multilingual_entry(r,[[.8,.1,.1]]) for k,r in prepared.items() if r['evidence']}
    head={'columns':[1],'mean':[0],'scale':[1],'coefficients':[[1]],'intercept':[2]}
    heads={n:head for n in ('nli','embedding','combined')};used={}
    with sqlite3.connect(tmp_path/'generation_cache.sqlite') as db:
        db.execute('CREATE TABLE answers (key TEXT PRIMARY KEY, response TEXT)')
        def generate(kind,question):
            prompt=qwen_prompt(question);key=response_key(kind,prompt)
            response={'answer':'Supported' if kind=='verdict' else 'The evidence says A is an island.',
                'prompt':prompt,'prompt_tokens':40}
            if key not in used:db.execute('INSERT INTO answers VALUES (?,?)',(key,json.dumps(response)))
            used[key]=dict(response,kind=kind,origin='current_run_live_or_resume')
            return response
        policies,direct=evaluate(rows,prepared,en,ml,heads,generate)
    for name,value in [('prepared_contexts',prepared),('feature_cache',en),('multilingual_feature_cache',ml),
                       ('responses_used',used),('predictions',policies),('direct_predictions',direct)]:
        write_json_atomic(tmp_path/(name+'.json'),value)
    gold=[dict(key=r['key'],file=r['file'],row=r['row'],claim_id=r['claim_id'],
        label='NOT ENOUGH INFO' if not r['retrieved_evidence'] else 'SUPPORTS',target_id='P') for r in rows]
    return rows,heads,gold,used


def test_saved_controller_replay_shared_prompts_and_scoring(tmp_path):
    from run_multilingual_retrieved_pipeline import replay_data,score_outputs
    rows,heads,gold,used=saved_controller_fixture(tmp_path)
    policies,direct,responses=replay_data(tmp_path,rows,heads)
    assert len(policies)==12 and len(direct)==3 and len(responses)==2
    assert all(r['correctness_confidence'] is None for r in policies)
    assert 'true_label' not in policies[0]
    summary=score_outputs(policies,direct,gold,used)
    assert summary['queries']==3 and summary['target_fitting'] is False
    assert summary['metrics_by_file']['en/test.6h.jsonl']['policies']['no_gate']['accepted']==1
    assert summary['metrics_by_file']['zh/test.6h.jsonl']['direct_baselines']['english_nli']['accuracy']==1
    assert summary['metrics_by_file']['zh/test.6h.jsonl']['direct_baselines']['multilingual_nli']['accuracy']==0


def test_replay_rejects_altered_answers_and_invented_confidence(tmp_path):
    import json
    import sqlite3
    from run_multilingual_retrieved_pipeline import replay_data
    from apv_rag.splits import write_json_atomic
    rows,heads,_,_=saved_controller_fixture(tmp_path)
    path=tmp_path/'predictions.json';original=json.loads(path.read_text())
    changed=json.loads(path.read_text());changed[0]['correctness_confidence']=.99
    write_json_atomic(path,changed)
    with pytest.raises(ValueError,match='prediction replay'):replay_data(tmp_path,rows,heads)
    write_json_atomic(path,original)
    with sqlite3.connect(tmp_path/'generation_cache.sqlite') as db:
        key,value=db.execute('SELECT key,response FROM answers LIMIT 1').fetchone()
        response=json.loads(value);response['answer']='Refuted'
        db.execute('UPDATE answers SET response=? WHERE key=?',(json.dumps(response),key))
    with pytest.raises(ValueError,match='response conflict'):replay_data(tmp_path,rows,heads)


def test_complete_audit_binds_gold_and_frozen_outputs(tmp_path,monkeypatch):
    import run_multilingual_retrieved_pipeline as runner
    from apv_rag.splits import write_json_atomic,sha256
    out=tmp_path/'run';out.mkdir()
    rows,heads,gold,used=saved_controller_fixture(out)
    source=tmp_path/'sample';source.mkdir()
    write_json_atomic(source/'gold.json',gold)
    targets={}
    for g in gold:
        targets.setdefault(g['file'],[]).append({n:g[n] for n in ('row','claim_id','label','target_id')})
    # The fixture's two files use contiguous row indices just as upstream source arrays do.
    upstream=tmp_path/'artifacts/xfever_retrieval_pool_v1';upstream.mkdir(parents=True)
    write_json_atomic(upstream/'scoring_targets.json',targets)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'identity',lambda ref:{'fixture':True})
    write_json_atomic(out/'input_manifest.json',{'fixture':True})
    write_json_atomic(out/'prediction_manifest.json',runner.prediction_binding(out))
    policies,direct,responses=runner.replay_data(out,rows,heads)
    write_json_atomic(out/'summary.json',runner.score_outputs(policies,direct,gold,responses))
    manifest={'gold_sha256':sha256(source/'gold.json')}
    write_json_atomic(out/'scoring_manifest.json',{'gold_sha256':manifest['gold_sha256'],
        'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),'target_fitting':False})
    def seal():write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.name!='output_manifest.json'})
    seal()
    assert runner.audit(out,source,manifest,{'gate_models':heads},rows)['policy_records']==12
    gold[0]['label']='REFUTES';write_json_atomic(source/'gold.json',gold)
    manifest['gold_sha256']=sha256(source/'gold.json')
    with pytest.raises(ValueError,match='Upstream gold'):runner.audit(out,source,manifest,{'gate_models':heads},rows)
