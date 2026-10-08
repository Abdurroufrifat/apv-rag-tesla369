"""One resumable local-model batch for five fixed text/context stress controls."""
import argparse
import gc
import sqlite3
from importlib.metadata import version

from apv_rag.fresh_pipeline import POLICIES, execute_policies, validate_feature_entry
from apv_rag.generation_robustness import question
from apv_rag.generative_rag import LABELS
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256, write_json_atomic
from apv_rag.text_stress import CONDITIONS, select_ids, build_contexts
from run_fresh_pipeline import ROOT, CODE as FRESH_CODE, PACKAGES, load, require, infer_missing, seed_responses, make_backend
from run_sentence_rag_scifact import summarize
from verify_fresh_pipeline import verify_export

CODE=tuple(dict.fromkeys(('scripts/run_text_stress.py','scripts/verify_text_stress.py',
    'src/apv_rag/text_stress.py','src/apv_rag/generation_robustness.py',*FRESH_CODE)))
SETTINGS={'claims_per_cohort':30,'conditions':list(CONDITIONS),'policies':list(POLICIES),
    'gate_threshold':.5,'ocr_fraction_of_characters':.02,'seed':369,'threads':4,'dtype':'float32',
    'nli_tokens':256,'embedding_tokens':384,'max_input_tokens':1024,'max_new_tokens':128,
    'do_sample':False,'num_beams':1,'baseline_reuse':'verified exact features and responses',
    'changed_contexts':'fresh neural features; exact prompt cache with live fallback',
    'absent_context_scoring':'structural abstention only; no accuracy score'}


def prepare(root=ROOT):
    folder=root/'artifacts/fresh_pipeline_received'
    verify_export(folder,root)
    meta=load(folder,'input_manifest.json')
    rows=[r for r in load(folder,'predictions.json') if r['policy']=='no_gate']
    sources={f"{r['cohort']}:{r['claim_id']}":r for r in rows}
    require(len(rows)==len(sources)==600,'Provider cohort mismatch')
    selected=select_ids(sources,30)
    contexts=build_contexts(sources,selected)
    models=load(root/'artifacts/semantic_sufficiency_received','models.json')
    files={p.relative_to(root).as_posix():sha256(p) for p in folder.glob('*.json')}
    files['artifacts/semantic_sufficiency_received/models.json']=sha256(root/'artifacts/semantic_sufficiency_received/models.json')
    baselines={cohort:{i:sources[f'{cohort}:{i}'] for i in ids} for cohort,ids in selected.items()}
    prior=load(folder,'feature_cache.json')
    initial={f'{cohort}:{i}:baseline':prior[f'{cohort}:{i}'] for cohort,ids in selected.items() for i in ids}
    for key,entry in initial.items():validate_feature_entry(entry,contexts[key]['shown_claim'],contexts[key]['evidence'])
    return {'contexts':contexts,'selected':selected,'models':models,'files':files,'baselines':baselines,
        'initial_features':initial,'generator':meta['generator'],'feature_models':meta['feature_models'],'packages':meta['packages']}


def identity(inputs,root=ROOT):
    return {'schema_version':1,'source_files':inputs['files'],'selected_claim_ids':inputs['selected'],
        'generator':inputs['generator'],'feature_models':inputs['feature_models'],'packages':inputs['packages'],
        'settings':SETTINGS,'code_sha256':{n:sha256(root/n) for n in CODE},
        'protocol_sha256':sha256(root/'docs/TEXT_STRESS.md')}


def execute(inputs,cache,generate):
    rows=[]
    for key,context in inputs['contexts'].items():
        for r in execute_policies(context,cache.get(key),inputs['models'],generate):
            r.update(cohort=context['cohort'],claim_id=context['claim_id'],condition=context['condition'],
                true_label=context['true_label'],context_sha256=cache[key]['context_sha256'] if key in cache else None,
                synthetic_reference_echo=bool(context['synthetic_reference'] and
                    context['synthetic_reference'].casefold() in (r['generated_explanation'] or '').casefold()))
            rows.append(r)
    return rows


def summary(rows,responses):
    lookup={(r['cohort'],r['claim_id'],r['condition'],r['policy']):r for r in rows}
    cohorts={}
    for cohort in ('scifact','climate_retrieved'):
        cohorts[cohort]={}
        for condition in CONDITIONS:
            cohorts[cohort][condition]={}
            for policy in POLICIES:
                subset=[r for r in rows if r['cohort']==cohort and r['condition']==condition and r['policy']==policy]
                base=[lookup[cohort,r['claim_id'],'baseline',policy] for r in subset]
                cohorts[cohort][condition][policy]={'rows':len(subset),
                    'metrics':None if condition=='evidence_absent' else summarize(subset,'candidate_label'),
                    'abstention_rate':sum(r['candidate_label'] is None for r in subset)/len(subset),
                    'gate_rejections':sum('gate_below_threshold' in r['reasons'] for r in subset),
                    'verdict_requests':sum('verdict' in r['generation_requests'] for r in subset),
                    'explanation_requests':sum('explanation' in r['generation_requests'] for r in subset),
                    'verdict_or_abstention_changes':sum(r['candidate_label']!=b['candidate_label'] for r,b in zip(subset,base,strict=True)),
                    'numeric_rejections':sum('numeric_value_absent' in r['reasons'] for r in subset),
                    'synthetic_reference_echoes':sum(r['synthetic_reference_echo'] for r in subset)}
    return {'claims':60,'policy_records':len(rows),'cohorts':cohorts,'distinct_responses_used':len(responses),
        'distinct_prior_responses_reused':sum(r['origin']=='prior_exact_prompt' for r in responses.values()),
        'distinct_current_run_responses':sum(r['origin']!='prior_exact_prompt' for r in responses.values()),
        'scope':'Fixed 30-claim samples per cohort; synthetic context changes, observed development data, no source authentication or confirmatory superiority claim.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    inputs=prepare()
    if args.preflight:
        out=ROOT/'artifacts/text_stress_preflight_v1';out.mkdir(exist_ok=True)
        write_json_atomic(out/'contexts.json',inputs['contexts'])
        write_json_atomic(out/'preflight.json',{'status':'input_and_recipe_checks_passed','claims':60,
            'contexts':360,'policy_records_planned':1440,'reused_baseline_feature_records':60,
            'new_feature_records_planned':240,'selected_claim_ids':inputs['selected'],
            'source_files':inputs['files'],'code_sha256':{n:sha256(ROOT/n) for n in CODE},
            'neural_inference_run':False,'tokenizer_budget_checked':False})
        write_json_atomic(out/'audit_manifest.json',{'files':{p.name:sha256(p) for p in out.iterdir() if p.name!='audit_manifest.json'}})
        print('Text stress preflight passed: 60 claims, 360 contexts, 240 fresh-feature contexts planned.')
        return
    out=ROOT/'artifacts/text_stress_v1'
    require(not (out/'output_manifest.json').exists(),'Completed run exists; verify/export it instead')
    require({n:version(n) for n in PACKAGES}==inputs['packages'],'Original model package profile required')
    for name,relative in [('generator','models/qwen2.5-1.5b-instruct'),('nli','models/nli-deberta-v3-small'),('embedding','models/all-MiniLM-L6-v2')]:
        expected=inputs['generator']['files'] if name=='generator' else inputs['feature_models'][name]
        require(_model_files(ROOT/relative)==expected,f'Pinned model checksum mismatch: {name}')
    out.mkdir(exist_ok=True);receipt=out/'input_manifest.json';meta=identity(inputs)
    if receipt.exists():require(load(out,receipt.name)==meta,'Run identity changed; resume refused')
    else:require(not any((out/n).exists() for n in ('feature_cache.json','generation_cache.sqlite')),'Cache without identity')
    write_json_atomic(receipt,meta)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/qwen2.5-1.5b-instruct',local_files_only=True)
    for context in inputs['contexts'].values():
        if not context['evidence']:continue
        qs=[question(context['shown_claim'],context['evidence'])]+[question(context['shown_claim'],context['evidence'],label) for label in LABELS]
        for q in qs:
            tokens=tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':q}],tokenize=True,add_generation_prompt=True,return_dict=False)
            require(len(tokens)<=1024,'Perturbed prompt exceeds budget; no silent truncation')
    del tokenizer;gc.collect()
    cachepath=out/'feature_cache.json';cache=load(out,cachepath.name) if cachepath.exists() else {}
    eligible={k for k,r in inputs['contexts'].items() if r['evidence']}
    require(set(cache)<=eligible,'Unexpected cached features')
    for k,v in inputs['initial_features'].items():
        require(k not in cache or cache[k]==v,'Changed baseline feature cache')
        cache[k]=v
    for k,v in cache.items():validate_feature_entry(v,inputs['contexts'][k]['shown_claim'],inputs['contexts'][k]['evidence'])
    write_json_atomic(cachepath,cache)
    count=infer_missing(inputs['contexts'],cache,cachepath);gc.collect()
    with sqlite3.connect(out/'generation_cache.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY,response TEXT)')
        backend,used=make_backend(ROOT,inputs['generator'],db,seed_responses(inputs['baselines']))
        calls=0
        def generate(kind,q):
            nonlocal calls
            response=backend(kind,q);calls+=1
            if calls%20==0:print(f'Controller response requests: {calls}',flush=True)
            return response
        rows=execute(inputs,cache,generate)
    write_json_atomic(out/'contexts.json',inputs['contexts'])
    write_json_atomic(out/'execution_progress.json',{'new_feature_records_this_invocation':count,
        'total_feature_records':len(cache),'baseline_feature_records_reused':60,'tokenizer_budget_checked':True})
    write_json_atomic(out/'predictions.json',rows);write_json_atomic(out/'responses.json',used)
    write_json_atomic(out/'summary.json',summary(rows,used))
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.glob('*.json') if p.name!='output_manifest.json'})
    print('Text stress batch completed.');print(out)


if __name__=='__main__':main()
