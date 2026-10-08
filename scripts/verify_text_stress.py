"""Bind changed-context model exports to the fixed recipe and frozen controller."""
import argparse

from apv_rag.fresh_pipeline import validate_feature_entry
from apv_rag.splits import sha256,write_json_atomic
from run_fresh_pipeline import ROOT,check_receipt,load,require,response_key,seed_responses
from run_gated_generation import qwen_prompt
from run_text_stress import prepare,identity,execute,summary
from verify_fresh_pipeline import equal

REQUIRED={'input_manifest.json','contexts.json','execution_progress.json','feature_cache.json',
          'predictions.json','responses.json','summary.json'}


def verify_records(inputs,cache,rows,responses):
    eligible={k for k,c in inputs['contexts'].items() if c['evidence']}
    require(set(cache)==eligible,'Feature coverage mismatch')
    for key,entry in cache.items():
        c=inputs['contexts'][key]
        validate_feature_entry(entry,c['shown_claim'],c['evidence'])
    for key,entry in inputs['initial_features'].items():require(cache[key]==entry,'Baseline feature reuse mismatch')
    seeds=seed_responses(inputs['baselines']);used=set()
    for key,r in responses.items():
        require(set(r)=={'kind','answer','prompt','prompt_tokens','origin'} and
                response_key(r['kind'],r['prompt'])==key and isinstance(r['answer'],str) and
                type(r['prompt_tokens']) is int and 0<r['prompt_tokens']<=1024,'Invalid response binding/budget')
        if key in seeds:
            require(r['origin']=='prior_exact_prompt' and
                    {n:r[n] for n in ('answer','prompt','prompt_tokens')}==seeds[key],'Baseline response changed')
        else:require(r['origin']=='current_run_live_or_resume','Current response provenance invalid')
    def generate(kind,q):
        prompt=qwen_prompt(q);key=response_key(kind,prompt)
        require(key in responses,'Missing requested response')
        used.add(key)
        return {n:responses[key][n] for n in ('answer','prompt','prompt_tokens')}
    expected=execute(inputs,cache,generate)
    key=lambda r:(r['cohort'],str(r['claim_id']),r['condition'],r['policy'])
    lookup={key(r):r for r in rows}
    require(len(rows)==len(lookup)==len(expected),'Duplicate or missing policy rows')
    require(set(lookup)=={key(r) for r in expected},'Policy identity mismatch')
    for r in expected:require(equal(lookup[key(r)],r),f'Policy/text/guard mismatch: {key(r)}')
    require(used==set(responses),'Unused response in export')
    return expected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',default='artifacts/text_stress_v1');args=parser.parse_args()
    folder=(ROOT/args.folder).resolve()
    require(folder.is_relative_to((ROOT/'artifacts').resolve()) and folder!=(ROOT/'artifacts').resolve(),'Export must be inside artifacts')
    require(set(check_receipt(folder))==REQUIRED,'Output receipt coverage mismatch')
    inputs=prepare()
    require(load(folder,'input_manifest.json')==identity(inputs),'Input/model/code/protocol identity mismatch')
    require(equal(load(folder,'contexts.json'),inputs['contexts']),'Context recipe, donor or label changed')
    cache=load(folder,'feature_cache.json');responses=load(folder,'responses.json')
    rows=verify_records(inputs,cache,load(folder,'predictions.json'),responses)
    result=summary(rows,responses)
    require(equal(load(folder,'summary.json'),result),'Summary mismatch')
    progress=load(folder,'execution_progress.json')
    require(set(progress)=={'new_feature_records_this_invocation','total_feature_records','baseline_feature_records_reused','tokenizer_budget_checked'} and
            type(progress['new_feature_records_this_invocation']) is int and 0<=progress['new_feature_records_this_invocation']<=240 and
            progress['total_feature_records']==300 and progress['baseline_feature_records_reused']==60 and
            progress['tokenizer_budget_checked'] is True,'Progress declaration mismatch')
    out=ROOT/'artifacts/text_stress_verification_v1';out.mkdir(exist_ok=True)
    write_json_atomic(out/'verification.json',{'status':'changed_context_export_checks_passed',
        'source_output_manifest_sha256':sha256(folder/'output_manifest.json'),'summary':result,
        'independent_neural_inference':False,'independent_tokenizer_recount':False,
        'scope':'Small fixed synthetic stress sample; donor context is not guaranteed unrelated. Citation echoes are literal matches, not source authentication. Evidence absence scores abstention only.'})
    (out/'RESULTS.md').write_text('# Changed-context stress export verification\n\n1440 policy records passed fixed-recipe, source, feature, prompt, guard and metric checks.\n\nOnly 30 claims per cohort are evaluated. These are synthetic sensitivity tests on observed development claims; no source authentication, explanation truth, calibration or superiority claim is established. Missing evidence is evaluated through abstention, with no benchmark accuracy score. Model inference, tokenizer counts, weights and SQLite were not independently rerun here. Detailed metrics are in verification.json.\n',encoding='utf-8')
    write_json_atomic(out/'audit_manifest.json',{'source_output_manifest_sha256':sha256(folder/'output_manifest.json'),
        'verifier_sha256':sha256(ROOT/'scripts/verify_text_stress.py'),'files':{p.name:sha256(p) for p in out.iterdir() if p.name!='audit_manifest.json'}})
    print('Text stress export verified: 60 claims, 1440 policy records.')


if __name__=='__main__':main()
