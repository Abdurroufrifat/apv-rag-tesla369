"""Run/replay the fixed multilingual retrieved-context comparison using local models."""
import argparse
import gc
import json
import sqlite3
import sys
import zipfile
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.direct_nli import TARGET_LABELS
from apv_rag.fever_nli import LABEL_MAP
from apv_rag.fresh_pipeline import POLICIES,context_digest,validate_feature_entry
from apv_rag.integrated_gate import collapse_context
from apv_rag.metrics import classification_metrics
from apv_rag.multilingual_nli import cen_order,normalize_model_scores
from apv_rag.multilingual_pipeline import (select_sample,prepare_context,multilingual_entry,
    validate_multilingual_entry,direct_prediction,execute)
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256
from apv_rag.multilingual_io import write_json_atomic,infer_english
from run_fever_calibration import verify_manifest
from run_fresh_pipeline import make_backend,response_key,CODE as FEATURE_CODE
from run_gated_generation import qwen_prompt
from run_sentence_rag_scifact import summarize
from verify_fresh_pipeline import equal

SAMPLE_SHA='88547c805f92ce6dc88291c14b5a01e2de534714b20bf1fbc2691eeb5c2935d2'
REFERENCE_SHA='6407ad5fd21475dde4c2e8d38909fdf9a5049facf00a08e00ba8bd87f1658fbf'
CODE=tuple(dict.fromkeys((*FEATURE_CODE,'scripts/run_multilingual_retrieved_pipeline.py',
    'src/apv_rag/multilingual_pipeline.py','src/apv_rag/multilingual_io.py','src/apv_rag/direct_nli.py',
    'src/apv_rag/fever_pipeline.py','src/apv_rag/fever_nli.py','src/apv_rag/metrics.py',
    'scripts/run_fever_calibration.py','src/apv_rag/multilingual_generation.py')))
SETTINGS={'seed':369,'threads':4,'dtype':'float32','claim_tokens':64,'passage_tokens':96,
    'nli_tokens':256,'embedding_tokens':384,'gate_threshold':0.5,'max_input_tokens':1024,
    'max_new_tokens':128,'do_sample':False,'num_beams':1,'policies':list(POLICIES),
    'retrieval':'frozen unicode_words_cjk_1_2 positive top3 excerpt pool',
    'gate_features':'original English Deberta CEN + English MiniLM cosines; unchanged heads',
    'multilingual_nli':'separate direct baseline only; never substituted into gate features',
    'ece_bins':15,'risk_coverages':[0.5,0.8,1.0],
    'prior_response_reuse':False,'target_fitting':False,'correctness_calibration':'unavailable'}
LIMITATIONS=[
    'Observed sixty English claim IDs across eleven aligned variants; exploratory transfer, not independent confirmation.',
    'Target-derived closed excerpt pool contains all targets; not full-page or open-web retrieval.',
    'Gate and embedding models remain English trained; multilingual NLI is a separate baseline.',
    'Translated variants share claims and must not be treated as independent samples.',
    'No target tuning or English FEVER confidence calibrator is applied; generated-answer correctness confidence is unavailable.',
    'Numeric checks are lexical and English centric; they do not establish explanation truth, translation quality or localized number equivalence.',
    'Excerpt identity does not authenticate publishers or provenance independence.',
    'Response counts are not measured latency or energy savings; replay does not rerun neural models or recount tokenizer budgets.']


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def require(ok,message):
    if not ok:raise ValueError(message)


def check_manifest(folder,name='output_manifest.json',coverage=False):
    verify_manifest(folder,name)
    if coverage:
        require(set(load(folder/name))=={p.name for p in folder.iterdir() if p.is_file() and p.name!=name},
                'Output manifest coverage differs')


def load_inputs():
    source=ROOT/'data/processed/xfever/retrieved_pipeline_v1'
    reference=ROOT/'config/multilingual_retrieved_pipeline_reference_v1.json'
    require(sha256(source/'selection_manifest.json')==SAMPLE_SHA and sha256(reference)==REFERENCE_SHA,
            'Frozen sample/reference differs')
    manifest=load(source/'selection_manifest.json');ref=load(reference)
    require(sha256(source/'model_inputs.json')==manifest['model_inputs_sha256'],'Model-only input checksum differs')
    rows=load(source/'model_inputs.json')
    retrieved=ROOT/'artifacts/xfever_retrieval_pool_v1'
    require(sha256(retrieved/'output_manifest.json')==manifest['retrieval_output_manifest_sha256'],
            'Upstream retrieval receipt differs')
    check_manifest(retrieved,coverage=True)
    raw=load(retrieved/'predictions.json')['unicode_words_cjk_1_2']
    require(rows==select_sample(raw,manifest['claim_ids']),'Sample/source alignment differs')
    require(len(rows)==660 and len({r['file'] for r in rows})==11 and
            all(sum(r['file']==name for r in rows)==60 for name in raw),'Expected sixty rows per variant')
    return source,manifest,ref,rows


def identity(ref):
    return {'sample_manifest_sha256':SAMPLE_SHA,'reference_sha256':REFERENCE_SHA,
        'packages':ref['packages'],'settings':SETTINGS,'code_sha256':{n:sha256(ROOT/n) for n in CODE},
        'protocol_sha256':sha256(ROOT/'docs/MULTILINGUAL_RETRIEVED_PIPELINE.md')}


ORIGINAL_RUNNER_SHA='3d02c9bfd2aea3cb2828d964b1d8c4ff91cbbba7308efb307a4a4405e1fe64c4'
ORIGINAL_PROTOCOL_SHA='d35a4c85614e760ac3641351fcba62652842356210550bc1d1336e25c74538f8'


def original_identity(meta):
    code=dict(meta['code_sha256'])
    code.pop('src/apv_rag/multilingual_io.py')
    code['scripts/run_multilingual_retrieved_pipeline.py']=ORIGINAL_RUNNER_SHA
    return dict(meta,code_sha256=code,protocol_sha256=ORIGINAL_PROTOCOL_SHA)


def accept_resume(out,meta,rows):
    old=load(out/'input_manifest.json')
    if old==meta:return
    require(old==original_identity(meta),'Run identity changed beyond the exact file-save fix; refusing caches')
    allowed={'input_manifest.json','prepared_contexts.json','feature_cache.json',
        'feature_cache.json.partial','io_resume_receipt.json','io_resume_receipt.json.partial',
        'input_manifest.json.partial'}
    require({p.name for p in out.iterdir()}<=allowed,'File-save migration requires the English-feature stage only')
    prepared=load(out/'prepared_contexts.json');validate_prepared(rows,prepared)
    cache=load(out/'feature_cache.json')
    require(set(cache)<={k for k,r in prepared.items() if r['evidence']},'Interrupted feature keys differ')
    for key,entry in cache.items():validate_feature_entry(entry,prepared[key]['shown_claim'],prepared[key]['evidence'])
    record={'original_input_manifest':old,'change':'bounded PermissionError save retries only; model/settings/sample unchanged',
        'validated_english_feature_records':len(cache),'checkpoint_sha256':{
            n:sha256(out/n) for n in ('prepared_contexts.json','feature_cache.json')}}
    path=out/'io_resume_receipt.json'
    if path.exists():require(load(path)==record,'Interrupted migration receipt differs')
    write_json_atomic(path,record)
    write_json_atomic(out/'input_manifest.json',meta)
    print(f'Validated file-save fix resume: {len(cache)} saved English feature records retained.',flush=True)


def check_resume_receipt(out,meta):
    path=out/'io_resume_receipt.json'
    if not path.exists():return
    record=load(path)
    require(set(record)=={'original_input_manifest','change','validated_english_feature_records','checkpoint_sha256'} and
        record['original_input_manifest']==original_identity(meta) and
        record['change']=='bounded PermissionError save retries only; model/settings/sample unchanged' and
        type(record['validated_english_feature_records']) is int and 0<=record['validated_english_feature_records']<=660 and
        set(record['checkpoint_sha256'])=={'prepared_contexts.json','feature_cache.json'} and
        all(isinstance(h,str) and len(h)==64 and set(h)<=set('0123456789abcdef')
            for h in record['checkpoint_sha256'].values()),'File-save resume receipt differs')


def infer_multilingual(prepared,cache,path):
    pending=[(k,r) for k,r in prepared.items() if r['evidence'] and k not in cache]
    if not pending:return
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
    model_path=ROOT/'models/mdeberta-multilingual-nli'
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        model_path,local_files_only=True,torch_dtype=torch.float32).eval()
    order=cen_order(model.config.id2label)
    for i,(key,r) in enumerate(pending,1):
        texts=[d['text'] for d in r['evidence']]
        tokens=tokenizer(texts,[r['shown_claim']]*len(texts),return_tensors='pt',padding=True,
                         truncation=True,max_length=256)
        with torch.inference_mode():scores=model(**tokens).logits.float().softmax(-1).cpu().numpy()[:,order]
        cache[key]=multilingual_entry(r,normalize_model_scores(scores.tolist()))
        write_json_atomic(path,cache)
        if i==1 or i%10==0 or i==len(pending):print(f'Multilingual NLI: {i}/{len(pending)}',flush=True)


def annotate(row,source,prepared):
    return dict(row,**{n:source[n] for n in ('key','file','row','claim_id','language','translation_origin')},
        claim=source['claim'],shown_claim=prepared['shown_claim'],
        retrieved_evidence=prepared['retrieved_evidence'],clipping=prepared['clipping'],
        context_sha256=context_digest(prepared['shown_claim'],prepared['evidence']))


def evaluate(rows,prepared,en,ml,heads,generate):
    policies=[];direct=[]
    for i,source in enumerate(rows,1):
        key=source['key'];r=prepared[key]
        direct.append(annotate(direct_prediction(r,en.get(key),ml.get(key)),source,r))
        policies.extend(annotate(p,source,r) for p in execute(r,en.get(key),ml.get(key),heads,generate))
        if i==1 or i%10==0 or i==len(rows):print(f'Gate + verdict + explanation: {i}/{len(rows)}',flush=True)
    return policies,direct


def score_outputs(rows,direct,gold,responses):
    keys=[g['key'] for g in gold]
    require([r['key'] for r in direct]==keys and len(set(keys))==len(keys),'Gold/direct alignment differs')
    bykey={g['key']:g for g in gold};files=list(dict.fromkeys(g['file'] for g in gold));metrics={}
    for name in files:
        subset=[r for r in direct if r['file']==name];truth=[LABEL_MAP[bykey[r['key']]['label']] for r in subset]
        metrics[name]={'queries':len(subset),'direct_baselines':{},'policies':{}}
        for method in ('english_nli','multilingual_nli'):
            predicted=[r[method]['label'] for r in subset]
            values=classification_metrics(truth,predicted,np.array([r[method]['probabilities'] for r in subset]),TARGET_LABELS,
                ece_bins=15,coverages=(0.5,0.8,1.0))
            values['accuracy']=float(np.mean([a==b for a,b in zip(truth,predicted,strict=True)]))
            metrics[name]['direct_baselines'][method]=values
        for policy in POLICIES:
            selected=[r for r in rows if r['file']==name and r['policy']==policy]
            require([r['key'] for r in selected]==[r['key'] for r in subset],'Policy/gold alignment differs')
            scored=[dict(r,true_label=LABEL_MAP[bykey[r['key']]['label']]) for r in selected]
            metrics[name]['policies'][policy]={'metrics':summarize(scored,'candidate_label'),
                'accepted':sum(r['candidate_label'] is not None for r in selected),
                'correct':sum(r['candidate_label']==r['true_label'] for r in scored),
                'gate_rejected':sum('gate_below_threshold' in r['reasons'] for r in selected),
                'verdict_requests':sum('verdict' in r['generation_requests'] for r in selected),
                'explanation_requests':sum('explanation' in r['generation_requests'] for r in selected)}
    return {'scope':'exploratory retrieved multilingual controller transfer on a closed excerpt pool',
        'queries':len(direct),'policy_records':len(rows),'distinct_responses':len(responses),
        'metrics_by_file':metrics,'correctness_calibration_available':False,'target_fitting':False,
        'limitations':LIMITATIONS}


def validate_prepared(rows,prepared):
    require(set(prepared)=={r['key'] for r in rows},'Prepared query coverage differs')
    for source in rows:
        r=prepared[source['key']]
        require(set(r)=={'claim','shown_claim','retrieved_evidence','evidence','clipping'} and
                r['claim']==source['claim'] and isinstance(r['shown_claim'],str) and r['shown_claim'].strip(),
                'Prepared claim binding differs')
        counts=r['clipping']
        require(set(counts)=={'claim','passages'} and set(counts['claim'])=={'original','shown'},'Clipping schema differs')
        def valid_count(c,limit):
            return type(c['original']) is int and type(c['shown']) is int and c['original']>=0 and c['shown']==min(c['original'],limit)
        require(valid_count(counts['claim'],64),'Claim clipping counts differ')
        require([d['id'] for d in counts['passages']]==[d['id'] for d in source['retrieved_evidence']],
                'Passage clipping coverage/order differs')
        require(all(set(c)=={'id','original','shown'} and valid_count(c,96) for c in counts['passages']),
                'Passage clipping counts differ')
        originals={d['id']:d for d in source['retrieved_evidence']}
        ids=[d['id'] for d in r['retrieved_evidence']]
        require(len(ids)==len(set(ids)) and ids==[d['id'] for d in source['retrieved_evidence'] if d['id'] in ids],
                'Prepared passage order differs')
        for d in r['retrieved_evidence']:
            require(d['id'] in originals and set(d)==set(originals[d['id']]) and
                    all(d[n]==originals[d['id']][n] for n in d if n!='text') and
                    isinstance(d['text'],str) and d['text'].strip(),'Prepared passage metadata differs')
        require(r['evidence']==collapse_context(r['retrieved_evidence']),'Collapsed context differs')


def replay_data(out,rows,heads):
    prepared=load(out/'prepared_contexts.json');validate_prepared(rows,prepared)
    en=load(out/'feature_cache.json');ml=load(out/'multilingual_feature_cache.json')
    expected={k for k,r in prepared.items() if r['evidence']}
    require(set(en)==set(ml)==expected,'Feature cache coverage differs')
    for key in expected:
        validate_feature_entry(en[key],prepared[key]['shown_claim'],prepared[key]['evidence'])
        validate_multilingual_entry(prepared[key],ml[key])
    responses=load(out/'responses_used.json');used=set()
    with closing(sqlite3.connect(f'file:{(out/"generation_cache.sqlite").resolve().as_posix()}?mode=ro',uri=True)) as db:
        require(db.execute('PRAGMA integrity_check').fetchone()==('ok',),'SQLite integrity differs')
        cached={k:json.loads(v) for k,v in db.execute('SELECT key,response FROM answers')}
    require(set(cached)==set(responses),'SQLite response coverage differs')
    for key,r in responses.items():
        require(set(r)=={'kind','answer','prompt','prompt_tokens','origin'} and
                response_key(r['kind'],r['prompt'])==key and isinstance(r['answer'],str) and
                type(r['prompt_tokens']) is int and 0<r['prompt_tokens']<=1024 and
                r['origin']=='current_run_live_or_resume','Response binding/budget differs')
        require(cached[key]=={n:r[n] for n in ('answer','prompt','prompt_tokens')},'SQLite response conflict')
    def generate(kind,question):
        prompt=qwen_prompt(question);key=response_key(kind,prompt)
        require(key in responses,'Missing requested response')
        used.add(key);return {n:responses[key][n] for n in ('answer','prompt','prompt_tokens')}
    policies,direct=evaluate(rows,prepared,en,ml,heads,generate)
    require(used==set(responses),'Unrequested cached response')
    require(equal(policies,load(out/'predictions.json')) and equal(direct,load(out/'direct_predictions.json')),
            'Controller/direct prediction replay differs')
    return policies,direct,responses


def prediction_binding(out):
    return {name:sha256(out/name) for name in ('prepared_contexts.json','feature_cache.json',
        'multilingual_feature_cache.json','responses_used.json','generation_cache.sqlite',
        'predictions.json','direct_predictions.json')}


def audit(out,source,manifest,ref,rows):
    check_manifest(out,coverage=True)
    require(load(out/'input_manifest.json')==identity(ref),'Producer identity differs')
    check_resume_receipt(out,identity(ref))
    check_manifest(out,'prediction_manifest.json')
    require(load(out/'prediction_manifest.json')==prediction_binding(out),'Frozen prediction coverage differs')
    policies,direct,responses=replay_data(out,rows,ref['gate_models'])
    require(sha256(source/'gold.json')==manifest['gold_sha256'],'Gold checksum differs')
    gold=load(source/'gold.json')
    targets=load(ROOT/'artifacts/xfever_retrieval_pool_v1/scoring_targets.json')
    require(gold==[dict(key=r['key'],file=r['file'],row=r['row'],claim_id=r['claim_id'],
        label=targets[r['file']][r['row']]['label'],target_id=targets[r['file']][r['row']]['target_id']) for r in rows],
        'Upstream gold binding differs')
    summary=score_outputs(policies,direct,gold,responses)
    require(equal(summary,load(out/'summary.json')),'Metric replay differs')
    require(load(out/'scoring_manifest.json')=={'gold_sha256':manifest['gold_sha256'],
        'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),'target_fitting':False},'Scoring binding differs')
    return summary


def export(out):
    path=ROOT/'multilingual_retrieved_pipeline_outputs.zip';temporary=path.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():z.write(p,'multilingual_retrieved_pipeline_v1/'+p.name)
    temporary.replace(path);print(f'Upload this result ZIP: {path}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--verify',action='store_true')
    parser.add_argument('--output',default='artifacts/multilingual_retrieved_pipeline_v1')
    args=parser.parse_args();source,manifest,ref,rows=load_inputs();out=(ROOT/args.output).resolve()
    require(out.is_relative_to(ROOT/'artifacts'),'Output must be inside artifacts')
    if args.preflight:
        print('Inputs passed: 660 queries, eleven variants, original sixty aligned IDs. Neural execution not run.')
        return
    if args.verify or (out/'output_manifest.json').exists():
        audit(out,source,manifest,ref,rows);print('Saved controller/response/metric replay passed; neural models not rerun.')
        export(out);return
    packages={n:version(n) for n in ref['packages']}
    require(packages==ref['packages'],f'Original model environment required: {ref["packages"]}; got {packages}')
    print('Checking existing local model bytes; no downloads.',flush=True)
    for name,path in [('generator','qwen2.5-1.5b-instruct'),('nli','nli-deberta-v3-small'),
                      ('embedding','all-MiniLM-L6-v2'),('multilingual','mdeberta-multilingual-nli')]:
        expected=(ref['generator']['files'] if name=='generator' else ref['multilingual_nli']['files']
                  if name=='multilingual' else ref['feature_models'][name])
        require(_model_files(ROOT/'models'/path)==expected,f'Model checksum differs: {name}')
    out.mkdir(parents=True,exist_ok=True);receipt=out/'input_manifest.json';meta=identity(ref)
    if receipt.exists():accept_resume(out,meta,rows)
    else:require(not any(out.iterdir()),'Cache exists without input identity')
    write_json_atomic(receipt,meta)
    from transformers import AutoTokenizer
    gt=AutoTokenizer.from_pretrained(ROOT/'models/qwen2.5-1.5b-instruct',local_files_only=True)
    def clip(text,limit):
        tokens=gt.encode(text,add_special_tokens=False)
        return gt.decode(tokens[:limit],skip_special_tokens=True),len(tokens),min(len(tokens),limit)
    prepared={r['key']:prepare_context(r,clip) for r in rows};validate_prepared(rows,prepared)
    path=out/'prepared_contexts.json'
    if path.exists():require(load(path)==prepared,'Tokenized context changed; resume refused')
    write_json_atomic(path,prepared);del gt;gc.collect()
    enpath=out/'feature_cache.json';mlpath=out/'multilingual_feature_cache.json'
    en=load(enpath) if enpath.exists() else {};ml=load(mlpath) if mlpath.exists() else {}
    expected={k for k,r in prepared.items() if r['evidence']}
    require(set(en)<=expected and set(ml)<=expected,'Unexpected feature cache keys')
    for key in en:validate_feature_entry(en[key],prepared[key]['shown_claim'],prepared[key]['evidence'])
    for key in ml:validate_multilingual_entry(prepared[key],ml[key])
    infer_english(prepared,en,enpath,root=ROOT);write_json_atomic(enpath,en);gc.collect()
    infer_multilingual(prepared,ml,mlpath);write_json_atomic(mlpath,ml);gc.collect()
    with closing(sqlite3.connect(out/'generation_cache.sqlite')) as db:
        db.execute('CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT)')
        generate,used=make_backend(ROOT,ref['generator'],db,seeds={})
        policies,direct=evaluate(rows,prepared,en,ml,ref['gate_models'],generate)
        write_json_atomic(out/'responses_used.json',used)
    write_json_atomic(out/'predictions.json',policies);write_json_atomic(out/'direct_predictions.json',direct)
    write_json_atomic(out/'prediction_manifest.json',prediction_binding(out))
    # First gold file read follows frozen outputs. No fitting or multilingual confidence calibration.
    require(sha256(source/'gold.json')==manifest['gold_sha256'],'Gold checksum differs')
    summary=score_outputs(policies,direct,load(source/'gold.json'),used)
    write_json_atomic(out/'summary.json',summary)
    write_json_atomic(out/'scoring_manifest.json',{'gold_sha256':manifest['gold_sha256'],
        'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),'target_fitting':False})
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir()
        if p.is_file() and p.name!='output_manifest.json'})
    audit(out,source,manifest,ref,rows)
    print('Multilingual retrieved-context run and non-neural replay passed.',flush=True);export(out)


if __name__=='__main__':main()
