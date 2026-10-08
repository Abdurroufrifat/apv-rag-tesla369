"""Freeze, score and verify a claim/page-disjoint supplied-evidence comparison."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from apv_rag.fresh_copy_confirmation import select_fresh, model_input, normalized
from apv_rag.evidence_baseline import LABELS
from apv_rag.repetition_stress import inject_derivative_copies
from apv_rag.splits import sha256, write_json_atomic

OUT=ROOT/'artifacts/fresh_copy_confirmation_v1'
SOURCE=ROOT/'data/external/climate_fever/frozen_v1'
MODELS=ROOT/'artifacts/trained_copy_robustness_v1'
MAPPING={'SUPPORTS':LABELS[0],'REFUTES':LABELS[1],'NOT_ENOUGH_INFO':LABELS[2],'DISPUTED':LABELS[3]}
COPIES=(0,1,5,10,25)

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]

def freeze():
    if OUT.exists():
        raise FileExistsError('Frozen cohort exists; refusing replacement')
    manifest=read(SOURCE/'manifest.json')
    if sha256(SOURCE/'climate-fever.jsonl')!=manifest['raw_sha256']:
        raise ValueError('Pinned climate source differs')
    files=[SOURCE/'climate-fever.jsonl',SOURCE/'manifest.json',SOURCE/'corpus.jsonl']
    rows=lines(files[0]); prior_ids={str(i) for i in manifest['cohort']}
    texts=set(); pages=set()
    # Exclude all earlier benchmark input cohorts, not the unused raw FEVER pool.
    previous=list((ROOT/'data/external/fever').rglob('model_inputs.jsonl'))
    previous+=list((ROOT/'data/external/fever').rglob('gold.jsonl'))
    previous+=list((ROOT/'data/external/xfever/zenodo_8206962').glob('*inputs_v1/en/*.jsonl'))
    previous+=[ROOT/'data/external/scifact/sealed_v1/claims_dev.jsonl']
    for path in previous:
        files.append(path)
        for r in lines(path):
            if 'claim' in r:texts.add(r['claim'])
            if 'page' in r:pages.add(r['page'])
            evidence=r.get('evidence')
            if isinstance(evidence,list):
                for group in evidence:
                    for e in group:
                        if len(e)>2 and isinstance(e[2],str):pages.add(e[2])
    for name in ('train','dev'):
        path=ROOT/f'data/external/averitec/official_7c62d1e/{name}.json';files.append(path)
        for r in read(path):
            texts.add(r['claim'])
            for q in r.get('questions',[]):
                for a in q.get('answers',[]):
                    url=urlsplit(str(a.get('source_url') or ''))
                    if url.hostname=='en.wikipedia.org' and url.path.startswith('/wiki/'):
                        pages.add(unquote(url.path[6:]))
    corpus={r['doc_id']:r['title'] for r in lines(SOURCE/'corpus.jsonl')}
    # Previously retrieved passages count as observed, even without their gold labels.
    for name in ('climate_rag_received','climate_supplied_received','climate_decoder_received'):
        path=ROOT/f'artifacts/{name}/predictions.json';files.append(path)
        for r in read(path):
            if 'claim' in r:texts.add(r['claim'])
            if 'claim_id' in r:prior_ids.add(str(r['claim_id']))
            for e in r.get('evidence',[]):
                if 'article' in e:pages.add(e['article'])
                elif isinstance(e.get('id'),int):pages.add(corpus[e['id']])
            for i in r.get('context_source_ids',[]):
                if isinstance(i,int):pages.add(corpus[i])
    selected,groups=select_fresh(rows,prior_ids,pages,texts)
    if not selected:raise ValueError('No disjoint cohort remains')
    files+=list(MODELS.glob('*.joblib'))+[MODELS/'receipt.json',MODELS/'training_receipt.json']
    code=[Path(__file__).resolve(),ROOT/'src/apv_rag/fresh_copy_confirmation.py',ROOT/'src/apv_rag/repetition_stress.py',ROOT/'src/apv_rag/trained_copy_robustness.py',ROOT/'src/apv_rag/provenance_model.py',ROOT/'src/apv_rag/disagreement_model.py',ROOT/'src/apv_rag/evidence_baseline.py']
    OUT.mkdir()
    write_json_atomic(OUT/'model_inputs.json',[{'claim_id':r['claim_id'],'record':model_input(r)} for r in selected])
    write_json_atomic(OUT/'gold.json',[{'claim_id':r['claim_id'],'label':MAPPING[r['claim_label']]} for r in selected])
    write_json_atomic(OUT/'groups.json',groups)
    write_json_atomic(OUT/'protocol.json',{
        'scope':'Fresh supplied-evidence component confirmation on a previously observed public dataset; not full RAG or an official blind test.',
        'selection':'All remaining unique normalized claims, excluding complete article-connected components touching earlier climate claims, observed retrieved pages, prior benchmark pages, exact prior claim text or token-set Jaccard >=0.8.',
        'selection_uses_gold_labels':False,'selected_ids':[r['claim_id'] for r in selected],
        'claims':len(selected),'article_connected_groups':len(set(groups.values())),
        'models':['control','copy_augmented'],'copies':list(COPIES),'mapping':MAPPING,
        'decision':'Four-way argmax; no tuning, calibration, refitting or deployment selection',
        'primary':'Paired macro-F1 difference at 25 copies; clean difference secondary',
        'uncertainty':'2000 article-connected group bootstrap draws, seed 369, fixed four-label macro-F1; descriptive 95% percentile intervals',
        'limitations':['Small cohort and domain/format shift','Annotation-associated supplied sentences; no retrieval','Source identity not authentication','Near-duplicate exclusion is heuristic','Dataset inspected earlier; pretraining overlap unknown','Single training seed; confidence uncalibrated'],
        'inputs':{p.relative_to(ROOT).as_posix():sha256(p) for p in files},
        'code':{p.relative_to(ROOT).as_posix():sha256(p) for p in code}})
    write_json_atomic(OUT/'freeze_receipt.json',{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()})
    print('Frozen without scoring:',len(selected),'claims;',len(set(groups.values())),'article-connected groups.')

def check_freeze():
    protocol=read(OUT/'protocol.json')
    for group in ('inputs','code'):
        for name,digest in protocol[group].items():
            if sha256(ROOT/name)!=digest:raise ValueError(f'{group} SHA mismatch: {name}')
    for name,digest in read(OUT/'freeze_receipt.json').items():
        if sha256(OUT/name)!=digest:raise ValueError(f'Frozen output SHA mismatch: {name}')
    return protocol

def score(predictions):
    gold=read(OUT/'gold.json');groups=read(OUT/'groups.json');ids=[r['claim_id'] for r in gold]
    expected={(m,k,i) for m in ('control','copy_augmented') for k in COPIES for i in ids}
    lookup={(r['model'],r['copies'],r['claim_id']):r for r in predictions}
    if len(lookup)!=len(predictions) or set(lookup)!=expected:raise ValueError('Prediction coverage differs')
    truth={r['claim_id']:LABELS.index(r['label']) for r in gold}
    matrices={}; summary={}
    unique=sorted(set(groups.values()));groupidx={g:i for i,g in enumerate(unique)}
    def macro(cm):
        d=cm.sum(axis=-1)+cm.sum(axis=-2)
        return np.divide(2*np.diagonal(cm,axis1=-2,axis2=-1),d,out=np.zeros_like(d,dtype=float),where=d!=0).mean(axis=-1)
    for model in ('control','copy_augmented'):
        for k in COPIES:
            cm=np.zeros((len(unique),4,4),dtype=int);shifts=[]
            for i in ids:
                r=lookup[model,k,i];p=np.asarray(r['probabilities'])
                if p.shape!=(4,) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or abs(p.sum()-1)>1e-10:
                    raise ValueError('Invalid probabilities')
                cm[groupidx[groups[i]],truth[i],int(p.argmax())]+=1
                shifts.append(float(p[0]-lookup[model,0,i]['probabilities'][0]))
            total=cm.sum(axis=0);matrices[model,k]=cm
            summary[f'{model}:copies_{k}']={'macro_f1':float(macro(total)),'accuracy':float(np.trace(total)/len(ids)),
                'mean_absolute_support_probability_shift':float(np.mean(np.abs(shifts))),
                'mean_signed_support_probability_shift':float(np.mean(shifts)), 'confusion_matrix':total.tolist()}
    draws=np.random.default_rng(369).integers(0,len(unique),size=(2000,len(unique)))
    uncertainty={}
    for k in (0,25):
        delta=macro(matrices['copy_augmented',k][draws].sum(axis=1))-macro(matrices['control',k][draws].sum(axis=1))
        uncertainty[f'copies_{k}']={'macro_f1_difference':summary[f'copy_augmented:copies_{k}']['macro_f1']-summary[f'control:copies_{k}']['macro_f1'],
             'group_bootstrap_95_percent_interval':np.quantile(delta,[.025,.975]).tolist()}
    return {'claims':len(ids),'groups':len(unique),'label_counts':dict(Counter(r['label'] for r in gold)), 'metrics':summary,'paired_uncertainty':uncertainty}

def evaluate():
    protocol=check_freeze()
    if (OUT/'evaluation_receipt.json').exists():raise FileExistsError('Completed evaluation exists; use --verify')
    rows=read(OUT/'model_inputs.json');predictions=[]
    with threadpool_limits(1):
        for name in protocol['models']:
            model=joblib.load(MODELS/f'{name}.joblib')
            columns=[list(model.classes_).index(label) for label in LABELS]
            for k in COPIES:
                data=[inject_derivative_copies(r['record'],k) for r in rows]
                for original,modified in zip(rows,data,strict=True):
                    if len(modified['questions'][0]['answers'])!=len(original['record']['questions'][0]['answers'])+k:
                        raise ValueError('Copy intervention failed')
                probs=model.predict_proba(data)[:,columns]
                predictions.extend({'model':name,'copies':k,'claim_id':r['claim_id'],'probabilities':p.tolist()} for r,p in zip(rows,probs,strict=True))
    write_json_atomic(OUT/'predictions.json',predictions)
    write_json_atomic(OUT/'summary.json',score(predictions))
    write_json_atomic(OUT/'evaluation_receipt.json',{'freeze_receipt_sha256':sha256(OUT/'freeze_receipt.json'),
        'packages':{n:version(n) for n in ('numpy','scikit-learn','scipy','joblib')},
        'outputs':{n:sha256(OUT/n) for n in ('predictions.json','summary.json')}})
    print(json.dumps(read(OUT/'summary.json'),indent=2))

def verify():
    check_freeze();receipt=read(OUT/'evaluation_receipt.json')
    if sha256(OUT/'freeze_receipt.json')!=receipt['freeze_receipt_sha256']:raise ValueError('Freeze receipt differs')
    for name,digest in receipt['outputs'].items():
        if sha256(OUT/name)!=digest:raise ValueError(f'Output SHA mismatch: {name}')
    if read(OUT/'summary.json')!=score(read(OUT/'predictions.json')):raise ValueError('Recomputed summary differs')
    print('Fresh component results verified: source/code/model hashes, prediction coverage, metrics and grouped uncertainty.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    for name in ('freeze','evaluate','verify'):group.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.evaluate:evaluate()
    else:verify()
