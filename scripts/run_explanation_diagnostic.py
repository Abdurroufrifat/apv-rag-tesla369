"""Machine-only, label-free explanation-to-passage NLI diagnostic on saved outputs."""
import argparse
import hashlib
import json
import math
import re
from importlib.metadata import version
from pathlib import Path

from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256,write_json_atomic

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=('ocr_noise','fabricated_citation','authoritative_wording','swapped_context')
CODE=('scripts/run_explanation_diagnostic.py','scripts/verify_explanation_diagnostic.py')


def require(ok,message):
    if not ok:raise ValueError(message)


def receipt(folder):
    hashes=json.loads((folder/'output_manifest.json').read_text(encoding='utf-8'))
    for name,digest in hashes.items():
        require(Path(name).name==name and sha256(folder/name)==digest,'Received file hash mismatch')
    return sha256(folder/'output_manifest.json')


def norm(value):return ' '.join(value.casefold().split())


def quote_check(explanation,claim,evidence):
    quotes=re.findall(r'[“\"]([^“”\"]+)[”\"]',explanation)
    result={'quoted':len(quotes),'found_in_evidence':0,'found_in_claim_only':0,'unmatched':0}
    for quote in quotes:
        q=norm(quote)
        if q and q in norm(evidence):result['found_in_evidence']+=1
        elif q and q in norm(claim):result['found_in_claim_only']+=1
        else:result['unmatched']+=1
    return result


def select_tasks(fresh,stress):
    tasks={}
    for row in fresh:
        if row['policy']!='no_gate' or not row['generated_explanation']:continue
        key=f"{row['cohort']}:{row['claim_id']}:original"
        if key in tasks:raise ValueError('Duplicate original explanation')
        tasks[key]={'key':key,'cohort':row['cohort'],'condition':'original',
                    'claim':row['shown_claim'],'explanation':row['generated_explanation'],
                    'evidence':row['evidence']}
    for row in stress:
        if row['policy']!='no_gate' or row['condition'] not in CONDITIONS:continue
        if not row['generated_explanation']:raise ValueError('Missing changed-context explanation')
        key=f"{row['cohort']}:{row['claim_id']}:{row['condition']}"
        if key in tasks:raise ValueError('Duplicate stress explanation')
        parent=f"{row['cohort']}:{row['claim_id']}:original"
        if parent not in tasks:raise ValueError('Missing original claim')
        tasks[key]={'key':key,'cohort':row['cohort'],'condition':row['condition'],
                    'claim':tasks[parent]['claim'],
                    'explanation':row['generated_explanation'],'evidence':row['evidence']}
    return [tasks[key] for key in sorted(tasks)]


def pair_specs(tasks):
    pairs=[]
    for task in tasks:
        for i,passage in enumerate(task['evidence']):
            pairs.append({'key':f"{task['key']}:{i}",'task_key':task['key'],
                'doc_id':str(passage['id']),'premise_sha256':hashlib.sha256(passage['text'].encode()).hexdigest(),
                'hypothesis_sha256':hashlib.sha256(task['explanation'].encode()).hexdigest()})
    return pairs


def summarize(tasks,cache):
    pairs=pair_specs(tasks)
    require(set(cache)=={p['key'] for p in pairs},'Pair coverage mismatch')
    details={}
    for task in tasks:
        rows=[]
        for pair in (p for p in pairs if p['task_key']==task['key']):
            value=cache[pair['key']]
            require(set(value)=={'premise_sha256','hypothesis_sha256','scores_cen','skip'} and
                    value['premise_sha256']==pair['premise_sha256'] and
                    value['hypothesis_sha256']==pair['hypothesis_sha256'],
                    'Pair identity mismatch')
            scores=value['scores_cen']
            if value['skip'] is None:
                require(isinstance(scores,list) and len(scores)==3 and all(
                    isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1
                    for v in scores) and abs(sum(scores)-1)<1e-6,'Invalid NLI scores')
                rows.append(scores)
            else:require(value['skip']=='token_budget' and scores is None,'Unexpected NLI skip')
        quote=quote_check(task['explanation'],task['claim'],' '.join(e['text'] for e in task['evidence']))
        details[task['key']]={'pair_count':len(task['evidence']),'scored':len(rows),'skipped':len(task['evidence'])-len(rows),
             'max_entailment':max((r[1] for r in rows),default=None),
             'max_contradiction':max((r[0] for r in rows),default=None),'quotes':quote}
    groups={}
    for task in tasks:
        name=f"{task['cohort']}:{task['condition']}";g=groups.setdefault(name,{
            'explanations':0,'pairs_scored':0,'pairs_skipped':0,'quotes':0,
            'quotes_found_in_evidence':0,'quotes_found_in_claim_only':0,'quotes_unmatched':0})
        d=details[task['key']];g['explanations']+=1;g['pairs_scored']+=d['scored'];g['pairs_skipped']+=d['skipped']
        q=d['quotes'];g['quotes']+=q['quoted'];g['quotes_found_in_evidence']+=q['found_in_evidence']
        g['quotes_found_in_claim_only']+=q['found_in_claim_only'];g['quotes_unmatched']+=q['unmatched']
        g.setdefault('max_entailment_values',[])
        if d['max_entailment'] is not None:g['max_entailment_values'].append(d['max_entailment'])
    for group in groups.values():
        values=group.pop('max_entailment_values')
        group['mean_max_entailment']=sum(values)/len(values) if values else None
        group['explanations_with_score']=len(values)
    return {'scope':'English observed development explanations only; NLI is a diagnostic of single-passage entailment, not a factual gold label.',
            'tasks':len(tasks),'pairs':len(pairs),'groups':groups,'details':details}


def prepare():
    fresh=ROOT/'artifacts/fresh_pipeline_received';stress=ROOT/'artifacts/text_stress_received'
    hashes={p.name:receipt(p) for p in (fresh,stress)}
    audit=json.loads((ROOT/'artifacts/source_trace_audit_v1/audit_manifest.json').read_text())
    require(audit['received_receipts_sha256']==hashes,'Source trace binding differs')
    tasks=select_tasks(json.loads((fresh/'predictions.json').read_text()),
                       json.loads((stress/'predictions.json').read_text()))
    require(len(tasks)==840 and len(pair_specs(tasks))>0,'Explanation cohort changed')
    model_files=json.loads((ROOT/'artifacts/semantic_sufficiency_received/input_manifest.json').read_text())['models']['nli']
    identity={'schema_version':1,'source_receipts_sha256':hashes,
        'source_trace_receipt_sha256':sha256(ROOT/'artifacts/source_trace_audit_v1/audit_manifest.json'),
        'task_sha256':hashlib.sha256(json.dumps(tasks,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        'model_files':model_files,'max_pair_tokens':256,'model_language':'English',
        'code_sha256':{name:sha256(ROOT/name) for name in CODE},
        'protocol_sha256':sha256(ROOT/'docs/EXPLANATION_DIAGNOSTIC.md')}
    return tasks,identity


def write_preflight_receipt(folder):
    digest=sha256(folder/'preflight.json')
    write_json_atomic(folder/'audit_manifest.json',{
        'preflight_sha256':digest,'files':{'preflight.json':digest}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    tasks,identity=prepare()
    if args.preflight:
        folder=ROOT/'artifacts/explanation_diagnostic_preflight_v1';folder.mkdir(exist_ok=True)
        quote_totals={'quoted':0,'found_in_evidence':0,'found_in_claim_only':0,'unmatched':0}
        for task in tasks:
            counts=quote_check(task['explanation'],task['claim'],
                               ' '.join(p['text'] for p in task['evidence']))
            for name,count in counts.items():quote_totals[name]+=count
        write_json_atomic(folder/'preflight.json',{'tasks':len(tasks),'pairs':len(pair_specs(tasks)),
            'input_manifest':identity,'conditions':['original',*CONDITIONS],
            'literal_quote_counts':quote_totals})
        write_preflight_receipt(folder)
        print(f"Explanation preflight passed: {len(tasks)} saved explanations, {len(pair_specs(tasks))} passage pairs.")
        return
    out=ROOT/'artifacts/explanation_diagnostic_v1';out.mkdir(exist_ok=True)
    require(not (out/'output_manifest.json').exists(),'Completed run already exists')
    if (out/'input_manifest.json').exists():
        require(json.loads((out/'input_manifest.json').read_text())==identity,'Resume identity changed')
    else:write_json_atomic(out/'input_manifest.json',identity)
    from importlib.metadata import version
    packages={'torch':version('torch'),'transformers':version('transformers'),'numpy':version('numpy')}
    expected=json.loads((ROOT/'artifacts/semantic_sufficiency_received/input_manifest.json').read_text())['packages']
    require(all(packages[n]==expected[n] for n in packages),'Model package profile changed')
    model_path=ROOT/'models/nli-deberta-v3-small'
    require(_model_files(model_path)==identity['model_files'],'Pinned NLI weights changed')
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from apv_rag.multilingual_nli import cen_order,normalize_model_scores
    torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        model_path,local_files_only=True,torch_dtype=torch.float32).eval()
    order=cen_order(model.config.id2label)
    by_key={t['key']:t for t in tasks};specs=pair_specs(tasks)
    cachepath=out/'pairs.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {}
    require(set(cache)<={p['key'] for p in specs},'Unexpected cached pair')
    pending=[]
    def flush():
        if not pending:return
        inputs=tokenizer([a for _,a,_ in pending],[b for _,_,b in pending],
                         return_tensors='pt',padding=True,truncation=False)
        with torch.inference_mode():
            scores=model(**inputs).logits.softmax(dim=-1).cpu().numpy()[:,order]
        for (spec,_,_),values in zip(pending,normalize_model_scores(scores),strict=True):
            cache[spec['key']]={'premise_sha256':spec['premise_sha256'],
                                'hypothesis_sha256':spec['hypothesis_sha256'],
                                'scores_cen':values,'skip':None}
        pending.clear();write_json_atomic(cachepath,cache)
    for position,spec in enumerate(specs,1):
        if spec['key'] in cache:
            val=cache[spec['key']]
            require(val['premise_sha256']==spec['premise_sha256'] and
                    val['hypothesis_sha256']==spec['hypothesis_sha256'],'Cached pair changed')
            continue
        task=by_key[spec['task_key']];premise=task['evidence'][int(spec['key'].rsplit(':',1)[1])]['text']
        hypothesis=task['explanation']
        length=len(tokenizer(premise,hypothesis,truncation=False)['input_ids'])
        if length>256:
            cache[spec['key']]={'premise_sha256':spec['premise_sha256'],
                'hypothesis_sha256':spec['hypothesis_sha256'],'scores_cen':None,'skip':'token_budget'}
            write_json_atomic(cachepath,cache)
        else:
            pending.append((spec,premise,hypothesis))
            if len(pending)==8:flush()
        if position%100==0:print(f'Explanation pairs: {position}/{len(specs)}',flush=True)
    flush()
    result=summarize(tasks,cache)
    write_json_atomic(out/'summary.json',result)
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.glob('*.json') if p.name!='output_manifest.json'})
    print(f'Explanation diagnostic complete: {result["tasks"]} explanations, {result["pairs"]} passage pairs.')


if __name__=='__main__':main()
