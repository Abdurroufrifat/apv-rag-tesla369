"""Post-hoc fixed-model test of the existing domain-collapse rule; no refitting."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from run_fresh_copy_confirmation import check_freeze
from apv_rag.evidence_baseline import LABELS
from apv_rag.repetition_stress import collapse_provenance_families,inject_derivative_copies
from apv_rag.splits import sha256,write_json_atomic


def main():
    prior=ROOT/'artifacts/fresh_copy_confirmation_v1';out=ROOT/'artifacts/domain_collapse_transfer_v1'
    if out.exists():raise FileExistsError('Completed diagnostic exists; refusing overwrite')
    check_freeze()
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    inputs=read(prior/'model_inputs.json');gold=read(prior/'gold.json');groups=read(prior/'groups.json')
    ids=[r['claim_id'] for r in inputs];truth={r['claim_id']:r['label'] for r in gold}
    old=read(prior/'predictions.json');original_receipt=read(prior/'evaluation_receipt.json')
    if sha256(prior/'predictions.json')!=original_receipt['outputs']['predictions.json']:
        raise ValueError('Prior probability export changed')
    copies=(0,1,5,10,25);modelpath=ROOT/'artifacts/trained_copy_robustness_v1/control.joblib'
    # Freeze this diagnostic before its new model prediction. Prior outcomes are observed.
    out.mkdir()
    dependencies=[Path(__file__).resolve(),ROOT/'src/apv_rag/repetition_stress.py',ROOT/'src/apv_rag/trained_copy_robustness.py',ROOT/'src/apv_rag/provenance_model.py',ROOT/'src/apv_rag/disagreement_model.py',ROOT/'src/apv_rag/evidence_baseline.py',modelpath]
    dependencies += [prior/n for n in ('model_inputs.json','gold.json','groups.json','predictions.json','evaluation_receipt.json')]
    write_json_atomic(out/'protocol.json',{'scope':'Post-hoc supplied-evidence diagnostic on already observed 48 claims; not independent confirmation or full RAG.',
        'intervention':'Existing first-answer-per-source-domain collapse applied to the frozen plain control classifier; no new family model fitted.',
        'copies':list(copies),'labels':'Four original benchmark labels; false support means predicted Supported when reference label is another class.',
        'primary_descriptive_contrast':'Domain collapse minus raw control false-support rate at 25 copies',
        'uncertainty':'2000 paired article-group bootstrap draws; seed369; descriptive intervals only',
        'promotion_or_tuning':False,'new_human_labels':False,
        'inputs_and_code':{p.relative_to(ROOT).as_posix():sha256(p) for p in dependencies}})
    original=[r['record'] for r in inputs];collapsed=[collapse_provenance_families(r) for r in original]
    for k in copies:
        if [collapse_provenance_families(inject_derivative_copies(r,k)) for r in original]!=collapsed:
            raise ValueError('Collapse differs after exact copies')
    with threadpool_limits(1):
        model=joblib.load(modelpath);columns=[list(model.classes_).index(x) for x in LABELS]
        probabilities=model.predict_proba(collapsed)[:,columns]
    rows=list(old)
    for k in copies:
        rows.extend({'model':'control_domain_collapsed','copies':k,'claim_id':i,'probabilities':p.tolist()} for i,p in zip(ids,probabilities,strict=True))
    names=sorted(set(groups.values()));gidx={g:i for i,g in enumerate(names)};counts={};summary={}
    for name in ('control','copy_augmented','control_domain_collapsed'):
        for k in copies:
            data=[r for r in rows if r['model']==name and r['copies']==k]
            if [r['claim_id'] for r in data]!=ids:raise ValueError('Prediction coverage differs')
            predicted=[LABELS[int(np.argmax(r['probabilities']))] for r in data]
            negatives=sum(truth[i]!=LABELS[0] for i in ids)
            fp=sum(truth[i]!=LABELS[0] and prediction==LABELS[0] for i,prediction in zip(ids,predicted,strict=True))
            summary[f'{name}:copies_{k}']={'false_support_count':fp,'non_support_reference_claims':negatives,'false_support_rate':fp/negatives,
                'accuracy':sum(truth[i]==prediction for i,prediction in zip(ids,predicted,strict=True))/len(ids),
                'macro_f1':float(f1_score([truth[i] for i in ids],predicted,labels=list(LABELS),average='macro',zero_division=0))}
            groupcounts=np.zeros((len(names),2),dtype=int)
            for i,prediction in zip(ids,predicted,strict=True):
                if truth[i]!=LABELS[0]:groupcounts[gidx[groups[i]]]+=[prediction==LABELS[0],1]
            counts[name,k]=groupcounts
    draws=np.random.default_rng(369).integers(0,len(names),size=(2000,len(names)))
    a=counts['control',25][draws].sum(axis=1);b=counts['control_domain_collapsed',25][draws].sum(axis=1)
    valid=(a[:,1]>0)&(b[:,1]>0);delta=b[valid,0]/b[valid,1]-a[valid,0]/a[valid,1]
    result={'claims':len(ids),'article_groups':len(names),'metrics':summary,
        'primary_descriptive_difference':summary['control_domain_collapsed:copies_25']['false_support_rate']-summary['control:copies_25']['false_support_rate'],
        'descriptive_95_percent_group_bootstrap_interval':np.quantile(delta,[.025,.975]).tolist(),
        'valid_bootstrap_draws':int(valid.sum()),'original_clean_answers':sum(len(q['answers']) for r in original for q in r['questions']),
        'collapsed_clean_answers':sum(len(q['answers']) for r in collapsed for q in r['questions']),
        'no_refitting':True,'no_new_neural_inference':True,
        'qualification':'All evidence uses the Wikipedia domain; domain collapse can remove different articles as well as copies. Zero copy sensitivity is imposed by input collapse, not proof of factual improvement or correct provenance families.'}
    write_json_atomic(out/'predictions.json',rows);write_json_atomic(out/'summary.json',result)
    write_json_atomic(out/'receipt.json',{n:sha256(out/n) for n in ('protocol.json','predictions.json','summary.json')})
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))
    for name in ('control','copy_augmented','control_domain_collapsed'):
        for k in (0,25):print(name,k,summary[f'{name}:copies_{k}'])


if __name__=='__main__':main()
