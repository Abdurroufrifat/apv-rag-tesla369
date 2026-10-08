"""Descriptive calibration/matched-coverage evaluation of saved copy policies."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from apv_rag.evidence_baseline import LABELS
from apv_rag.metrics import multiclass_brier,expected_calibration_error,risk_coverage
from apv_rag.splits import sha256,write_json_atomic


def main():
    out=ROOT/'artifacts/copy_calibration_diagnostic_v1'
    if out.exists():raise FileExistsError('Completed diagnostic exists; refusing overwrite')
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    domain=ROOT/'artifacts/domain_collapse_transfer_v1';exact=ROOT/'artifacts/exact_copy_removal_v1'
    for folder in (domain,exact):
        for n,d in read(folder/'receipt.json').items():
            if sha256(folder/n)!=d:raise ValueError('Saved policy export changed: '+n)
        protocol=read(folder/'protocol.json')
        for n,d in protocol['inputs_and_code'].items():
            if sha256(ROOT/n)!=d:raise ValueError('Source/model/code changed: '+n)
    parent=ROOT/'artifacts/fresh_copy_confirmation_v1'
    gold=read(parent/'gold.json');groups=read(parent/'groups.json');ids=[r['claim_id'] for r in gold];truth=[r['label'] for r in gold]
    rows=read(domain/'predictions.json')+read(exact/'predictions_by_input_identity.json')
    out.mkdir()
    inputs=[domain/'predictions.json',domain/'receipt.json',exact/'predictions_by_input_identity.json',exact/'receipt.json',parent/'gold.json',parent/'groups.json',Path(__file__).resolve(),ROOT/'src/apv_rag/metrics.py']
    write_json_atomic(out/'protocol.json',{'scope':'Post-hoc descriptive diagnostic on the same 48 observed supplied-evidence claims; not independent confirmation or a new calibration head.',
        'copies':[0,25],'models':['control','copy_augmented','control_domain_collapsed','control_exact_copy_removal'],
        'brier':'Mean sum of squared four-class probability errors, range 0 to 2; lower is better.',
        'ece':'Existing 15-bin equal-width top-label ECE; small sample and binning-sensitive.',
        'coverages':[.25,.5,.75,1.0],'ranking':'Existing stable descending maximum-probability ranking; fixed original row order breaks ties.',
        'selection':'Same retained count per policy, not necessarily same claims. No thresholds fitted or policies promoted.',
        'uncertainty':'2000 paired article-group bootstrap draws for Brier differences at 25 copies, seed369; descriptive unadjusted intervals.',
        'inputs_and_code':{p.relative_to(ROOT).as_posix():sha256(p) for p in inputs}})
    targets=np.eye(4)[[LABELS.index(y) for y in truth]];metrics={};losses={}
    for model in ('control','copy_augmented','control_domain_collapsed','control_exact_copy_removal'):
        for copies in (0,25):
            selected=[r for r in rows if r['model']==model and r['copies']==copies]
            byid={r['claim_id']:r for r in selected}
            if len(byid)!=48 or len(selected)!=48 or set(byid)!=set(ids):raise ValueError('Policy claim coverage differs')
            matrix=np.asarray([byid[i]['probabilities'] for i in ids]);predicted=[LABELS[i] for i in matrix.argmax(axis=1)]
            key=f'{model}:copies_{copies}';losses[key]=((matrix-targets)**2).sum(axis=1)
            brier=multiclass_brier(truth,matrix,LABELS)
            if abs(brier-float(losses[key].mean()))>1e-12:raise ValueError('Independent Brier calculation differs')
            curve=risk_coverage(truth,predicted,matrix.max(axis=1),(.25,.5,.75,1.0))
            correct=np.asarray([t==p for t,p in zip(truth,predicted,strict=True)])
            order=np.argsort(-matrix.max(axis=1),kind='stable')
            for coverage,n in ((.25,12),(.5,24),(.75,36),(1.0,48)):
                if curve[f'{coverage:.6f}']['selected']!=n or abs(curve[f'{coverage:.6f}']['accuracy']-float(correct[order[:n]].mean()))>1e-12:
                    raise ValueError('Matched-coverage calculation differs')
            metrics[key]={'multiclass_brier':brier,'top_label_ece_15_bins':expected_calibration_error(truth,predicted,matrix,bins=15),
                'mean_top_label_confidence':float(matrix.max(axis=1).mean()),'accuracy':float(correct.mean()),'risk_coverage':curve}
    unique=sorted(set(groups.values()));draws=np.random.default_rng(369).integers(0,len(unique),size=(2000,len(unique)))
    uncertainty={}
    for model in ('copy_augmented','control_domain_collapsed','control_exact_copy_removal'):
        delta=losses[f'{model}:copies_25']-losses['control:copies_25']
        stats=np.asarray([[delta[[groups[i]==g for i in ids]].sum(),sum(groups[i]==g for i in ids)] for g in unique])
        sampled=stats[draws].sum(axis=1);values=sampled[:,0]/sampled[:,1]
        uncertainty[model]={'brier_difference_vs_control_at_25_copies':float(delta.mean()),'descriptive_95_percent_group_bootstrap_interval':np.quantile(values,[.025,.975]).tolist()}
    result={'claims':48,'article_groups':len(unique),'metrics':metrics,'brier_contrasts':uncertainty,'new_model_calls':0,'fitting':False,
        'limitations':'Probabilities are uncalibrated. Already observed small cohort; 15-bin ECE and selection curves are noisy. No new sufficiency head, learned abstention policy, full RAG confirmation or publisher authentication is demonstrated.'}
    write_json_atomic(out/'summary.json',result)
    write_json_atomic(out/'receipt.json',{n:sha256(out/n) for n in ('protocol.json','summary.json')})
    for model in ('control','copy_augmented','control_domain_collapsed','control_exact_copy_removal'):
        m=metrics[f'{model}:copies_25'];print(model,'Brier',round(m['multiclass_brier'],4),'ECE',round(m['top_label_ece_15_bins'],4),'half-coverage accuracy',m['risk_coverage']['0.500000']['accuracy'])


if __name__=='__main__':main()
