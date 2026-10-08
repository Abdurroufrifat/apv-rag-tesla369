"""Post-hoc temperature calibration; frozen classifiers and group-disjoint fit/check."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.evidence_baseline import LABELS
from apv_rag.metrics import multiclass_brier, expected_calibration_error, risk_coverage
from apv_rag.splits import sha256, write_json_atomic

def read(p):return json.loads(p.read_text())
def transform(p,t):return softmax(np.log(np.clip(p,1e-15,1))/t,axis=1)
def metrics(p,y):
    pred=[LABELS[i] for i in p.argmax(axis=1)]
    return {'brier':multiclass_brier(y,p,LABELS),'ece_15_bins':expected_calibration_error(y,pred,p),'accuracy':float(np.mean(np.array(pred)==y)), 'risk_coverage':risk_coverage(y,pred,p.max(axis=1),(.25,.5,.75,1))}
def main():
    out=ROOT/'artifacts/copy_temperature_v1'
    if out.exists():raise FileExistsError(out)
    parent=ROOT/'artifacts/trained_copy_robustness_v1'
    receipt=read(parent/'receipt.json')
    for section in ('inputs','outputs'):
        for n,h in receipt[section].items():
            if sha256((ROOT if section=='inputs' else parent)/n)!=h:raise ValueError('Changed input '+n)
    rows=read(parent/'predictions.json'); gp=ROOT/'data/processed/averitec/phase2b_split_v0_1/group_assignments.json'; groups=read(gp)
    clean=[r for r in rows if r['model']=='control' and r['copies']==0]
    unique=sorted({groups[str(r['upstream_index'])] for r in clean}); rng=np.random.default_rng(369); rng.shuffle(unique)
    fit_groups=set(unique[:len(unique)//2]); mask=np.array([groups[str(r['upstream_index'])] in fit_groups for r in clean])
    out.mkdir()
    write_json_atomic(out/'protocol.json',{'scope':'Post-hoc development: original validation outcomes and climate outcomes previously inspected; no independent confirmation.', 'method':'One positive scalar temperature per frozen classifier fitted by clean calibration-side negative log likelihood; fixed bounds 0.05 to 20.', 'fit_indices':[r['upstream_index'] for r,m in zip(clean,mask) if m], 'check_indices':[r['upstream_index'] for r,m in zip(clean,mask) if not m], 'seed':369, 'groups_disjoint':True, 'inputs':{p.relative_to(ROOT).as_posix():sha256(p) for p in (parent/'predictions.json',gp,ROOT/'artifacts/fresh_copy_confirmation_v1/predictions.json',Path(__file__).resolve())}})
    result={}; exports=[]
    for model in ('control','copy_augmented'):
        base=[r for r in rows if r['model']==model and r['copies']==0]
        assert [r['upstream_index'] for r in base]==[r['upstream_index'] for r in clean]
        p=np.array([r['probabilities'] for r in base]); y=[r['true_label'] for r in base]; targets=np.array([LABELS.index(v) for v in y])
        def loss(logt):
            q=transform(p[mask],np.exp(logt));return float(-np.log(np.clip(q[np.arange(mask.sum()),targets[mask]],1e-15,1)).mean())
        opt=minimize_scalar(loss,bounds=(np.log(.05),np.log(20)),method='bounded')
        if not opt.success:raise ValueError('Optimizer failed')
        t=float(np.exp(opt.x)); result[model]={'temperature':t,'fit_records':int(mask.sum()),'check_records':int((~mask).sum()),'conditions':{}}
        for count in (0,25):
            subset=[r for r in rows if r['model']==model and r['copies']==count]
            assert [r['upstream_index'] for r in subset]==[r['upstream_index'] for r in clean]
            matrix=np.array([r['probabilities'] for r in subset])[~mask]; truth=np.array(y)[~mask].tolist(); calibrated=transform(matrix,t)
            assert np.array_equal(matrix.argmax(axis=1),calibrated.argmax(axis=1))
            result[model]['conditions']['internal_check_copies_'+str(count)]={'raw':metrics(matrix,truth),'calibrated':metrics(calibrated,truth)}
        fresh=[r for r in read(ROOT/'artifacts/fresh_copy_confirmation_v1/predictions.json') if r['model']==model and r['copies']==0]
        gold={r['claim_id']:r['label'] for r in read(ROOT/'artifacts/fresh_copy_confirmation_v1/gold.json')}
        matrix=np.array([r['probabilities'] for r in fresh]); truth=[gold[r['claim_id']] for r in fresh]; calibrated=transform(matrix,t)
        result[model]['conditions']['observed_climate_clean']={'raw':metrics(matrix,truth),'calibrated':metrics(calibrated,truth)}
        exports.extend({'model':model,'claim_id':r['claim_id'],'probabilities':q.tolist()} for r,q in zip(fresh,calibrated))
    write_json_atomic(out/'summary.json',result);write_json_atomic(out/'climate_probabilities.json',exports)
    write_json_atomic(out/'receipt.json',{n:sha256(out/n) for n in ('protocol.json','summary.json','climate_probabilities.json')})
    for model,r in result.items():
        print(model,'temperature',round(r['temperature'],4))
        for name,m in r['conditions'].items():print(name,'Brier',round(m['raw']['brier'],4),'->',round(m['calibrated']['brier'],4))
if __name__=='__main__':main()
