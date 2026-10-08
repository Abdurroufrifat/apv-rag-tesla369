"""Constructed rationale-completeness baseline and shortcut controls."""
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from apv_rag.splits import sha256, write_json_atomic


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'data/processed/sufficiency_annotations_v1'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():assert sha256(source/name)==digest
    rows=json.loads((source/'examples.json').read_text(encoding='utf-8'))
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    assert not {r['group_id'] for r in train}&{r['group_id'] for r in val}
    y=np.array([r['target_complete_rationale'] for r in train]);truth=np.array([r['target_complete_rationale'] for r in val])
    def text(r,mode):
        if mode=='claim_only':return r['claim']
        return r['claim']+' EVIDENCE '+' '.join(e['text'] for e in r['evidence'])
    def length(r):
        return [len(r['evidence']),len(' '.join(e['text'] for e in r['evidence']).split()),len(r['claim'].split())]
    def metrics(p):
        label=(np.array(p)>=.5).astype(int)
        return {'accuracy':float(accuracy_score(truth,label)), 'macro_f1':float(f1_score(truth,label,average='macro',zero_division=0)), 'balanced_accuracy':float(balanced_accuracy_score(truth,label)), 'roc_auc':float(roc_auc_score(truth,p)), 'brier':float(brier_score_loss(truth,p))}
    output=root/'artifacts/annotation_sufficiency_v1';output.mkdir(exist_ok=True)
    results={};predictions=[];models={}
    for mode in ('claim_evidence_text','claim_only','context_length'):
        if mode=='context_length':
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=369))
            xtrain=[length(r) for r in train];xval=[length(r) for r in val]
        else:
            model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=20000,sublinear_tf=True),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=369))
            xtrain=[text(r,mode) for r in train];xval=[text(r,mode) for r in val]
        model.fit(xtrain,y);prob=model.predict_proba(xval)[:,1]
        results[mode]=metrics(prob)
        clf=model[-1]
        if mode=='context_length':
            models[mode]={'mean':model[0].mean_.tolist(),'scale':model[0].scale_.tolist(),'coefficients':clf.coef_.tolist(),'intercept':clf.intercept_.tolist()}
        else:
            models[mode]={'vocabulary':{k:int(v) for k,v in model[0].vocabulary_.items()},'idf':model[0].idf_.tolist(),'coefficients':clf.coef_.tolist(),'intercept':clf.intercept_.tolist()}
        for r,p in zip(val,prob,strict=True):predictions.append({'model':mode,'example_id':r['example_id'],'claim_id':r['claim_id'],'group_id':r['group_id'],'construction':r['construction'],'true_label':r['target_complete_rationale'],'probability_complete':float(p)})
    results['training_prevalence']=metrics(np.full(len(val),float(y.mean())))
    write_json_atomic(output/'metrics.json',results)
    write_json_atomic(output/'validation_predictions.json',predictions)
    write_json_atomic(output/'models.json',models)
    write_json_atomic(output/'input_manifest.json',{'dataset_manifest_sha256':sha256(source/'manifest.json'),'dataset_files':manifest['files'],'code_sha256':sha256(Path(__file__)),'settings':{'seed':369,'C':1,'threshold':.5,'tfidf_ngram_range':[1,2],'min_df':2,'max_features':20000,'class_weight':'balanced'},'scope':'Constructed complete-rationale target. Train-only fitting; frozen group validation; no hyperparameter selection. No real-world sufficiency claim or deployment gate.'})
    lines=['# Annotation-completeness baseline', '', '| Model | Macro F1 | Balanced accuracy | ROC AUC | Brier |','|---|---:|---:|---:|---:|']
    for mode,m in results.items():lines.append(f"| {mode} | {m['macro_f1']:.4f} | {m['balanced_accuracy']:.4f} | {m['roc_auc']:.4f} | {m['brier']:.4f} |")
    lines+=['', '635 training examples and149 validation examples;60 validation claims in40 groups. Every derivative shares its parent split. TF-IDF and scaler fit only training. FixedC1,seed369,threshold.5; no tuning or calibration. These probabilities are not calibrated.', '', 'Target is complete annotated rationale inclusion. Sentence-count/length shortcuts are expected from sentence-removal construction. Claim-only control tests whether claim identity predicts the target. Construction names, sentence indices, gold rationales and target labels never enter model inputs. Do not interpret validation success as learned universal semantic sufficiency or deploy this model as an evidence gate.', '', 'Next requirement: compare against the length control and test matched-length negative evidence before integration. No manuscript or GitHub push.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json_atomic(output/'output_manifest.json',{p.name:sha256(p) for p in output.iterdir() if p.name!='output_manifest.json'})
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
