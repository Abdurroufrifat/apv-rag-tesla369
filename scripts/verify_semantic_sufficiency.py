"""Replay cached semantic features, classifier probabilities and metrics."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,roc_auc_score,brier_score_loss
from run_semantic_sufficiency import aggregate_features
from apv_rag.splits import sha256,write_json_atomic


def main():
    root=Path(__file__).resolve().parents[1];folder=root/'artifacts/semantic_sufficiency_received'
    load=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    hashes=load('output_manifest.json')
    for n,h in hashes.items():assert sha256(folder/n)==h,n
    identity=load('input_manifest.json');data=root/'data/processed/sufficiency_matched_v2'
    assert sha256(data/'manifest.json')==identity['dataset_manifest_sha256']
    assert sha256(root/'scripts/run_semantic_sufficiency.py')==identity['code_sha256']
    assert sha256(root/'docs/SEMANTIC_SUFFICIENCY.md')==identity['protocol_sha256']
    rows=json.loads((data/'examples.json').read_text(encoding='utf-8'));cache=load('feature_cache.json')
    assert set(cache)=={r['example_id'] for r in rows}
    for r in rows:
        c=cache[r['example_id']];scores=np.asarray(c['scores_cen']);cos=np.asarray(c['cosines'])
        assert scores.shape==(len(r['evidence']),3)
        assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
        assert np.allclose(scores.sum(axis=1),1,atol=1e-6)
        assert np.isfinite(cos).all() and (abs(cos)<=1.00001).all()
        assert np.allclose(aggregate_features(scores,cos),c['features'],atol=1e-12)
    val=[r for r in rows if r['split']=='validation'];pred=load('validation_predictions.json');models=load('models.json');metrics=load('metrics.json')
    assert len(pred)==3*len(val)==294
    result={}
    for name,m in models.items():
        subset={r['example_id']:r for r in pred if r['model']==name}
        assert set(subset)=={r['example_id'] for r in val}
        x=np.asarray([cache[r['example_id']]['features'] for r in val])[:,m['columns']]
        z=((x-np.array(m['mean']))/np.array(m['scale']))@np.array(m['coefficients'])[0]+m['intercept'][0]
        p=1/(1+np.exp(-z));truth=np.array([r['target_complete_rationale'] for r in val])
        assert np.allclose(p,[subset[r['example_id']]['probability_complete'] for r in val],atol=1e-10)
        assert all(subset[r['example_id']]['true_label']==r['target_complete_rationale'] for r in val)
        label=(p>=.5).astype(int)
        result[name]={'accuracy':float(accuracy_score(truth,label)),'macro_f1':float(f1_score(truth,label,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(truth,label)),'roc_auc':float(roc_auc_score(truth,p)),'brier':float(brier_score_loss(truth,p))}
        for k,v in result[name].items():assert abs(v-metrics[name][k])<1e-10
    out=root/'artifacts/semantic_sufficiency_verification';out.mkdir(exist_ok=True)
    write_json_atomic(out/'verified_metrics.json',result)
    (out/'RESULTS.md').write_text('''# Matched semantic completeness results

All580 feature-cache records,294 validation prediction records, file/code/protocol/dataset hashes, feature aggregation, serialized classifier probabilities and metrics passed non-neural replay. Neural NLI/embedding inference and training fits were not independently rerun.

| Features | Accuracy | Macro F1 | ROC AUC | Brier |
|---|---:|---:|---:|---:|
| NLI | .6327 | .6167 | .6989 | .2261 |
| Embedding | .8878 | .8876 | .9521 | .0855 |
| Combined | .8776 | .8771 | .9575 | .0888 |

Validation:98 examples from49 matched claim pairs. Context-length and claim-only controls scored50% accuracy/AUC.5. Semantic features distinguish this constructed annotation-coverage target, unlike the lexical baseline. No significance or real-world sufficiency claim:49 claims, dependent same-split distractors, topic cues and adaptive dataset construction limit interpretation. Probabilities uncalibrated; fixed.5 threshold not established as safe abstention policy. Embedding performance may mainly detect topic mismatch.

Integration may proceed as an explicitly experimental annotation-completeness component. Preserve NLI,embedding,combined variants and a no-gate ablation; do not select the best validation result and call it independent confirmation. Real retrieved passages, multi-sentence reasoning, source independence and explanation grounding remain unvalidated. Do not call this trained classifier an authentication model or universal sufficiency head. No manuscript or GitHub push.
''',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
