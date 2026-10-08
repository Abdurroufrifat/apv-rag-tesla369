"""Fixed matched-construction semantic features; no universal sufficiency claim."""
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,roc_auc_score,brier_score_loss
from apv_rag.nli_comparison import _model_files
from apv_rag.multilingual_nli import cen_order,normalize_model_scores
from apv_rag.splits import sha256,write_json_atomic


def aggregate_features(scores,cosines):
    a=np.asarray(scores,dtype=float);c=np.asarray(cosines,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3 or len(c)!=len(a) or not len(c):raise ValueError('Feature inputs malformed')
    f=np.concatenate([a.mean(0),a.max(0),a.std(0),[c.mean(),c.max(),c.min(),c.std()]])
    if not np.isfinite(f).all():raise ValueError('Nonfinite features')
    return f.tolist()


def main():
    root=Path(__file__).resolve().parents[1]
    data=root/'data/processed/sufficiency_matched_v2'
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for n,h in manifest['files'].items():
        if sha256(data/n)!=h:raise ValueError('Dataset checksum mismatch')
    rows=json.loads((data/'examples.json').read_text(encoding='utf-8'))
    original=root/'artifacts/retrieval_nli_comparison'
    hashes=json.loads((original/'output_manifest.json').read_text(encoding='utf-8'))
    if sha256(original/'input_manifest.json')!=hashes['input_manifest.json']:raise ValueError('Model metadata checksum mismatch')
    meta=json.loads((original/'input_manifest.json').read_text(encoding='utf-8'))
    paths={'nli':root/'models/nli-deberta-v3-small','embedding':root/'models/all-MiniLM-L6-v2'}
    for name,path in paths.items():
        if _model_files(path)!=meta['models'][name]:raise ValueError('Pinned model checksum mismatch')
    out=root/'artifacts/semantic_sufficiency_v1'
    if (out/'output_manifest.json').exists():raise FileExistsError('Completed run exists')
    out.mkdir(exist_ok=True)
    identity={'dataset_manifest_sha256':sha256(data/'manifest.json'),'models':meta['models'],'code_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(root/'docs/SEMANTIC_SUFFICIENCY.md'),'packages':{n:version(n) for n in ('torch','transformers','sentence-transformers','numpy','scikit-learn')},'settings':{'seed':369,'threads':4,'dtype':'float32','nli_pair_tokens':256,'embedding_tokens':384,'classifier_C':1,'threshold':.5,'class_weight':'balanced'}}
    receipt=out/'input_manifest.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8'))!=identity:raise ValueError('Identity changed; cache reuse refused')
    write_json_atomic(receipt,identity)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(paths['nli'],local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(paths['nli'],local_files_only=True,torch_dtype=torch.float32).eval()
    order=cen_order(model.config.id2label)
    embedding=SentenceTransformer(str(paths['embedding']),device='cpu',local_files_only=True)
    embedding.float();embedding.max_seq_length=384
    cachefile=out/'feature_cache.json';cache=json.loads(cachefile.read_text(encoding='utf-8')) if cachefile.exists() else {}
    for position,row in enumerate(rows):
        key=row['example_id']
        if key not in cache:
            text=[e['text'] for e in row['evidence']]
            tokens=tokenizer(text,[row['claim']]*len(text),return_tensors='pt',padding=True,truncation=True,max_length=256)
            with torch.inference_mode():
                scores=model(**tokens).logits.float().softmax(-1).cpu().numpy()[:,order]
            scores=normalize_model_scores(scores.tolist())
            vectors=embedding.encode([row['claim']]+text,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            cosine=(vectors[1:]@vectors[0]).tolist()
            cache[key]={'scores_cen':scores,'cosines':cosine,'features':aggregate_features(scores,cosine)}
            write_json_atomic(cachefile,cache)
        if position%20==0:print(f'Features {position+1}/{len(rows)}',flush=True)
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    assert not {r['group_id'] for r in train}&{r['group_id'] for r in val}
    truth=np.array([r['target_complete_rationale'] for r in val]);y=np.array([r['target_complete_rationale'] for r in train])
    summary={};predictions=[];models={}
    for name,columns in [('nli',list(range(9))),('embedding',list(range(9,13))),('combined',list(range(13)))]:
        xtrain=np.array([cache[r['example_id']]['features'] for r in train])[:,columns]
        xval=np.array([cache[r['example_id']]['features'] for r in val])[:,columns]
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=369))
        clf.fit(xtrain,y);p=clf.predict_proba(xval)[:,1];label=(p>=.5).astype(int)
        summary[name]={'accuracy':float(accuracy_score(truth,label)),'macro_f1':float(f1_score(truth,label,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(truth,label)),'roc_auc':float(roc_auc_score(truth,p)),'brier':float(brier_score_loss(truth,p))}
        models[name]={'columns':columns,'mean':clf[0].mean_.tolist(),'scale':clf[0].scale_.tolist(),'coefficients':clf[1].coef_.tolist(),'intercept':clf[1].intercept_.tolist()}
        predictions.extend({'model':name,'example_id':r['example_id'],'group_id':r['group_id'],'true_label':r['target_complete_rationale'],'probability_complete':float(prob)} for r,prob in zip(val,p,strict=True))
    write_json_atomic(out/'models.json',models);write_json_atomic(out/'metrics.json',summary);write_json_atomic(out/'validation_predictions.json',predictions)
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.glob('*.json') if p.name!='output_manifest.json'})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
