"""Verify received fresh RAG outputs without neural inference or model loading."""
import json
import math
from pathlib import Path
import re
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from run_fresh_climate_rag import summarize
from apv_rag.generative_rag import LABELS
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2
from apv_rag.retrieval import BM25Index
from apv_rag.sentence_context import select_sentences
from apv_rag.splits import sha256,write_json_atomic


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def verify_file_hashes(folder, expected):
    for name,digest in expected.items():
        if sha256(folder/name)!=digest:raise ValueError(f'SHA-256 mismatch: {name}')


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    folder=ROOT/'artifacts/fresh_climate_rag_received'
    source=ROOT/'data/external/climate_fever/frozen_v1'
    preflight=ROOT/'artifacts/fresh_climate_rag_preflight_v1'
    hashes=read(folder/'output_manifest.json')
    require(set(hashes)=={'predictions.json','generation_summary.json','input_manifest.json'},'Unexpected export coverage')
    verify_file_hashes(folder,hashes)
    identity=read(folder/'input_manifest.json')
    verify_file_hashes(ROOT,identity['code_sha256'])
    verify_file_hashes(source,identity['source_sha256'])
    verify_file_hashes(preflight,identity['preflight_sha256'])
    require(sha256(preflight/'protocol.json')==identity['cohort_sha256'],'Cohort hash differs')
    require(sha256(ROOT/'docs/FRESH_CLIMATE_RAG.md')==identity['protocol_sha256'],'Protocol hash differs')
    protocol=read(preflight/'protocol.json')
    verify_file_hashes(ROOT,protocol['inputs'])
    verify_file_hashes(preflight,read(preflight/'freeze_receipt.json'))
    prior=read(ROOT/'artifacts/climate_rag_received/input_manifest.json')
    require(identity['settings']==prior['settings'],'Previously frozen inference settings differ')
    require(identity['models']==prior['models'],'Previously pinned model identities differ')
    claims={r['id']:r['claim'] for r in map(json.loads,(preflight/'claims.jsonl').read_text(encoding='utf-8').splitlines())}
    gold=read(preflight/'gold.json');groups=read(preflight/'groups.json')
    corpus=sorted(map(json.loads,(source/'corpus.jsonl').read_text(encoding='utf-8').splitlines()),key=lambda r:r['doc_id'])
    byid={r['doc_id']:r for r in corpus}
    index=BM25Index([' '.join(r['abstract']) for r in corpus])
    rows=read(folder/'predictions.json')
    require(len(rows)==46 and [r['claim_id'] for r in rows]==protocol['cohort'],'Claim coverage/order differs')
    for r in rows:
        i=r['claim_id'];require(r['claim']==claims[i] and r['true_label']==gold[i],'Claim/gold alignment differs')
        require(bool(r['shown_claim']) and r['claim'].startswith(r['shown_claim']),'Shown claim is not a source prefix')
        retrieved=[(corpus[pos]['doc_id'],float(score)) for pos,score in index.search(r['claim'],top_k=3) if score>0]
        require(len(retrieved)==len(r['evidence']),'Retrieval length differs')
        for (doc_id,score),e in zip(retrieved,r['evidence'],strict=True):
            require(e['id']==doc_id and math.isclose(e['bm25_score'],score,rel_tol=1e-12),'BM25 ID/score differs')
            selected=select_sentences(r['claim'],byid[doc_id]['abstract'])
            require(e['selected_sentence_indices']==selected['sentence_indices'],'Sentence indices differ')
            require(bool(e['text']) and selected['text'].startswith(e['text']),'Passage is not a selected-source prefix')
        require(r['context_source_ids']==[e['id'] for e in r['evidence']],'Source ID alignment differs')
        context='\n'.join(f"[{e['id']}] {e['text']}" for e in r['evidence'])
        verdict=r['generated_verdict'];explanation=r['generated_explanation'];reasons=[]
        if not r['evidence']:reasons.append('no_evidence')
        else:
            vp=f"Evidence: {context}\nClaim: {r['shown_claim']}\nReply only Supported, Refuted, or Not Enough Evidence."
            require(r['verdict_prompt']==('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+vp+'<|im_end|>\n<|im_start|>assistant\n'),'Serialized verdict prompt differs')
            if verdict not in LABELS:reasons.append('invalid_verdict')
            else:
                ep=f"Evidence: {context}\nClaim: {r['shown_claim']}\nVerdict: {verdict}\nExplain why this evidence supports, contradicts, or cannot establish the claim. Use only the supplied evidence and keep the explanation short."
                require(r['explanation_prompt']==('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+ep+'<|im_end|>\n<|im_start|>assistant\n'),'Serialized explanation prompt differs')
                if not explanation:reasons.append('empty_explanation')
        for key in ('verdict_prompt_tokens','explanation_prompt_tokens'):
            require(r[key] is None or (type(r[key]) is int and 0<r[key]<=1024),'Recorded prompt budget invalid')
        numeric=numeric_provenance_v2(explanation or '',r['shown_claim'],r['evidence'])
        require(numeric==r['numeric_provenance'],'Numeric provenance differs')
        if numeric['absent_from_inputs']:reasons.append('novel_numeric_value')
        require(reasons==r['reasons'],'Guard reasons differ')
        require(r['raw_candidate_label']==(verdict if verdict in LABELS else None),'Raw decision differs')
        require(r['structural_candidate_label']==(verdict if not reasons else None),'Guard decision differs')
        sentences=re.split(r'(?<=[.!?])\s+',explanation) if explanation else []
        require([a['sentence'] for a in r['explanation_nli_audit']]==sentences,'NLI sentence alignment differs')
        for audit in r['explanation_nli_audit']:
            scores=audit['scores_cen'];require(len(scores)==len(r['evidence']),'NLI evidence alignment differs')
            for probabilities in scores:
                require(len(probabilities)==3 and all(math.isfinite(p) and 0<=p<=1 for p in probabilities) and math.isclose(sum(probabilities),1,abs_tol=1e-6),'Invalid NLI probabilities')
            require(math.isclose(audit['max_entailment'],max(p[1] for p in scores)),'NLI aggregate differs')
    summary=read(folder/'generation_summary.json')
    for field,key in (('raw_candidate_label','raw_verdict_metrics'),('structural_candidate_label','structural_guard_metrics')):
        require(summarize(rows,field)==summary[key],'Metric replay differs')
    require(summary['rows']==46 and summary['sentences_audited']==sum(len(r['explanation_nli_audit']) for r in rows),'Summary count differs')
    # Descriptive paired uncertainty; never used to tune or promote a guard.
    names=sorted(set(groups.values()));groupidx={g:i for i,g in enumerate(names)}
    cms=[]
    for field in ('raw_candidate_label','structural_candidate_label'):
        cm=np.zeros((len(names),4,4),dtype=int)
        for r in rows:
            prediction=LABELS.index(r[field]) if r[field] in LABELS else 3
            cm[groupidx[groups[r['claim_id']]],LABELS.index(r['true_label']),prediction]+=1
        cms.append(cm)
    def macro(cm):
        denominator=(cm.sum(axis=-1)+cm.sum(axis=-2))[...,:3]
        diagonal=np.diagonal(cm,axis1=-2,axis2=-1)[...,:3]
        return np.divide(2*diagonal,denominator,out=np.zeros_like(denominator,dtype=float),where=denominator!=0).mean(axis=-1)
    draws=np.random.default_rng(369).integers(0,len(names),size=(2000,len(names)))
    deltas=macro(cms[1][draws].sum(axis=1))-macro(cms[0][draws].sum(axis=1))
    rejected=[r for r in rows if r['raw_candidate_label'] is not None and r['structural_candidate_label'] is None]
    result={'claims':46,'groups':len(names),'raw_correct':sum(r['raw_candidate_label']==r['true_label'] for r in rows),
        'guarded_correct':sum(r['structural_candidate_label']==r['true_label'] for r in rows),'guarded_answers':sum(r['structural_candidate_label'] is not None for r in rows),
        'rejected_correct':sum(r['raw_candidate_label']==r['true_label'] for r in rejected),'rejected_incorrect':sum(r['raw_candidate_label']!=r['true_label'] for r in rejected),
        'rejection_counts':dict(Counter(x for r in rows for x in r['reasons'])), 'metrics':summary,
        'guard_minus_raw_macro_f1':float(macro(cms[1].sum(axis=0))-macro(cms[0].sum(axis=0))),
        'descriptive_95_percent_group_bootstrap_interval':np.quantile(deltas,[.025,.975]).tolist(),
        'claim_prefix_clipped_count':sum(r['shown_claim']!=r['claim'] for r in rows),
        'checks':'Export/input/code/model-identity hashes; claim/gold alignment; BM25 IDs/scores; source-text prefixes; sentence selection; literal prompts; recorded budgets; numeric guard; NLI score consistency; metrics.',
        'limits':'Neural inference, tokenizer clipping and chat-template rendering not rerun locally. Recorded model identity is checked; weights unavailable for local rehashing. Source membership is not publisher authentication; NLI is not factual explanation validation. Already component-observed public claims; no blind independent dataset.'}
    output=ROOT/'artifacts/fresh_climate_rag_verification_v1';output.mkdir(exist_ok=True)
    write_json_atomic(output/'verification.json',result)
    write_json_atomic(output/'receipt.json',{'scope':'Received export verification; no local neural rerun',
        'received_files':{p.name:sha256(p) for p in folder.glob('*.json')},'analysis_code_sha256':sha256(Path(__file__).resolve()),'verification_sha256':sha256(output/'verification.json')})
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','checks','limits')},indent=2))


if __name__=='__main__':main()
