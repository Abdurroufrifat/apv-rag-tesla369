"""Exploratory NLI reranking of fixed top10 candidates; no new verdict generation."""
import argparse,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from apv_rag.retrieval import BM25Index
from apv_rag.sentence_context import select_sentences
from apv_rag.multilingual_nli import cen_order,normalize_model_scores
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256,write_json_atomic

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-inputs',action='store_true');args=parser.parse_args()
    source=ROOT/'data/external/climate_fever/frozen_v1';claims_path=ROOT/'artifacts/fresh_climate_rag_preflight_v1/claims.jsonl'
    frozen=read(ROOT/'artifacts/semantic_retrieval_preflight_v1/protocol.json')
    for n,h in frozen['inputs'].items():
        if sha256(ROOT/n)!=h:raise ValueError('Frozen input changed: '+n)
    corpus=sorted([json.loads(x) for x in (source/'corpus.jsonl').read_text(encoding='utf-8').splitlines()],key=lambda r:r['doc_id'])
    claims=[json.loads(x) for x in claims_path.read_text(encoding='utf-8').splitlines()]
    index=BM25Index([' '.join(d['abstract']) for d in corpus])
    if args.check_inputs:
        for c in claims:assert len(index.search(c['claim'],top_k=10))<=10
        print('46 claim-only inputs verified; model inference not run.');return
    out=ROOT/'artifacts/semantic_retrieval_v1'
    if (out/'summary.json').exists():raise FileExistsError('Completed run exists')
    original=ROOT/'artifacts/retrieval_nli_comparison'
    if sha256(original/'input_manifest.json')!=read(original/'output_manifest.json')['input_manifest.json']:raise ValueError('NLI manifest changed')
    model_path=ROOT/'models/nli-deberta-v3-small';files=_model_files(model_path)
    if files!=read(original/'input_manifest.json')['models']['nli']:raise ValueError('Pinned NLI model differs')
    identity={'protocol':sha256(ROOT/'artifacts/semantic_retrieval_preflight_v1/protocol.json'),'model_files':files}
    out.mkdir(exist_ok=True)
    if (out/'input_manifest.json').exists() and read(out/'input_manifest.json')!=identity:raise ValueError('Resume identity differs')
    write_json_atomic(out/'input_manifest.json',identity)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(4);torch.manual_seed(369);torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float32).eval();order=cen_order(model.config.id2label)
    for position,c in enumerate(claims):
        path=out/('claim_'+str(c['id'])+'.json')
        if path.exists():continue
        candidates=index.search(c['claim'],top_k=10);passages=[select_sentences(c['claim'],corpus[i]['abstract']) for i,_ in candidates]
        encoded=tokenizer([p['text'] for p in passages],[c['claim']]*len(passages),padding=True,truncation='only_first',max_length=512,return_tensors='pt')
        with torch.inference_mode():scores=normalize_model_scores(model(**encoded).logits.softmax(-1).cpu().numpy()[:,order])
        rank=sorted(range(len(scores)),key=lambda i:(-max(scores[i][0],scores[i][1]),i))[:3]
        write_json_atomic(path,{'claim_id':str(c['id']),'candidates':[{'doc_id':corpus[i]['doc_id'],'title':corpus[i]['title'],'bm25_score':float(score),'selected_sentence_indices':p['sentence_indices'],'nli_cen':s} for (i,score),p,s in zip(candidates,passages,scores)],'selected_candidate_positions':rank})
        print(str(position+1)+'/46',flush=True)
    rows=[read(out/('claim_'+str(c['id'])+'.json')) for c in claims]
    raw={r['claim_id']:r for r in map(json.loads,(source/'climate-fever.jsonl').read_text(encoding='utf-8').splitlines())}
    eligible=baseline=reranked=0
    for row in rows:
        targets={e['article'] for e in raw[row['claim_id']]['evidences'] if e['evidence_label'] in ('SUPPORTS','REFUTES')}
        if targets:
            eligible+=1;baseline+=bool(targets & {x['title'] for x in row['candidates'][:3]});reranked+=bool(targets & {row['candidates'][i]['title'] for i in row['selected_candidate_positions']})
    write_json_atomic(out/'summary.json',{'eligible_claims':eligible,'baseline_article_hits':baseline,'semantic_article_hits':reranked,'claims':len(rows),'scope':'Post-hoc retrieval recall on observed limited corpus; no factual authentication or verdict improvement measured. NLI scores are uncalibrated ranking signals, not gold labels.'})
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    export=ROOT/'semantic_retrieval_outputs.zip'
    with zipfile.ZipFile(export,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():z.write(p,p.name)
    print(read(out/'summary.json'));print('Send:',export)
if __name__=='__main__':main()
