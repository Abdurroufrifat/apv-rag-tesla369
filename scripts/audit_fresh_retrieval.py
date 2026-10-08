"""Post-hoc benchmark evidence recall; labels used only after claim-only retrieval."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.retrieval import BM25Index
from apv_rag.sentence_context import select_sentences
from apv_rag.splits import sha256,write_json_atomic

def main():
    source=ROOT/'data/external/climate_fever/frozen_v1';out=ROOT/'artifacts/fresh_retrieval_audit_v1'
    if out.exists():raise FileExistsError(out)
    corpus=sorted([json.loads(x) for x in (source/'corpus.jsonl').read_text().splitlines()],key=lambda r:r['doc_id'])
    claims=[json.loads(x) for x in (ROOT/'artifacts/fresh_climate_rag_preflight_v1/claims.jsonl').read_text().splitlines()]
    raw={r['claim_id']:r for r in [json.loads(x) for x in (source/'climate-fever.jsonl').read_text().splitlines()]}
    index=BM25Index([' '.join(d['abstract']) for d in corpus]);ranking={str(c['id']):index.search(c['claim'],top_k=20) for c in claims}
    # No evidence labels enter the index or query.
    rows=[]
    for c in claims:
        cid=str(c['id']);r=raw[cid]; decisive=[e for e in r['evidences'] if e['evidence_label'] in ('SUPPORTS','REFUTES')]
        targets={e['article'] for e in decisive};rank=ranking[cid]; titles=[corpus[i]['title'] for i,_ in rank]
        sentences=[s for i,_ in rank[:3] for s in [corpus[i]['abstract'][j] for j in select_sentences(c['claim'],corpus[i]['abstract'])['sentence_indices']]]
        normalize=lambda t:' '.join(t.split())
        hit=sum(normalize(e['evidence']) in {normalize(s) for s in sentences} for e in decisive)
        rows.append({'claim_id':cid,'claim_label':r['claim_label'],'decisive_evidence_count':len(decisive),'decisive_articles':sorted(targets),'retrieved_titles_top20':titles,'article_hit':{str(k):bool(targets.intersection(titles[:k])) for k in (3,10,20)},'exact_decisive_sentence_hits_top3':hit})
    eligible=[r for r in rows if r['decisive_evidence_count']]
    summary={'claims':len(rows),'corpus_articles':len(corpus),'claims_with_decisive_annotations':len(eligible),'claims_without_decisive_annotations':len(rows)-len(eligible),'article_hit_counts':{str(k):sum(r['article_hit'][str(k)] for r in eligible) for k in (3,10,20)},'claims_with_exact_decisive_sentence_in_top3_context':sum(r['exact_decisive_sentence_hits_top3']>0 for r in eligible),'scope':'Post-hoc recall on the observed cohort and annotation-derived 1344-article pool. NEI annotations do not establish exhaustive evidence absence. Top10/20 only diagnostic; no new verdicts or adoption. Exact sentence matching before tokenizer clipping; full passage entailment not measured.'}
    out.mkdir();write_json_atomic(out/'rows.json',rows);write_json_atomic(out/'summary.json',summary)
    files=[source/'corpus.jsonl',source/'climate-fever.jsonl',ROOT/'artifacts/fresh_climate_rag_preflight_v1/claims.jsonl',Path(__file__).resolve()]
    write_json_atomic(out/'receipt.json',{'inputs':{p.relative_to(ROOT).as_posix():sha256(p) for p in files},'outputs':{n:sha256(out/n) for n in ('rows.json','summary.json')}})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
