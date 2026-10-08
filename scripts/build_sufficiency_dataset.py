"""Annotation-derived complete-rationale target; not universal semantic sufficiency."""
import hashlib
import json
import tarfile
from pathlib import Path
from apv_rag.scifact import connected_groups, normalized_claim
from apv_rag.splits import sha256, write_json_atomic


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'data/external/scifact/sealed_v1'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    archive=source/'source_archive.tar.gz'
    assert sha256(archive)==manifest['files']['source_archive.tar.gz']
    with tarfile.open(archive) as t:
        train=list(map(json.loads,t.extractfile('data/claims_train.jsonl').read().decode('utf-8').splitlines()))
    dev=list(map(json.loads,(source/'claims_dev.jsonl').read_text(encoding='utf-8').splitlines()))
    corpus={d['doc_id']:d for d in map(json.loads,(source/'corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    dev_docs={d for r in dev for d in r['cited_doc_ids']} | {int(d) for r in dev for d in r['evidence']}
    dev_text={normalized_claim(r['claim']) for r in dev}
    # Include actual annotated documents in grouping, even if absent from citation lists.
    grouped=[dict(r,cited_doc_ids=sorted(set(r['cited_doc_ids'])|{int(d) for d in r['evidence']})) for r in train]
    grouping_inputs=[dict(r,cited_doc_ids=r["cited_doc_ids"]+["canonical:"+normalized_claim(r["claim"])]) for r in grouped]
    groups=connected_groups(grouping_inputs)
    excluded_groups={g for r,g in zip(grouped,groups,strict=True) if set(r['cited_doc_ids'])&dev_docs or normalized_claim(r['claim']) in dev_text}
    examples=[];parents=[]
    for r,g in zip(grouped,groups,strict=True):
        if g in excluded_groups or not r['evidence']:
            continue
        rationales=[{(int(doc),i) for i in a['sentences']} for doc,anns in r['evidence'].items() for a in anns if a['sentences']]
        if not rationales:
            continue
        decisive=set().union(*rationales)
        background=[]
        for doc in sorted({d for d,_ in decisive}):
            background += [(doc,i) for i in range(len(corpus[doc]['abstract'])) if (doc,i) not in decisive]
        background=background[:2]
        if not background:
            continue  # Avoid an empty-context shortcut for every negative.
        split='validation' if int(hashlib.sha256(f'APV-sufficiency-v1:{g}'.encode()).hexdigest()[:8],16)%5==0 else 'train'
        parent={'claim_id':r['id'],'group_id':g,'split':split,'cited_doc_ids':r['cited_doc_ids']}
        parents.append(parent)
        variants=[('complete',decisive|set(background)),('all_rationales_removed',set(background))]
        # Keep one partial-removal negative only if every alternative rationale is incomplete.
        for removed in sorted(decisive):
            kept=(decisive-{removed})|set(background)
            if not any(s<=kept for s in rationales):
                variants.append(('one_required_sentence_removed',kept));break
        for kind,kept in variants:
            complete=any(s<=kept for s in rationales)
            examples.append({'example_id':f"{r['id']}:{kind}",'claim_id':r['id'],'claim':r['claim'],'group_id':g,'split':split,'construction':kind,'target_complete_rationale':int(complete),'evidence':[{'doc_id':d,'sentence_index':i,'text':corpus[d]['abstract'][i]} for d,i in sorted(kept)],'annotation_rationale_sets':[[{'doc_id':d,'sentence_index':i} for d,i in sorted(s)] for s in rationales]})
    assert examples and {r['split'] for r in examples}=={'train','validation'}
    for row in examples:
        kept={(e['doc_id'],e['sentence_index']) for e in row['evidence']}
        label=any({(e['doc_id'],e['sentence_index']) for e in rationale}<=kept for rationale in row['annotation_rationale_sets'])
        assert label==bool(row['target_complete_rationale'])
        assert not {e['doc_id'] for e in row['evidence']}&dev_docs
    splits={s:{p['group_id'] for p in parents if p['split']==s} for s in ('train','validation')}
    assert not splits['train']&splits['validation']
    documents={s:{d for p in parents if p['split']==s for d in p['cited_doc_ids']} for s in splits}
    assert not documents['train']&documents['validation']
    out=root/'data/processed/sufficiency_annotations_v1';out.mkdir(exist_ok=True)
    write_json_atomic(out/'examples.json',examples)
    write_json_atomic(out/'parents.json',parents)
    summary={'parents':len(parents),'examples':len(examples),'excluded_dev_connected_groups':len(excluded_groups),'split_counts':{s:{'parents':sum(p['split']==s for p in parents),'examples':sum(e['split']==s for e in examples),'positives':sum(e['split']==s and e['target_complete_rationale']==1 for e in examples),'groups':len(splits[s])} for s in splits},'source_archive_sha256':sha256(archive),'code_sha256':sha256(Path(__file__)),'files':{n:sha256(out/n) for n in ('examples.json','parents.json')},'scope':'Constructed complete annotated rationale detection, not universal semantic sufficiency. All derivatives grouped. Dev-connected training components excluded. Climate data unused. Background-only and partial-removal negatives may introduce construction cues. No training or model improvement claim.'}
    write_json_atomic(out/'manifest.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
