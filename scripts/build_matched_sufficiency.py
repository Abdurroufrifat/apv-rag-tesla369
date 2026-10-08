"""Matched sentence-count/word-count construction; annotation coverage target only."""
import json
from pathlib import Path
from apv_rag.splits import sha256,write_json_atomic


def main():
    root=Path(__file__).resolve().parents[1]
    original=root/'data/processed/sufficiency_annotations_v1'
    manifest=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
    rows=json.loads((original/'examples.json').read_text(encoding='utf-8'))
    parents=json.loads((original/'parents.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():assert sha256(original/name)==digest
    positives=[r for r in rows if r['construction']=='complete']
    pools={s:{} for s in ('train','validation')}
    for row in positives:
        for e in row['evidence']:
            pools[row['split']][(e['doc_id'],e['sentence_index'])]=e
    matched=[];skipped=0
    for positive in positives:
        gold={(e['doc_id'],e['sentence_index']) for rationale in positive['annotation_rationale_sets'] for e in rationale}
        own_docs={e['doc_id'] for e in positive['evidence']}
        replaced=[];used=set();ok=True
        for e in positive['evidence']:
            if (e['doc_id'],e['sentence_index']) not in gold:
                replaced.append(e);continue
            words=len(e['text'].split())
            candidates=[(key,d) for key,d in pools[positive['split']].items() if key[0] not in own_docs and key not in used and len(d['text'].split())==words]
            if not candidates:ok=False;break
            key,d=min(candidates,key=lambda pair:pair[0]);used.add(key);replaced.append(d)
        if not ok:skipped+=1;continue
        negative=dict(positive,example_id=f"{positive['claim_id']}:matched_distractor",construction='matched_distractor',target_complete_rationale=0,evidence=replaced)
        original_keys={(e['doc_id'],e['sentence_index']) for e in negative['evidence']}
        assert not any({(e['doc_id'],e['sentence_index']) for e in rationale}<=original_keys for rationale in positive['annotation_rationale_sets'])
        assert len(positive['evidence'])==len(negative['evidence'])
        assert [len(e['text'].split()) for e in positive['evidence']]==[len(e['text'].split()) for e in negative['evidence']]
        matched.extend([positive,negative])
    docs={s:{e['doc_id'] for r in matched if r['split']==s for e in r['evidence']} for s in pools}
    assert not docs['train']&docs['validation']
    out=root/'data/processed/sufficiency_matched_v2';out.mkdir(exist_ok=True)
    write_json_atomic(out/'examples.json',matched)
    ids={r['claim_id'] for r in matched};write_json_atomic(out/'parents.json',[p for p in parents if p['claim_id'] in ids])
    write_json_atomic(out/'manifest.json',{'source_manifest_sha256':sha256(original/'manifest.json'),'code_sha256':sha256(Path(__file__)),'examples':len(matched),'skipped_unmatched_parents':skipped,'split_counts':{s:sum(r['split']==s for r in matched) for s in pools},'files':{n:sha256(out/n) for n in ('examples.json','parents.json')},'scope':'Exact per-sentence word-count matched pairs; same-split external-document distractors. Derived annotation coverage task, not real-world semantic sufficiency. Shared distractors within a split create dependence; document leakage across splits prohibited.'})
    print(f'Matched {len(matched)} examples; skipped {skipped} parents')


if __name__=='__main__':main()
