"""Freeze compatible fresh claims before neural retrieval/generation."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.fresh_rag_cohort import build_cohort
from apv_rag.splits import sha256,write_json_atomic
from run_fresh_copy_confirmation import check_freeze


def main():
    prior=ROOT/'artifacts/fresh_copy_confirmation_v1'
    out=ROOT/'artifacts/fresh_climate_rag_preflight_v1'
    check_freeze()
    if out.exists():raise FileExistsError('Preflight already frozen; refusing overwrite')
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    claims,gold,excluded=build_cohort(read(prior/'model_inputs.json'),read(prior/'gold.json'))
    if len(claims)!=46 or len(excluded)!=2:raise ValueError('Expected fixed 48-claim parent cohort')
    groups=read(prior/'groups.json')
    files=[prior/'protocol.json',prior/'freeze_receipt.json',prior/'model_inputs.json',prior/'gold.json',prior/'groups.json',
           ROOT/'data/external/climate_fever/frozen_v1/corpus.jsonl',ROOT/'scripts/run_fresh_climate_rag.py',
           ROOT/'src/apv_rag/fresh_rag_cohort.py',Path(__file__).resolve(),ROOT/'docs/FRESH_CLIMATE_RAG.md']
    out.mkdir()
    (out/'claims.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in claims),encoding='utf-8')
    write_json_atomic(out/'gold.json',gold)
    write_json_atomic(out/'groups.json',{r['id']:groups[r['id']] for r in claims})
    write_json_atomic(out/'protocol.json',{'cohort':[r['id'] for r in claims],'excluded_disputed_ids':excluded,
        'claims':len(claims),'groups':len({groups[r['id']] for r in claims}),
        'scope':'Fixed three-class climate RAG on fresh pipeline claims already observed in a different supplied-evidence component test; not a blind independent dataset.',
        'parent_cohort':'All compatible claims from frozen 48-claim/page-disjoint cohort; no outcome-based filtering.',
        'source_authentication_verified':False,'generation_has_run':False,
        'inputs':{p.relative_to(ROOT).as_posix():sha256(p) for p in files}})
    write_json_atomic(out/'freeze_receipt.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    print('Preflight frozen:',len(claims),'claims; neural inference not run.')


if __name__=='__main__':main()
