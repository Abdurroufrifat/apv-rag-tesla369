"""Freeze unused official XFEVER test claims and prior-development confidence models."""
import argparse
import json
import sys
import tarfile
from pathlib import Path
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.multilingual_confirmation import FILES,select_new,page_key,fit_development
from apv_rag.multilingual_retrieval import normalize,excerpt_id,build_pool,retrieve
from apv_rag.multilingual_pipeline import select_sample
from apv_rag.splits import sha256
from apv_rag.multilingual_io import write_json_atomic
from analyze_multilingual_calibration import inputs,fits_equal
from run_fever_calibration import verify_manifest

SOURCE=ROOT/'data/external/xfever/zenodo_8206962'
OUT=ROOT/'data/processed/xfever/confirmation_v1'
ARCHIVE_SHA='8b7948894c8724d9a52e86e18fe0e369c5d58b12ddba853e836b8d80611a4895'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def previous_inputs():
    paths=sorted((ROOT/'data/external/fever').rglob('model_inputs.jsonl'))
    paths+=sorted((SOURCE/'evaluation_inputs_v1').rglob('*.jsonl'))
    paths+=sorted((ROOT/'data/external/fever').rglob('gold.jsonl'))
    ids=set();texts=set();pages=set();excerpts=set()
    for path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);ids.add(r['id'])
            if 'claim' in r:texts.add(normalize(r['claim']))
            if 'page' in r:pages.add(page_key(r['page']))
            if isinstance(r.get('evidence'),str):excerpts.add(excerpt_id(r['evidence']))
            elif isinstance(r.get('evidence'),list):
                for group in r['evidence']:
                    for e in group:
                        if len(e)>=4 and isinstance(e[2],str):pages.add(page_key(e[2]))
    return {'ids':sorted(ids),'normalized_claims':sorted(texts),'normalized_pages':sorted(pages),
        'excerpt_ids':sorted(excerpts),'files':{p.relative_to(ROOT).as_posix():sha256(p) for p in paths}}


def compute(raw,exclusions):
    sets={name:[json.loads(line) for line in (raw/name).read_text(encoding='utf-8').splitlines()] for name in FILES}
    indices=select_new(sets,set(exclusions['ids']),set(exclusions['normalized_claims']),
        set(exclusions['normalized_pages']),set(exclusions['excerpt_ids']),100)
    ids=[sets[FILES[0]][i]['id'] for i in indices]
    pools={name:build_pool([r['evidence'] for r in rows]) for name,rows in sets.items()}
    queries={name:[{'row':i,'claim_id':sets[name][i]['id'],'claim':sets[name][i]['claim']} for i in indices] for name in FILES}
    retrieved={name:retrieve(queries[name],pools[name],'unicode_words_cjk_1_2') for name in FILES}
    rows=select_sample(retrieved,ids)
    gold=[{'key':r['key'],'file':r['file'],'row':r['row'],'claim_id':r['claim_id'],
        'label':sets[r['file']][r['row']]['label'],'target_id':excerpt_id(sets[r['file']][r['row']]['evidence'])} for r in rows]
    for i in indices:
        if len({sets[name][i]['label'] for name in FILES})!=1:raise ValueError('Parallel target labels differ')
    return indices,ids,pools,rows,gold


def verify(*,refit=True):
    verify_manifest(OUT)
    receipt=load(OUT/'selection_manifest.json')
    if set(load(OUT/'output_manifest.json'))!={p.name for p in OUT.iterdir() if p.is_file() and p.name!='output_manifest.json'}:
        raise ValueError('Preparation receipt coverage differs')
    if receipt['protocol_sha256']!=sha256(ROOT/'docs/MULTILINGUAL_CONFIRMATION.md'):
        raise ValueError('Confirmation protocol changed')
    for name,digest in receipt['code_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Confirmation preparation code changed')
    excluded=previous_inputs()
    if load(OUT/'exclusions.json')!=excluded:raise ValueError('Prior-used input inventory changed')
    for name,digest in receipt['official_test_sha256'].items():
        if sha256(SOURCE/'confirmation_inputs_v1'/name)!=digest:raise ValueError('Official test bytes changed')
    indices,ids,pools,rows,gold=compute(SOURCE/'confirmation_inputs_v1',excluded)
    if (receipt['selected_rows']!=indices or receipt['claim_ids']!=ids or
        load(OUT/'model_inputs.json')!=rows or load(OUT/'gold.json')!=gold or load(OUT/'corpora.json')!=pools):
        raise ValueError('Label-blind selection or retrieval replay differs')
    prior,development,identity=inputs()
    if receipt['development_identity']!=identity or sha256(OUT/'calibrators.json')!=receipt['calibrators_sha256']:
        raise ValueError('Frozen development source or calibrator bytes differ')
    if refit and not fits_equal(load(OUT/'calibrators.json'),fit_development(prior,development)):
        raise ValueError('Development-only frozen calibrators differ')
    ref=ROOT/'config/multilingual_retrieved_pipeline_reference_v1.json'
    if receipt['reference_sha256']!=sha256(ref):raise ValueError('Model reference changed')
    return receipt,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');args=p.parse_args()
    if args.verify:
        _,rows=verify();print(f'Unused-claim selection, exclusion, retrieval and development-only fits replayed: {len(rows)} queries.');return
    if OUT.exists():raise FileExistsError('Confirmation already frozen; use --verify')
    prior,gold,identity=inputs();fits=fit_development(prior,gold)
    OUT.mkdir(parents=True)
    # Freeze only prior development outcomes before opening any new target records.
    write_json_atomic(OUT/'calibrators.json',fits)
    raw=SOURCE/'confirmation_inputs_v1';raw.mkdir()
    archive=SOURCE/'data.tar.gz'
    if sha256(archive)!=ARCHIVE_SHA:raise ValueError('Official archive checksum differs')
    with tarfile.open(archive,'r:gz') as t:
        for name in FILES:
            m=t.getmember('data/'+name)
            if not m.isfile() or m.size>6*1024*1024:raise ValueError('Unexpected test file')
            dest=raw/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(t.extractfile(m).read())
    excluded=previous_inputs();write_json_atomic(OUT/'exclusions.json',excluded)
    indices,ids,pools,rows,gold=compute(raw,excluded)
    write_json_atomic(OUT/'corpora.json',pools);write_json_atomic(OUT/'model_inputs.json',rows)
    write_json_atomic(OUT/'gold.json',gold)
    receipt={'claim_ids':ids,'selected_rows':indices,'claims':100,'queries':600,
        'selection':'SHA256 apv-multilingual-confirmation:369:<id>; reject prior IDs/text/pages/excerpts and shared new pages; no label balancing',
        'archive_sha256':ARCHIVE_SHA,'official_test_sha256':{n:sha256(raw/n) for n in FILES},
        'pool_sizes':{n:len(v) for n,v in pools.items()},'development_identity':identity,
        'development_claims':len(fits['development_claim_ids']),'calibrators_sha256':sha256(OUT/'calibrators.json'),
        'calibrator_runtime':{'numpy':version('numpy'),'scikit-learn':version('scikit-learn')},
        'reference_sha256':sha256(ROOT/'config/multilingual_retrieved_pipeline_reference_v1.json'),
        'code_sha256':{n:sha256(ROOT/n) for n in ('scripts/prepare_multilingual_confirmation.py',
            'src/apv_rag/multilingual_confirmation.py','src/apv_rag/multilingual_retrieval.py','src/apv_rag/retrieval.py')},
        'protocol_sha256':sha256(ROOT/'docs/MULTILINGUAL_CONFIRMATION.md'),
        'status':'frozen inputs and development-only calibrators; confirmation neural experiment NOT RUN'}
    write_json_atomic(OUT/'selection_manifest.json',receipt)
    write_json_atomic(OUT/'output_manifest.json',{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()})
    verify();print('Frozen and replayed: 100 unused claims, 600 queries, four development-only confidence models. Neural run pending.')


if __name__=='__main__':main()
