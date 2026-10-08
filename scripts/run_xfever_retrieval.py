"""Retrieve XFEVER excerpt pools without query-specific evidence, then score saved outputs."""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.multilingual_retrieval import METHODS,build_pool,excerpt_id,retrieve,score
from apv_rag.splits import sha256,write_json_atomic
from run_fever_calibration import verify_manifest
from run_xfever_generation import FILES,prepare_inputs
from verify_fresh_pipeline import equal

CODE=('scripts/run_xfever_retrieval.py','src/apv_rag/multilingual_retrieval.py',
      'src/apv_rag/retrieval.py','scripts/run_xfever_generation.py','src/apv_rag/xfever.py',
      'scripts/verify_fresh_pipeline.py')
SETTINGS={'methods':list(METHODS),'top_k':3,'k1':1.2,'b':0.75,
    'corpus':'deduplicated paired evidence texts within each file; no page titles, labels or claim-specific target IDs',
    'query':'row, claim_id and claim only','score_replay_tolerance':1e-12,'tie_rule':'ascending normalized-excerpt SHA256',
    'normalization':'NFKC, casefold, whitespace collapse for dedup and CJK method',
    'pool_contains_targets':True,'new_inference':False,'new_fitting':False,
    'primary_metric':'paired target excerpt hit at 3 for SUPPORTS/REFUTES rows',
    'nei':'paired target matching descriptive only; not evidence completeness or verdict correctness'}


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def require(ok,message):
    if not ok:raise ValueError(message)


def prepare():
    sets,files,_,dataset_sha=prepare_inputs(ROOT)
    identity={'dataset_manifest_sha256':dataset_sha,'files':files,'settings':SETTINGS,
        'code_sha256':{n:sha256(ROOT/n) for n in CODE},
        'protocol_sha256':sha256(ROOT/'docs/XFEVER_RETRIEVAL_POOL.md')}
    pools={name:build_pool([r['evidence'] for r in rows]) for name,rows in sets.items()}
    queries={name:[{'row':i,'claim_id':r['id'],'claim':r['claim']} for i,r in enumerate(rows)]
             for name,rows in sets.items()}
    return sets,identity,pools,queries


def score_saved(predictions,sets,pools):
    targets={name:[{'row':i,'claim_id':r['id'],'target_id':excerpt_id(r['evidence']),'label':r['label']}
                   for i,r in enumerate(rows)] for name,rows in sets.items()}
    metrics={method:{name:score(predictions[method][name],targets[name]) for name in FILES} for method in METHODS}
    english=sets[FILES[0]]
    summary={'scope':'exploratory closed excerpt-pool multilingual retrieval; observed XFEVER cohorts',
        'files':len(FILES),'queries_per_method':sum(map(len,sets.values())),
        'underlying_english_pairs':len(english),'underlying_english_claim_ids':len({r['id'] for r in english}),
        'pool_sizes':{n:len(pools[n]) for n in FILES},'metrics':metrics,
        'limitations':['The corpus is assembled from paired benchmark target excerpts and necessarily contains every target.',
            'This is an optimistic limited-pool retrieval diagnostic, not full-page or open-web multilingual search.',
            'Repeated translated variants share English claims and must not be treated as independent samples.',
            'NEI pairs can contain unrelated text; their target-matching rates are not factual verification accuracy.',
            'Exact normalized text IDs establish excerpt identity only, not publisher authentication or provenance independence.',
            'No NLI, embedding, generator, gate or confidence model was run; end-to-end multilingual transfer remains open.',
            'The two fixed analyzers are compared on already observed data; no significance, novel-method or independent-confirmation claim is made.']}
    return targets,summary


def verify(out,sets,identity,pools,queries):
    verify_manifest(out)
    hashes=load(out/'output_manifest.json')
    expected={p.name for p in out.iterdir() if p.is_file() and p.name!='output_manifest.json'}
    require(set(hashes)==expected,'Output manifest coverage differs')
    require(load(out/'input_manifest.json')['identity']==identity,'Frozen input/code/protocol identity differs')
    require(load(out/'corpora.json')==pools and load(out/'queries.json')==queries,'Corpus or model-only query replay differs')
    replay={method:{name:retrieve(queries[name],pools[name],method) for name in FILES} for method in METHODS}
    require(equal(load(out/'predictions.json'),replay),'Retrieval ordering/text/score replay differs')
    targets,summary=score_saved(replay,sets,pools)
    require(load(out/'scoring_targets.json')==targets and load(out/'summary.json')==summary,'Target/metric replay differs')
    require(load(out/'prediction_manifest.json')=={
        'predictions.json':sha256(out/'predictions.json'),'corpora.json':sha256(out/'corpora.json'),
        'queries.json':sha256(out/'queries.json')},'Frozen retrieval output binding differs')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--output',default='artifacts/xfever_retrieval_pool_v1')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    require(out.is_relative_to(ROOT/'artifacts'),'Output must be inside artifacts')
    sets,identity,pools,queries=prepare()
    if args.verify:
        summary=verify(out,sets,identity,pools,queries)
        print(f'Retrieval IDs/text/ranks reproduced; score tolerance 1e-12: {summary["queries_per_method"]} queries per analyzer.')
        return
    require(not out.exists(),'Completed or incomplete output exists; use --verify or a new output path')
    out.mkdir()
    write_json_atomic(out/'input_manifest.json',{'identity':identity,'producer_runtime':{
        'python':sys.version,'unicode_database':unicodedata.unidata_version}})
    write_json_atomic(out/'corpora.json',pools)
    write_json_atomic(out/'queries.json',queries)
    predictions={}
    for method in METHODS:
        predictions[method]={}
        for name in FILES:
            predictions[method][name]=retrieve(queries[name],pools[name],method)
            print(f'{method}: {name}, {len(queries[name])} claim-only queries, pool {len(pools[name])}',flush=True)
    write_json_atomic(out/'predictions.json',predictions)
    write_json_atomic(out/'prediction_manifest.json',{
        'predictions.json':sha256(out/'predictions.json'),'corpora.json':sha256(out/'corpora.json'),
        'queries.json':sha256(out/'queries.json')})
    # Target IDs and labels are used for scoring only after the retrieval file is frozen.
    targets,summary=score_saved(predictions,sets,pools)
    write_json_atomic(out/'scoring_targets.json',targets)
    write_json_atomic(out/'summary.json',summary)
    lines=['# XFEVER multilingual excerpt-pool retrieval','',
        'Each file has 600 claim-only queries. Corpus documents are pooled paired evidence excerpts from that file, '
        'deduplicated by normalized text, with no labels, page titles or query-specific target metadata passed to retrieval. '
        'Every target is present by construction. This is an optimistic closed-pool diagnostic and does not evaluate '
        'full Wikipedia articles, open-web retrieval or the complete multilingual APV-RAG controller.','',
        '| File | Unique excerpts | Verifiable pairs | Stock word BM25 hit@3 | Words + CJK characters/bigrams hit@3 |',
        '|---|---:|---:|---:|---:|']
    for name in FILES:
        a=summary['metrics'][METHODS[1]][name]['verifiable'];b=summary['metrics'][METHODS[0]][name]['verifiable']
        lines.append(f'| {name} | {len(pools[name])} | {b["queries"]} | {a["target_hit_at_3"]:.2%} | {b["target_hit_at_3"]:.2%} |')
    lines+=['',f'There are {summary["queries_per_method"]} query rows per analyzer, representing '
        f'{summary["underlying_english_pairs"]} aligned English pairs and {summary["underlying_english_claim_ids"]} unique English claim IDs. '
        'The translated variants overlap by underlying claim. Hit@1, reciprocal rank at three, NEI descriptive matching '
        'and empty retrieval counts are available in summary.json. Retrieval outputs contain no target IDs or labels.','',
        'The added tokenizer handles Chinese/Japanese text without requiring whitespace; it uses the same fixed BM25 '
        'parameters and tie rule as the stock-token control. This is a preprocessing control, not a novel retrieval model. '
        'No tuning or preferred-analyzer selection is performed after scoring.','',*summary['limitations'],'',
        'No new human annotation, model download, manuscript or GitHub push is included.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    print('\n'.join(lines[:18]))
    print(out/'RESULTS.md')


if __name__=='__main__':main()
