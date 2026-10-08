"""Freeze the pre-existing 60-ID sample on the new claim-only XFEVER retrieval contexts."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.multilingual_pipeline import select_sample
from apv_rag.splits import sha256,write_json_atomic
from run_fever_calibration import verify_manifest
from run_xfever_retrieval import prepare,verify
from download_multilingual_nli import MODEL_ID,REVISION


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    out=ROOT/'data/processed/xfever/retrieved_pipeline_v1'
    if out.exists():raise FileExistsError('Sample already frozen')
    received=ROOT/'artifacts/xfever_retrieval_pool_v1'
    sets,identity,pools,queries=prepare()
    verify(received,sets,identity,pools,queries)
    prior=ROOT/'artifacts/xfever_generation_received'
    verify_manifest(prior)
    metadata=load(prior/'input_manifest.json')
    ids=metadata['explanation_claim_ids']
    raw=load(received/'predictions.json')['unicode_words_cjk_1_2']
    rows=select_sample(raw,ids)
    if len(ids)!=60 or len(rows)!=660 or any(sum(r['file']==name for r in rows)!=60 for name in raw):
        raise ValueError('Expected original sixty aligned rows in all eleven files')
    out.mkdir(parents=True)
    write_json_atomic(out/'model_inputs.json',rows)
    targets=load(received/'scoring_targets.json')
    gold=[{'key':r['key'],'file':r['file'],'row':r['row'],'claim_id':r['claim_id'],
           'label':sets[r['file']][r['row']]['label'],
           'target_id':targets[r['file']][r['row']]['target_id']} for r in rows]
    write_json_atomic(out/'gold.json',gold)
    write_json_atomic(out/'selection_manifest.json',{
        'sample_rule':'reuse original label-blind sixty explanation claim IDs; no new outcome-based selection',
        'claim_ids':ids,'queries':660,'rows_per_file':60,
        'retrieval_output_manifest_sha256':sha256(received/'output_manifest.json'),
        'prior_generation_input_manifest_sha256':sha256(prior/'input_manifest.json'),
        'model_inputs_sha256':sha256(out/'model_inputs.json'),'gold_sha256':sha256(out/'gold.json'),
        'limitations':'Observed XFEVER claims and target-derived excerpt pool; exploratory transfer, not independent confirmation.'})
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    enref=load(ROOT/'config/fever_pipeline_reference_v1.json')
    ml=ROOT/'artifacts/xfever_multilingual_received';verify_manifest(ml)
    mlref=load(ml/'input_manifest.json')
    if any(enref['packages'][n]!=mlref['packages'][n] for n in ('torch','transformers','numpy','scikit-learn')):
        raise ValueError('Original model profiles disagree')
    ref=dict(enref,multilingual_nli={'model_id':MODEL_ID,'revision':REVISION,'files':mlref['model']},
             source_multilingual_input_manifest_sha256=sha256(ml/'input_manifest.json'))
    ref['packages']=dict(enref['packages'],**{'huggingface-hub':mlref['packages']['huggingface-hub']})
    path=ROOT/'config/multilingual_retrieved_pipeline_reference_v1.json';write_json_atomic(path,ref)
    print('Sample SHA256:',sha256(out/'selection_manifest.json'))
    print('Reference SHA256:',sha256(path))
    print('Frozen 660 contexts: original sixty aligned IDs across eleven files. No new sampling by labels.')


if __name__=='__main__':main()
