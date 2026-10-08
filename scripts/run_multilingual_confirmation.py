"""Run a fixed new-claim multilingual confidence confirmation using existing local models."""
import argparse
import contextlib
import gc
import io
import sqlite3
import sys
import zipfile
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.multilingual_confirmation import apply_frozen,summarize_confirmation
from apv_rag.multilingual_pipeline import prepare_context,validate_multilingual_entry
from apv_rag.fresh_pipeline import validate_feature_entry
from apv_rag.multilingual_io import write_json_atomic,infer_english
from apv_rag.splits import sha256
from apv_rag.nli_comparison import _model_files
from prepare_multilingual_confirmation import verify,OUT
from run_multilingual_retrieved_pipeline import (load,require,check_manifest,validate_prepared,
    infer_multilingual,evaluate,replay_data,prediction_binding,score_outputs,CODE,SETTINGS)
from run_fresh_pipeline import make_backend
from verify_fresh_pipeline import equal

# Filled from the replayed preparation receipt before distribution.
SAMPLE_SHA='8d07d24f1ec200a06a13290b12382f3a63fc8f3b0af742d57a4b5ed365b8d45c'
CONFIRM_SETTINGS=dict(SETTINGS,retrieval='frozen full official test paired-excerpt pools; unchanged CJK BM25 top3',
    correctness_calibration='four frozen prior-development logistic models plus prevalence controls')
LIMITATIONS=[
    'One hundred project-held-out claims across six parallel variants; there are not 600 independent observations.',
    'The five new non-English variants are upstream machine translations; human-translated transfer is not confirmed.',
    'Full-test target-derived excerpt pools contain targets by construction; no full-page/open-web retrieval claim.',
    'The larger retrieval pool and machine-only confirmation distribution differ from the observed development cohort.',
    'Project-level ID/text/page/excerpt separation does not prove absence of pretrained-model FEVER exposure.',
    'English-trained gates and lexical numeric checks remain unchanged; no publisher authentication or explanation-truth claim.',
    'Replay validates saved inference bindings and calculations; it does not independently rerun neural models.']


def load_inputs():
    require(sha256(OUT/'selection_manifest.json')==SAMPLE_SHA,'Frozen confirmation sample changed')
    # Apply the delivered fits exactly; never refit under a different Windows sklearn version.
    manifest,rows=verify(refit=False)
    ref=load(ROOT/'config/multilingual_retrieved_pipeline_reference_v1.json')
    require(len(rows)==600 and len({r['claim_id'] for r in rows})==100,'Expected 100 claims/600 queries')
    fits=load(OUT/'calibrators.json')
    require(not set(manifest['claim_ids'])&set(fits['development_claim_ids']),'Development IDs overlap confirmation')
    return manifest,rows,ref,fits


def identity(manifest,ref):
    names=tuple(dict.fromkeys((*CODE,'scripts/run_multilingual_confirmation.py',
        'scripts/prepare_multilingual_confirmation.py','src/apv_rag/multilingual_confirmation.py',
        'src/apv_rag/multilingual_calibration.py','scripts/analyze_multilingual_calibration.py')))
    return {'sample_manifest_sha256':SAMPLE_SHA,'prepared_output_manifest_sha256':sha256(OUT/'output_manifest.json'),
        'calibrators_sha256':sha256(OUT/'calibrators.json'),'reference_sha256':manifest['reference_sha256'],
        'packages':ref['packages'],'settings':CONFIRM_SETTINGS,
        'code_sha256':{n:sha256(ROOT/n) for n in names},
        'protocol_sha256':sha256(ROOT/'docs/MULTILINGUAL_CONFIRMATION.md')}


def binding(out):
    return prediction_binding(out)|{'correctness_predictions.json':sha256(out/'correctness_predictions.json')}


def score(rows,direct,confidence,gold,responses):
    controller=score_outputs(rows,direct,gold,responses)
    return {'scope':'fixed new-claim multilingual closed-pool confirmation; no confirmation fitting',
        'queries':len(gold),'policy_records':len(rows),'distinct_responses':len(responses),
        'controller_metrics_by_file':controller['metrics_by_file'],
        'correctness_calibration':summarize_confirmation(confidence,gold),'limitations':LIMITATIONS}


def audit(out,manifest,rows,ref,fits):
    check_manifest(out,coverage=True)
    require(load(out/'input_manifest.json')==identity(manifest,ref),'Confirmation producer identity differs')
    check_manifest(out,'prediction_manifest.json')
    require(load(out/'prediction_manifest.json')==binding(out),'Prediction/confidence coverage differs')
    with contextlib.redirect_stdout(io.StringIO()):predicted,direct,responses=replay_data(out,rows,ref['gate_models'])
    confidence=apply_frozen(predicted,fits)
    require(equal(confidence,load(out/'correctness_predictions.json')),'Frozen confidence replay differs')
    require(load(out/'scoring_manifest.json')=={'gold_sha256':sha256(OUT/'gold.json'),
        'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),'confirmation_fitting':False},'Scoring binding differs')
    summary=score(predicted,direct,confidence,load(OUT/'gold.json'),responses)
    require(equal(summary,load(out/'summary.json')),'Confirmation metrics differ')
    return summary


def export(out):
    dest=ROOT/'multilingual_confirmation_outputs.zip';temp=dest.with_suffix('.partial.zip')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():z.write(p,'multilingual_confirmation_v1/'+p.name)
    with zipfile.ZipFile(temp) as z:require(z.testzip() is None,'Output ZIP integrity failed')
    temp.replace(dest);print(f'Upload this result ZIP: {dest}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--verify',action='store_true')
    parser.add_argument('--output',default='artifacts/multilingual_confirmation_v1');args=parser.parse_args()
    manifest,rows,ref,fits=load_inputs();out=(ROOT/args.output).resolve()
    require(out.is_relative_to(ROOT/'artifacts'),'Output must be inside artifacts')
    if args.preflight:
        print('Preparation replay passed: 100 unused IDs / 600 queries; four frozen development-only calibrators. Neural run NOT performed.');return
    if args.verify or (out/'output_manifest.json').exists():
        audit(out,manifest,rows,ref,fits);print('Saved confirmation responses, policies, fixed confidence and metrics replayed.');export(out);return
    packages={n:version(n) for n in ref['packages']}
    require(packages==ref['packages'],f'Pinned model environment required: {ref["packages"]}; got {packages}')
    print('Checking existing local models; no downloads.',flush=True)
    for name,path in [('generator','qwen2.5-1.5b-instruct'),('nli','nli-deberta-v3-small'),
        ('embedding','all-MiniLM-L6-v2'),('multilingual','mdeberta-multilingual-nli')]:
        expected=(ref['generator']['files'] if name=='generator' else ref['multilingual_nli']['files']
            if name=='multilingual' else ref['feature_models'][name])
        require(_model_files(ROOT/'models'/path)==expected,f'Model bytes changed: {name}')
    out.mkdir(parents=True,exist_ok=True);meta=identity(manifest,ref);receipt=out/'input_manifest.json'
    if receipt.exists():require(load(receipt)==meta,'Resume identity changed; refusing caches')
    else:require(not any(out.iterdir()),'Cache exists without identity')
    write_json_atomic(receipt,meta)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/qwen2.5-1.5b-instruct',local_files_only=True)
    def clip(text,limit):
        tokens=tokenizer.encode(text,add_special_tokens=False)
        return tokenizer.decode(tokens[:limit],skip_special_tokens=True),len(tokens),min(len(tokens),limit)
    prepared={r['key']:prepare_context(r,clip) for r in rows};validate_prepared(rows,prepared)
    path=out/'prepared_contexts.json'
    if path.exists():require(load(path)==prepared,'Tokenized context changed')
    write_json_atomic(path,prepared);del tokenizer;gc.collect()
    enpath=out/'feature_cache.json';mlpath=out/'multilingual_feature_cache.json'
    en=load(enpath) if enpath.exists() else {};ml=load(mlpath) if mlpath.exists() else {}
    expected={k for k,r in prepared.items() if r['evidence']}
    require(set(en)<=expected and set(ml)<=expected,'Unexpected cached feature keys')
    for key in en:validate_feature_entry(en[key],prepared[key]['shown_claim'],prepared[key]['evidence'])
    for key in ml:validate_multilingual_entry(prepared[key],ml[key])
    infer_english(prepared,en,enpath,root=ROOT);write_json_atomic(enpath,en);gc.collect()
    infer_multilingual(prepared,ml,mlpath);write_json_atomic(mlpath,ml);gc.collect()
    with closing(sqlite3.connect(out/'generation_cache.sqlite')) as db:
        db.execute('CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT)')
        generate,used=make_backend(ROOT,ref['generator'],db,seeds={})
        policies,direct=evaluate(rows,prepared,en,ml,ref['gate_models'],generate)
        write_json_atomic(out/'responses_used.json',used)
    write_json_atomic(out/'predictions.json',policies);write_json_atomic(out/'direct_predictions.json',direct)
    write_json_atomic(out/'correctness_predictions.json',apply_frozen(policies,fits))
    # The neural prediction path never reads confirmation gold; scoring follows frozen outputs.
    write_json_atomic(out/'prediction_manifest.json',binding(out))
    write_json_atomic(out/'summary.json',score(policies,direct,load(out/'correctness_predictions.json'),load(OUT/'gold.json'),used))
    write_json_atomic(out/'scoring_manifest.json',{'gold_sha256':sha256(OUT/'gold.json'),
        'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),'confirmation_fitting':False})
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir()
        if p.is_file() and p.name!='output_manifest.json'})
    audit(out,manifest,rows,ref,fits);print('Confirmation completed and saved-output replay passed. No refit.');export(out)


if __name__=='__main__':main()
