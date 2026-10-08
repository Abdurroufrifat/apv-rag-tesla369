"""Run fixed FEVER retrieval/features/generation/gates and development-only correctness fitting."""

import argparse
import gc
import json
import sqlite3
import sys
import zipfile
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.fever_calibration import normalized
from apv_rag.fever_corpus import EXPECTED_ARCHIVE_SHA256, EXPECTED_PAGES, retrieve_claims
from apv_rag.fever_nli import LABEL_MAP, read_jsonl, validate_contexts
from apv_rag.fever_pipeline import (POLICIES, answer_score, correctness_probability,
                                    fit_correctness, summarize_policy)
from apv_rag.fresh_pipeline import execute_policies, validate_feature_entry
from apv_rag.integrated_gate import collapse_context
from apv_rag.nli_comparison import _model_files
from apv_rag.splits import sha256, write_json_atomic
from run_fever_calibration import verify_manifest
from run_fresh_pipeline import infer_missing, make_backend

COHORT_SHA256 = '369b116b80ad1767a8ad036583d95fbecdb488caa2c3d790f3dc6f91d3f04a9b'
REFERENCE_SHA256 = '1b55fb1af7a2248ea820b8b9f4ed223f5d2522fe4eca19c184f7ef841c291eb2'
CODE = ('scripts/run_fever_pipeline.py', 'src/apv_rag/fever_pipeline.py',
        'src/apv_rag/fresh_pipeline.py', 'src/apv_rag/fever_corpus.py',
        'src/apv_rag/gated_generation_flow.py', 'src/apv_rag/integrated_gate.py',
        'src/apv_rag/numeric_integrity_v2.py', 'src/apv_rag/generative_interface.py',
        'src/apv_rag/generative_rag.py', 'src/apv_rag/sentence_context.py',
        'src/apv_rag/nli_comparison.py', 'src/apv_rag/multilingual_nli.py',
        'scripts/run_fresh_pipeline.py', 'scripts/run_gated_generation.py',
        'src/apv_rag/direct_nli.py', 'src/apv_rag/fever_nli.py',
        'src/apv_rag/fever_calibration.py', 'src/apv_rag/splits.py', 'scripts/run_fever_calibration.py')
SETTINGS = {'seed': 369, 'threads': 4, 'dtype': 'float32', 'top_pages': 3,
            'top_sentences': 3, 'claim_tokens': 64, 'passage_tokens': 96,
            'nli_tokens': 256, 'embedding_tokens': 384, 'gate_threshold': 0.5,
            'max_input_tokens': 1024, 'max_new_tokens': 128,
            'do_sample': False, 'num_beams': 1, 'policies': list(POLICIES),
            'correctness_fit': 'policy-specific logistic on logit NLI score assigned to final generated label; C=1',
            'single_outcome_fallback': 'Beta(1,1) smoothed development correctness prevalence',
            'no_development_answers': 'confidence unavailable; no invented calibration',
            'prior_generator_response_reuse': False}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_inputs():
    source = ROOT / 'data/external/fever/pipeline_v1'
    reference = ROOT / 'config/fever_pipeline_reference_v1.json'
    if sha256(source / 'selection_manifest.json') != COHORT_SHA256 or sha256(reference) != REFERENCE_SHA256:
        raise ValueError('Frozen pipeline cohort/reference differs')
    manifest, ref = load(source / 'selection_manifest.json'), load(reference)
    claims = {}
    for name in ('development', 'confirmation'):
        m = manifest['cohorts'][name]
        path = source / name / 'model_inputs.jsonl'
        if sha256(path) != m['model_inputs_sha256']:
            raise ValueError('Model-only claim checksum differs')
        claims[name] = read_jsonl(path)
        if len(claims[name]) != 300 or [r['id'] for r in claims[name]] != m['selected_ids']:
            raise ValueError('Pipeline cohort alignment differs')
    rows = claims['development'] + claims['confirmation']
    if (len({r['id'] for r in rows}) != 600 or len({normalized(r['claim']) for r in rows}) != 600 or
            {r['id'] for r in rows} & set(manifest['excluded_ids']) or
            {normalized(r['claim']) for r in rows} & set(manifest['excluded_texts'])):
        raise ValueError('New cohorts overlap each other or previous outcomes')
    return source, manifest, ref, claims


def verify_stage(folder, identity):
    verify_manifest(folder)
    if load(folder / 'input_manifest.json') != identity:
        raise ValueError('Completed stage identity differs')
    return load(folder / 'predictions.json')


def run_stage(name, claims, source, out, db, ref, identity):
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    stage_identity = dict(identity, cohort=name, claims_sha256=sha256(source / name / 'model_inputs.jsonl'))
    receipt = folder / 'input_manifest.json'
    if receipt.exists() and load(receipt) != stage_identity:
        raise ValueError('Stage identity changed; refusing saved caches')
    if (folder / 'output_manifest.json').exists():
        return verify_stage(folder, stage_identity)
    if not receipt.exists() and any(folder.iterdir()):
        raise ValueError('Stage files exist without input identity')
    write_json_atomic(receipt, stage_identity)
    raw = folder / 'retrieved_contexts.jsonl'
    if raw.exists():
        if not (folder / 'retrieval_manifest.json').exists():
            validate_contexts(claims, read_jsonl(raw))
            write_json_atomic(folder / 'retrieval_manifest.json', {raw.name: sha256(raw)})
        verify_manifest(folder, 'retrieval_manifest.json')
    else:
        retrieve_claims(db, source / name / 'model_inputs.jsonl', raw)
        write_json_atomic(folder / 'retrieval_manifest.json', {raw.name: sha256(raw)})
    rows = validate_contexts(claims, read_jsonl(raw))
    from transformers import AutoTokenizer

    gt = AutoTokenizer.from_pretrained(ROOT / 'models/qwen2.5-1.5b-instruct', local_files_only=True)
    def clip(text, limit):
        return gt.decode(gt.encode(text, add_special_tokens=False)[:limit], skip_special_tokens=True)
    prepared = {}
    for row in rows:
        evidence = [dict(d, text=clip(d['text'], 96)) for d in row['evidence'] if d['text'].strip()]
        evidence = [d for d in evidence if d['text'].strip()]
        prepared[str(row['id'])] = {'claim': row['claim'], 'shown_claim': clip(row['claim'], 64),
            'retrieved_evidence': evidence, 'evidence': collapse_context(evidence)}
    prepared_path = folder / 'prepared_contexts.json'
    if prepared_path.exists() and load(prepared_path) != prepared:
        raise ValueError('Tokenized source context changed')
    write_json_atomic(prepared_path, prepared)
    cachepath = folder / 'feature_cache.json'
    cache = load(cachepath) if cachepath.exists() else {}
    if not set(cache) <= set(prepared):
        raise ValueError('Unexpected feature cache claims')
    for key, entry in cache.items():
        validate_feature_entry(entry, prepared[key]['shown_claim'], prepared[key]['evidence'])
    print(f'{name}: checking/computing context-bound NLI and embedding features.', flush=True)
    infer_missing(prepared, cache, cachepath, root=ROOT)
    write_json_atomic(cachepath, cache)
    del gt
    gc.collect()
    predictions = []
    with closing(sqlite3.connect(folder / 'generation_cache.sqlite')) as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT)')
        generate, used = make_backend(ROOT, ref['generator'], connection, seeds={})
        for i, claim in enumerate(claims, 1):
            key = str(claim['id'])
            for row in execute_policies(prepared[key], cache.get(key), ref['gate_models'], generate):
                row.update(claim_id=claim['id'], claim=claim['claim'], shown_claim=prepared[key]['shown_claim'],
                           retrieved_evidence=prepared[key]['retrieved_evidence'],
                           context_sha256=cache[key]['context_sha256'] if key in cache else None,
                           score=answer_score(row, cache.get(key)))
                predictions.append(row)
            if i == 1 or i % 10 == 0:
                print(f'{name}: gate + verdict + explanation {i}/{len(claims)}', flush=True)
        write_json_atomic(folder / 'responses_used.json', used)
    write_json_atomic(folder / 'predictions.json', predictions)
    write_json_atomic(folder / 'output_manifest.json', {p.name: sha256(p)
        for p in folder.iterdir() if p.is_file() and p.name != 'output_manifest.json'})
    print(f'{name}: {len(predictions)} policy records frozen; no gold opened during inference.', flush=True)
    return predictions


def calibrate_development(rows, gold):
    if len(rows) != len(gold)*4:
        raise ValueError('Development policy record count differs')
    fits = {}
    for policy in POLICIES:
        subset = [r for r in rows if r['policy'] == policy]
        if [r['claim_id'] for r in subset] != [r['id'] for r in gold]:
            raise ValueError('Development policy/gold alignment differs')
        accepted = [(r['score'], r['candidate_label'] == LABEL_MAP[g['label']])
                    for r, g in zip(subset, gold, strict=True) if r['candidate_label'] is not None]
        fits[policy] = fit_correctness([a for a, _ in accepted], [b for _, b in accepted])
    return fits


def export(out):
    path = ROOT / 'fever_pipeline_outputs.zip'
    temporary = path.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, 'fever_pipeline_v1/' + p.relative_to(out).as_posix())
    temporary.replace(path)
    print(f'Upload: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='Check frozen inputs without weights or neural execution')
    args = parser.parse_args()
    source, manifest, ref, claims = load_inputs()
    if args.preflight:
        print('Pipeline inputs passed: 300 development + 300 confirmation; ID/text disjoint.')
        print('Neural execution and model/index bytes not checked.')
        return
    out = ROOT / 'artifacts/fever_pipeline_v1'
    if (out / 'output_manifest.json').exists():
        verify_manifest(out)
        print('Completed full-pipeline output verified; no rerun.', flush=True)
        export(out)
        return
    packages = {n: version(n) for n in ref['packages']}
    if packages != ref['packages']:
        raise ValueError(f'Original model environment required: {ref["packages"]}; got {packages}')
    print('Verifying existing local model files; no downloads.', flush=True)
    for name, path in (('generator', 'qwen2.5-1.5b-instruct'), ('nli', 'nli-deberta-v3-small'),
                       ('embedding', 'all-MiniLM-L6-v2')):
        expected = ref['generator']['files'] if name == 'generator' else ref['feature_models'][name]
        if _model_files(ROOT / 'models' / path) != expected:
            raise ValueError(f'Original {name} weights/tokenizer required')
    db = ROOT / 'data/processed/fever/heldout_v1/wiki.sqlite'
    with closing(sqlite3.connect(f'file:{db.resolve().as_posix()}?mode=ro', uri=True)) as connection:
        index_meta = dict(connection.execute('SELECT key,value FROM metadata'))
    if (index_meta.get('archive_sha256') != EXPECTED_ARCHIVE_SHA256 or
            index_meta.get('source_rows') != str(EXPECTED_PAGES) or
            index_meta.get('retrieval') != 'fts5_bm25_title3_body1' or
            int(index_meta['pages'])+int(index_meta['empty_placeholders']) != EXPECTED_PAGES):
        raise ValueError('Pinned Wikipedia index required')
    identity = {'cohort_manifest_sha256': COHORT_SHA256, 'reference_sha256': REFERENCE_SHA256,
                'packages': packages, 'settings': SETTINGS, 'index_metadata': index_meta,
                'code_sha256': {n: sha256(ROOT / n) for n in CODE},
                'protocol_sha256': sha256(ROOT / 'docs/FEVER_PIPELINE_PROTOCOL.md')}
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / 'input_manifest.json'
    if receipt.exists() and load(receipt) != identity:
        raise ValueError('Pipeline run identity changed; cache reuse refused')
    if not receipt.exists() and any(out.iterdir()):
        raise ValueError('Pipeline cache exists without identity')
    write_json_atomic(receipt, identity)
    development = run_stage('development', claims['development'], source, out, db, ref, identity)
    path = source / 'development/gold.jsonl'
    if sha256(path) != manifest['cohorts']['development']['gold_sha256']:
        raise ValueError('Development gold identity differs')
    fit_path = out / 'correctness_calibrators.json'
    fit_binding = {'development_predictions_sha256': sha256(out / 'development/predictions.json'),
                   'development_gold_sha256': sha256(path), 'confirmation_labels_used': False}
    if fit_path.exists() and (out / 'calibrator_manifest.json').exists():
        verify_manifest(out, 'calibrator_manifest.json')
        frozen_fit = load(fit_path)
        if any(frozen_fit[k] != v for k, v in fit_binding.items()) or set(frozen_fit['policies']) != set(POLICIES):
            raise ValueError('Frozen correctness fit/input identity changed')
        fits = frozen_fit['policies']
    else:
        fits = calibrate_development(development, read_jsonl(path))
        frozen_fit = dict(fit_binding, policies=fits)
        if fit_path.exists() and load(fit_path) != frozen_fit:
            raise ValueError('Incomplete development fit differs from development-only replay')
        write_json_atomic(fit_path, frozen_fit)
        write_json_atomic(out / 'calibrator_manifest.json', {fit_path.name: sha256(fit_path)})
    print('Policy-specific final-answer correctness fits frozen. Starting confirmation.', flush=True)
    gc.collect()
    confirmation = run_stage('confirmation', claims['confirmation'], source, out, db, ref, identity)
    annotated = [dict(r, correctness_confidence=correctness_probability(r['score'], fits[r['policy']])
                     if r['candidate_label'] is not None else None) for r in confirmation]
    write_json_atomic(out / 'confirmation_predictions.json', annotated)
    write_json_atomic(out / 'confirmation_prediction_manifest.json', {
        'confirmation_predictions.json': sha256(out / 'confirmation_predictions.json'),
        fit_path.name: sha256(fit_path), 'confirmation/predictions.json': sha256(out / 'confirmation/predictions.json')})
    # First confirmation gold access follows the frozen generation/abstention/confidence outputs.
    path = source / 'confirmation/gold.jsonl'
    if sha256(path) != manifest['cohorts']['confirmation']['gold_sha256']:
        raise ValueError('Confirmation gold identity differs')
    gold = read_jsonl(path)
    summary = {'scope': 'fixed full English retrieved-evidence pipeline; new FEVER claims; development-only correctness fits',
               'development_claims': 300, 'confirmation_claims': 300, 'policy_records': len(confirmation),
               'policies': {policy: summarize_policy([r for r in confirmation if r['policy'] == policy], gold, fits[policy])
                            for policy in POLICIES},
               'no_confirmation_fitting': True, 'calibrators_sha256': sha256(fit_path),
               'distinct_responses': {name: len(load(out / name / 'responses_used.json'))
                                      for name in ('development', 'confirmation')},
               'limitations': ['Same public FEVER development dataset, not official blind test.',
                   'Wikipedia snapshot identity does not authenticate historical publishers.',
                   'Numeric input checks do not establish explanation truth or meaning.',
                   'English experiment; full multilingual retrieval/gate transfer remains untested.',
                   'Correctness confidence is evaluated only on final accepted answers.',
                   'Generation request counts are not measured runtime or energy savings.']}
    write_json_atomic(out / 'summary.json', summary)
    write_json_atomic(out / 'scoring_manifest.json', {'confirmation_gold_sha256': sha256(path),
        'prediction_manifest_sha256': sha256(out / 'confirmation_prediction_manifest.json'),
        'confirmation_fitting': False})
    write_json_atomic(out / 'output_manifest.json', {p.relative_to(out).as_posix(): sha256(p)
        for p in sorted(out.rglob('*')) if p.is_file() and p != out / 'output_manifest.json'})
    print('Full FEVER pipeline evaluation complete. Final-answer correctness metrics saved.', flush=True)
    export(out)


if __name__ == '__main__':
    main()
