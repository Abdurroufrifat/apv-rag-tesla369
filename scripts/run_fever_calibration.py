"""Retrieve, infer, fit development temperature, freeze, then score fresh confirmation."""

import json
import sqlite3
import sys
import zipfile
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.direct_nli import TARGET_LABELS
from apv_rag.fever_calibration import (compare_confirmation, fit_temperature, normalized,
                                      transform_predictions)
from apv_rag.fever_corpus import EXPECTED_ARCHIVE_SHA256, EXPECTED_PAGES, retrieve_claims
from apv_rag.fever_nli import (LABEL_MAP, predict_context, read_jsonl, validate_contexts,
                              verify_prediction_files)
from apv_rag.nli_comparison import _fingerprint, _model_files, _pairs
from apv_rag.splits import sha256, write_json_atomic

COHORT_MANIFEST_SHA256 = 'bdfa158210b808343fde0d9bfaced2f51c1e59dd9a57d35da442d44bb6df11aa'
REFERENCE_SHA256 = 'ada02e3553c1be1a772ac46dc7f1397e5523fda80a792627dadaf3591b6680c9'
CODE_FILES = ('scripts/run_fever_calibration.py', 'src/apv_rag/fever_calibration.py',
              'src/apv_rag/fever_corpus.py', 'src/apv_rag/fever_nli.py',
              'src/apv_rag/direct_nli.py', 'src/apv_rag/nli_comparison.py',
              'src/apv_rag/multilingual_nli.py', 'src/apv_rag/sentence_context.py',
              'src/apv_rag/metrics.py')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_manifest(folder, filename='output_manifest.json'):
    hashes = read_json(folder / filename)
    for name, digest in hashes.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha256(folder / rel) != digest:
            raise ValueError(f'Frozen output checksum mismatch: {name}')


def export_outputs(output):
    destination = ROOT / 'fever_calibration_outputs.zip'
    temporary = destination.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, 'fever_calibration_v1/' + p.relative_to(output).as_posix())
    temporary.replace(destination)
    print(f'Upload this file: {destination}', flush=True)


def infer_cohort(name, claims, source, out, db, identity, model_path):
    folder = out / f'{name}_nli'
    folder.mkdir(parents=True, exist_ok=True)
    stage_identity = dict(identity, cohort=name,
                          model_inputs_sha256=sha256(source / name / 'model_inputs.jsonl'))
    receipt = folder / 'input_manifest.json'
    if receipt.exists() and read_json(receipt) != stage_identity:
        raise ValueError(f'{name} run identity changed; refusing cache reuse')
    write_json_atomic(receipt, stage_identity)
    contexts = folder / 'contexts.jsonl'
    context_receipt = folder / 'retrieval_manifest.json'
    if contexts.exists():
        if not context_receipt.exists():
            # Recovery from interruption between atomic context completion and receipt write.
            validate_contexts(claims, read_jsonl(contexts))
            write_json_atomic(context_receipt, {'contexts.jsonl': sha256(contexts)})
        verify_manifest(folder, 'retrieval_manifest.json')
    else:
        partial = contexts.with_name(contexts.name + '.partial')
        if partial.exists():
            raise ValueError(f'Interrupted retrieval file exists: {partial}; ensure no other run is active before removing it')
        retrieve_claims(db, source / name / 'model_inputs.jsonl', contexts)
        write_json_atomic(context_receipt, {'contexts.jsonl': sha256(contexts)})
    rows = validate_contexts(claims, read_jsonl(contexts))
    if (folder / 'output_manifest.json').exists():
        meta, predictions = verify_prediction_files(folder)
        if meta != stage_identity or len(predictions) != len(claims):
            raise ValueError('Frozen cohort prediction identity mismatch')
        for row, pred in zip(rows, predictions, strict=True):
            if pred != predict_context(row, pred['nli_probabilities_cen']):
                raise ValueError('Frozen cohort prediction/context mismatch')
        return predictions
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32).to('cpu').eval()
    if {int(k): str(v).lower() for k, v in model.config.id2label.items()} != {
            0: 'contradiction', 1: 'entailment', 2: 'neutral'}:
        raise ValueError('Unexpected NLI mapping')
    predictions = []
    model_hash = _fingerprint(stage_identity)
    with closing(sqlite3.connect(folder / 'pair_cache.sqlite')) as con:
        con.execute('CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, scores TEXT)')
        for i, row in enumerate(rows, 1):
            premises = [d['text'] for d in row['evidence'] if d['text'].strip()]
            probabilities = _pairs(con, model, tokenizer, torch, premises, row['claim'], model_hash, 8)
            predictions.append(predict_context(row, probabilities))
            if i == 1 or i % 25 == 0:
                print(f'{name} NLI: {i}/{len(rows)}', flush=True)
    write_json_atomic(folder / 'predictions.json', predictions)
    write_json_atomic(folder / 'output_manifest.json', {
        n: sha256(folder / n) for n in ('input_manifest.json', 'predictions.json')})
    print(f'{name}: raw predictions frozen; no {name} gold read by inference.', flush=True)
    return predictions


def main():
    source = ROOT / 'data/external/fever/calibration_v1'
    if sha256(source / 'selection_manifest.json') != COHORT_MANIFEST_SHA256:
        raise ValueError('Frozen calibration cohort manifest differs')
    manifest = read_json(source / 'selection_manifest.json')
    claims = {}
    for name, count in (('development', 600), ('confirmation', 300)):
        path = source / name / 'model_inputs.jsonl'
        cohort = manifest['cohorts'][name]
        if sha256(path) != cohort['model_inputs_sha256']:
            raise ValueError(f'{name} model-only input checksum mismatch')
        claims[name] = read_jsonl(path)
        if len(claims[name]) != count or [r['id'] for r in claims[name]] != cohort['selected_ids']:
            raise ValueError('Cohort claim alignment mismatch')
    all_claims = claims['development'] + claims['confirmation']
    if (len({r['id'] for r in all_claims}) != 900 or
            len({normalized(r['claim']) for r in all_claims}) != 900 or
            {r['id'] for r in all_claims} & set(manifest['excluded_prior_ids']) or
            {normalized(r['claim']) for r in all_claims} & set(manifest['excluded_prior_texts'])):
        raise ValueError('Calibration/confirmation claims overlap previously observed claims')
    out = ROOT / 'artifacts/fever_calibration_v1'
    if (out / 'output_manifest.json').exists():
        verify_manifest(out)
        print('Completed calibration/confirmation results verified; no rerun.', flush=True)
        export_outputs(out)
        return
    print('Checking existing Wikipedia index and model identities...', flush=True)
    db = ROOT / 'data/processed/fever/heldout_v1/wiki.sqlite'
    with closing(sqlite3.connect(f'file:{db.resolve().as_posix()}?mode=ro', uri=True)) as con:
        index_meta = dict(con.execute('SELECT key,value FROM metadata'))
    if (index_meta.get('archive_sha256') != EXPECTED_ARCHIVE_SHA256 or
            index_meta.get('source_rows') != str(EXPECTED_PAGES) or
            index_meta.get('retrieval') != 'fts5_bm25_title3_body1' or
            int(index_meta['pages']) + int(index_meta['empty_placeholders']) != EXPECTED_PAGES):
        raise ValueError('Pinned Wikipedia index metadata required')
    reference = ROOT / 'config/fever_nli_reference_v1.json'
    if sha256(reference) != REFERENCE_SHA256:
        raise ValueError('Original model reference changed')
    ref = read_json(reference)
    model_path = ROOT / 'models/nli-deberta-v3-small'
    if _model_files(model_path) != ref['model']:
        raise ValueError('Original local English NLI model required')
    packages = {name: version(name) for name in ref['packages']}
    if packages != ref['packages']:
        raise ValueError(f'Original inference package versions required: {ref["packages"]}; got {packages}')
    identity = {'cohort_manifest_sha256': COHORT_MANIFEST_SHA256,
                'reference_sha256': REFERENCE_SHA256, 'model': ref['model'],
                'packages': dict(packages, scipy=version('scipy')), 'index_metadata': index_meta,
                'seed': 369, 'threads': 4, 'batch_size': 8, 'max_pair_tokens': 256,
                'dtype': 'float32', 'device': 'cpu',
                'code_sha256': {n: sha256(ROOT / n) for n in CODE_FILES},
                'protocol_sha256': sha256(ROOT / 'docs/FEVER_CALIBRATION_PROTOCOL.md')}
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / 'input_manifest.json'
    if receipt.exists() and read_json(receipt) != identity:
        raise ValueError('Inputs or code changed; refusing previous inference/calibration cache')
    write_json_atomic(receipt, identity)
    development = infer_cohort('development', claims['development'], source, out, db, identity, model_path)
    gold_path = source / 'development/gold.jsonl'
    if sha256(gold_path) != manifest['cohorts']['development']['gold_sha256']:
        raise ValueError('Development gold checksum mismatch')
    gold_dev = read_jsonl(gold_path)
    if [r['id'] for r in gold_dev] != [r['id'] for r in development]:
        raise ValueError('Development label alignment mismatch')
    calibrator_path = out / 'calibrator.json'
    fit_identity = {'development_predictions_sha256': sha256(out / 'development_nli/predictions.json'),
                    'development_gold_sha256': sha256(gold_path),
                    'cohort_manifest_sha256': COHORT_MANIFEST_SHA256}
    if calibrator_path.exists():
        if not (out / 'calibrator_manifest.json').exists():
            # Recover an interruption after scalar save without changing the fitted value.
            expected = dict(fit_temperature([r['probabilities'] for r in development],
                [TARGET_LABELS.index(LABEL_MAP[r['label']]) for r in gold_dev]), **fit_identity)
            if read_json(calibrator_path) != expected:
                raise ValueError('Incomplete calibrator does not match development-only fit')
            write_json_atomic(out / 'calibrator_manifest.json', {'calibrator.json': sha256(calibrator_path)})
        verify_manifest(out, 'calibrator_manifest.json')
        calibrator = read_json(calibrator_path)
        if any(calibrator[k] != v for k, v in fit_identity.items()):
            raise ValueError('Frozen calibrator development identity mismatch')
    else:
        calibrator = dict(fit_temperature([r['probabilities'] for r in development],
            [TARGET_LABELS.index(LABEL_MAP[r['label']]) for r in gold_dev]), **fit_identity)
        write_json_atomic(calibrator_path, calibrator)
        write_json_atomic(out / 'calibrator_manifest.json', {'calibrator.json': sha256(calibrator_path)})
    print(f'Temperature frozen: {calibrator["temperature"]:.6f}; starting fresh confirmation.', flush=True)
    confirmation = infer_cohort('confirmation', claims['confirmation'], source, out, db, identity, model_path)
    transformed = transform_predictions(confirmation, calibrator['temperature'])
    write_json_atomic(out / 'confirmation_calibrated_predictions.json', transformed)
    write_json_atomic(out / 'confirmation_prediction_manifest.json', {
        'confirmation_calibrated_predictions.json': sha256(out / 'confirmation_calibrated_predictions.json'),
        'calibrator.json': sha256(calibrator_path),
        'confirmation_nli/predictions.json': sha256(out / 'confirmation_nli/predictions.json')})
    verify_manifest(out, 'confirmation_prediction_manifest.json')
    # Confirmation gold is first opened only after the fitted scalar and all predictions are frozen.
    confirmation_gold = source / 'confirmation/gold.jsonl'
    if sha256(confirmation_gold) != manifest['cohorts']['confirmation']['gold_sha256']:
        raise ValueError('Confirmation gold checksum mismatch')
    report, rebuilt = compare_confirmation(confirmation, read_jsonl(confirmation_gold), calibrator)
    if transformed != rebuilt:
        raise ValueError('Frozen calibrated probability replay mismatch')
    report['development_claims'] = 600
    report['calibrator_sha256'] = sha256(calibrator_path)
    write_json_atomic(out / 'summary.json', report)
    write_json_atomic(out / 'scoring_manifest.json', {
        'confirmation_gold_sha256': sha256(confirmation_gold),
        'frozen_prediction_manifest_sha256': sha256(out / 'confirmation_prediction_manifest.json'),
        'confirmation_fitting': False})
    write_json_atomic(out / 'output_manifest.json', {
        p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p != out / 'output_manifest.json'})
    print(f'Calibration confirmation complete: accuracy {report["raw_accuracy"]:.4f}; '
          f'ECE {report["raw_metrics"]["expected_calibration_error"]:.4f} -> '
          f'{report["calibrated_metrics"]["expected_calibration_error"]:.4f}', flush=True)
    export_outputs(out)


if __name__ == '__main__':
    main()
