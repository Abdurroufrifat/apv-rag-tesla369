"""Verify received FEVER outputs and describe errors without fitting or inference."""

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.direct_nli import TARGET_LABELS
from apv_rag.fever_corpus import EXPECTED_ARCHIVE_SHA256, EXPECTED_PAGES
from apv_rag.fever_nli import (GOLD_SHA256, LABEL_MAP, load_claims, predict_context, read_jsonl,
                              score_predictions, validate_contexts, verify_prediction_files)
from apv_rag.nli_comparison import _fingerprint
from apv_rag.splits import sha256, write_json_atomic


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(inputs, output):
    inputs, output = Path(inputs), Path(output)
    if output.exists():
        raise FileExistsError('Audit output exists; refusing overwrite')
    run, scoring, retrieval = [inputs / n for n in (
        'fever_nli_v1', 'fever_nli_scoring_v1', 'fever_retrieval_v1')]
    identity, predictions = verify_prediction_files(run)
    hashes = read_json(scoring / 'output_manifest.json')
    if set(hashes) != {'input_manifest.json', 'summary.json'}:
        raise ValueError('Incomplete scoring manifest')
    for name, digest in hashes.items():
        if sha256(scoring / name) != digest:
            raise ValueError(f'Scoring checksum mismatch: {name}')
    scoring_meta = read_json(scoring / 'input_manifest.json')
    retrieval_meta = read_json(retrieval / 'input_manifest.json')
    ref_path = ROOT / 'config/fever_nli_reference_v1.json'
    ref = read_json(ref_path)
    if (sha256(ref_path) != identity['reference_sha256'] or
            identity['models'] != ref['model'] or identity['packages'] != ref['packages'] or
            scoring_meta['prediction_manifest_sha256'] != sha256(run / 'output_manifest.json') or
            scoring_meta['predictions_sha256'] != sha256(run / 'predictions.json') or
            scoring_meta['no_target_fitting'] is not True or
            scoring_meta['gold_sha256'] != GOLD_SHA256):
        raise ValueError('Prediction, model-reference or scoring identity mismatch')
    source = ROOT / 'data/external/fever/heldout_v1'
    claims = load_claims(source)
    contexts_path = retrieval / 'contexts.jsonl'
    if (identity['claims'] != 300 or len(predictions) != 300 or
            sha256(contexts_path) != identity['contexts_sha256'] or
            sha256(contexts_path) != retrieval_meta['contexts_sha256'] or
            sha256(retrieval / 'input_manifest.json') != identity['retrieval_receipt_sha256'] or
            any(m['model_inputs_sha256'] != sha256(source / 'model_inputs.jsonl') or
                m['selection_manifest_sha256'] != sha256(source / 'selection_manifest.json')
                for m in (identity, retrieval_meta)) or
            retrieval_meta['archive_sha256'] != EXPECTED_ARCHIVE_SHA256 or
            retrieval_meta['source_records'] != EXPECTED_PAGES or
            retrieval_meta['page_count'] + retrieval_meta['empty_placeholders'] != EXPECTED_PAGES):
        raise ValueError('Frozen claim/context or archive receipt mismatch')
    code_checks = 0
    for meta in (identity, scoring_meta, retrieval_meta):
        for name, digest in meta['code_sha256'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or sha256(ROOT / path) != digest:
                raise ValueError(f'Producer code checksum mismatch: {name}')
            code_checks += 1
    if sha256(ROOT / 'docs/FEVER_EXTERNAL_EVALUATION.md') != identity['protocol_sha256']:
        raise ValueError('Frozen FEVER protocol checksum mismatch')
    contexts = validate_contexts(claims, read_jsonl(contexts_path))
    # Readonly pair-cache audit checks saved probabilities, not fresh neural logits.
    pair_count = 0
    with closing(sqlite3.connect(f'file:{(run / "pair_cache.sqlite").resolve().as_posix()}?mode=ro',
                                uri=True)) as con:
        for row, pred in zip(contexts, predictions, strict=True):
            if pred != predict_context(row, pred['nli_probabilities_cen']):
                raise ValueError('Saved prediction/context mismatch')
            premises = [d['text'] for d in row['evidence'] if d['text'].strip()]
            for premise, probabilities in zip(premises, pred['nli_probabilities_cen'], strict=True):
                key = _fingerprint([_fingerprint(identity), premise, row['claim'], 256])
                cached = con.execute('SELECT scores FROM pairs WHERE key=?', (key,)).fetchone()
                if cached is None or json.loads(cached[0]) != probabilities:
                    raise ValueError('Saved probability/cache mismatch')
                pair_count += 1
    gold_path = source / 'gold.jsonl'
    if sha256(gold_path) != GOLD_SHA256:
        raise ValueError('Frozen gold checksum mismatch')
    gold = read_jsonl(gold_path)
    replay = score_predictions(predictions, gold)
    if replay != read_json(scoring / 'summary.json'):
        raise ValueError('Full metric replay differs from received summary')
    confusion = np.zeros((3, 3), dtype=int)
    diagnostic_rows, subsets = [], {'complete_group_found': [], 'complete_group_missing': []}
    for pred, target in zip(predictions, gold, strict=True):
        truth = LABEL_MAP[target['label']]
        correct = pred['predicted_label'] == truth
        confusion[TARGET_LABELS.index(truth), TARGET_LABELS.index(pred['predicted_label'])] += 1
        found = None
        if target['label'] != 'NOT ENOUGH INFO':
            sentences = {(d['id'], i) for d in pred['evidence']
                         for i in d['selected_sentence_indices']}
            groups = [{(x[2], x[3]) for x in g if x[2] is not None and x[3] is not None}
                      for g in target['evidence']]
            found = any(g and g <= sentences for g in groups)
            subsets['complete_group_found' if found else 'complete_group_missing'].append(correct)
        diagnostic_rows.append({'id': pred['id'], 'true_label': truth,
                                'predicted_label': pred['predicted_label'], 'correct': correct,
                                'confidence': max(pred['probabilities']),
                                'complete_gold_sentence_group_found': found})
    confidence = np.array([r['confidence'] for r in diagnostic_rows])
    correctness = np.array([r['correct'] for r in diagnostic_rows])
    summary = {'verification': 'passed', 'claims': 300, 'saved_pairs_checked': pair_count,
               'producer_code_hash_checks': code_checks, 'metrics_recomputed': True,
               'model_identity': 'matches recorded reference; weights not present for fresh inference',
               'retrieval_identity': 'saved receipt and contexts verified; full index not replayed',
               'labels': list(TARGET_LABELS), 'confusion_true_rows_predicted_columns': confusion.tolist(),
               'mean_confidence': float(confidence.mean()),
               'accuracy': replay['accuracy'],
               'confidence_minus_accuracy': float(confidence.mean()) - replay['accuracy'],
               'retrieval_conditioned_accuracy': {
                   k: {'claims': len(v), 'correct': sum(v),
                       'accuracy': sum(v) / len(v) if v else None} for k, v in subsets.items()},
               'nei_false_decisive_predictions': int(confusion[2, :2].sum()),
               'no_new_inference_or_fitting': True,
               'cohort_status': 'target outcomes now observed; future changes on this cohort are exploratory',
               'limitations': ['This is a diagnostic baseline, not the full APV-RAG system.',
                               'Full evidence-group presence is not a causal estimate of retrieval impact.',
                               'No validation of historical attribution or generated explanation truth.']}
    output.mkdir(parents=True)
    write_json_atomic(output / 'audit_summary.json', summary)
    write_json_atomic(output / 'claim_diagnostics.json', diagnostic_rows)
    write_json_atomic(output / 'input_manifest.json', {
        'inputs': {str(p.relative_to(inputs)).replace('\\', '/'): sha256(p)
                   for p in sorted(inputs.rglob('*')) if p.is_file()},
        'gold_sha256': GOLD_SHA256, 'script_sha256': sha256(Path(__file__)),
        'packages': {'numpy': np.__version__}, 'new_inference': False, 'new_fitting': False})
    text = ('# Verified FEVER baseline\n\n'
            'All 300 saved predictions bind to the frozen claims and contexts. Producer code '
            f'hashes match ({code_checks} checks), all {pair_count} saved premise probabilities '
            'match their cache entries, and every reported metric is reproduced from gold. '
            'The complete archive/index and model inference were not independently rerun.\n\n'
            f'Accuracy is {replay["accuracy"]:.4%} (142/300), macro F1 '
            f'{replay["metrics"]["macro_f1"]:.6f}, Brier '
            f'{replay["metrics"]["multiclass_brier"]:.6f}, and ECE '
            f'{replay["metrics"]["expected_calibration_error"]:.6f}. Mean confidence is '
            f'{confidence.mean():.4%}. Complete annotated sentence groups are retrieved for '
            '106/196 verifiable claims (54.08%).\n\n'
            '| True label | Predicted Supported | Predicted Refuted | Predicted NEI |\n'
            '|---|---:|---:|---:|\n' + ''.join(
                f'| {label} | {row[0]} | {row[1]} | {row[2]} |\n'
                for label, row in zip(TARGET_LABELS, confusion, strict=True)) +
            '\nAccuracy is 56/106 (52.83%) when a complete gold sentence group is present '
            'and 47/90 (52.22%) when it is absent, considering only verifiable claims. '
            'This descriptive comparison does not isolate retrieval causally. NLI pooling, '
            'sentence context and possible token truncation remain candidate causes. '
            'Of 104 NEI claims, 65 receive a decisive Supported or Refuted prediction.\n\n'
            'Do not tune a gate or calibrator on these 300 claims and then report the same '
            'cohort as independent confirmation. Retain these frozen results. Fit any new '
            'calibrator or policy on separate development data and freeze it before using '
            'a new untouched confirmation cohort. No manuscript or GitHub publication '
            'was produced by this audit.\n')
    (output / 'RESULTS.md').write_text(text, encoding='utf-8')
    write_json_atomic(output / 'output_manifest.json', {
        p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    print(f'FEVER audit passed: 300 predictions; {pair_count} cached pairs; full metric replay.')
    print(output / 'RESULTS.md')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'artifacts/fever_received_v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/fever_audit_v1')
    args = parser.parse_args()
    audit(args.input_dir, args.output_dir)
