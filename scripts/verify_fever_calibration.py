"""Verify saved development calibration and fresh FEVER confirmation without neural inference."""

import json
import sqlite3
import sys
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.direct_nli import TARGET_LABELS
from apv_rag.fever_calibration import (PROBABILITY_FLOOR, compare_confirmation,
                                      fit_temperature, normalized)
from apv_rag.fever_nli import (LABEL_MAP, predict_context, read_jsonl,
                              validate_contexts, verify_prediction_files)
from apv_rag.nli_comparison import _fingerprint
from apv_rag.splits import sha256, write_json_atomic
from run_fever_calibration import COHORT_MANIFEST_SHA256, verify_manifest


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    received = ROOT / 'artifacts/fever_calibration_received_v1/fever_calibration_v1'
    output = ROOT / 'artifacts/fever_calibration_audit_v1'
    if output.exists():
        raise FileExistsError('Calibration audit exists; refusing overwrite')
    verify_manifest(received)
    identity = read_json(received / 'input_manifest.json')
    source = ROOT / 'data/external/fever/calibration_v1'
    if (sha256(source / 'selection_manifest.json') != COHORT_MANIFEST_SHA256 or
            identity['cohort_manifest_sha256'] != COHORT_MANIFEST_SHA256):
        raise ValueError('Frozen cohort identity differs')
    manifest = read_json(source / 'selection_manifest.json')
    ref_path = ROOT / 'config/fever_nli_reference_v1.json'
    ref = read_json(ref_path)
    if (sha256(ref_path) != identity['reference_sha256'] or ref['model'] != identity['model'] or
            any(identity['packages'][k] != v for k, v in ref['packages'].items())):
        raise ValueError('Recorded model/package reference differs')
    for name, digest in identity['code_sha256'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or sha256(ROOT / p) != digest:
            raise ValueError(f'Producer code mismatch: {name}')
    if sha256(ROOT / 'docs/FEVER_CALIBRATION_PROTOCOL.md') != identity['protocol_sha256']:
        raise ValueError('Frozen protocol differs')
    all_claims, predictions, golds = [], {}, {}
    pair_count = 0
    for name, count in (('development', 600), ('confirmation', 300)):
        m = manifest['cohorts'][name]
        folder = received / f'{name}_nli'
        claims_path = source / name / 'model_inputs.jsonl'
        gold_path = source / name / 'gold.jsonl'
        if sha256(claims_path) != m['model_inputs_sha256'] or sha256(gold_path) != m['gold_sha256']:
            raise ValueError('Claim/gold source checksum differs')
        claims = read_jsonl(claims_path)
        if len(claims) != count or [r['id'] for r in claims] != m['selected_ids']:
            raise ValueError('Cohort ID alignment differs')
        all_claims.extend(claims)
        verify_manifest(folder, 'retrieval_manifest.json')
        stage_meta, pred = verify_prediction_files(folder)
        expected = dict(identity, cohort=name, model_inputs_sha256=m['model_inputs_sha256'])
        if stage_meta != expected or len(pred) != count:
            raise ValueError('Inference-stage identity differs')
        rows = validate_contexts(claims, read_jsonl(folder / 'contexts.jsonl'))
        with closing(sqlite3.connect(f'file:{(folder / "pair_cache.sqlite").resolve().as_posix()}?mode=ro',
                                    uri=True)) as con:
            for row, p in zip(rows, pred, strict=True):
                if p != predict_context(row, p['nli_probabilities_cen']):
                    raise ValueError('Raw probability/context alignment differs')
                premises = [d['text'] for d in row['evidence'] if d['text'].strip()]
                for premise, scores in zip(premises, p['nli_probabilities_cen'], strict=True):
                    key = _fingerprint([_fingerprint(stage_meta), premise, row['claim'], 256])
                    cached = con.execute('SELECT scores FROM pairs WHERE key=?', (key,)).fetchone()
                    if not cached or json.loads(cached[0]) != scores:
                        raise ValueError('Saved premise probability/cache mismatch')
                    pair_count += 1
        predictions[name], golds[name] = pred, read_jsonl(gold_path)
        if [r['id'] for r in golds[name]] != [r['id'] for r in pred]:
            raise ValueError('Gold/prediction alignment differs')
    if (len({r['id'] for r in all_claims}) != 900 or
            len({normalized(r['claim']) for r in all_claims}) != 900 or
            {r['id'] for r in all_claims} & set(manifest['excluded_prior_ids']) or
            {normalized(r['claim']) for r in all_claims} & set(manifest['excluded_prior_texts'])):
        raise ValueError('Cohort or prior-outcome overlap found')
    verify_manifest(received, 'calibrator_manifest.json')
    calibrator = read_json(received / 'calibrator.json')
    if (calibrator['development_predictions_sha256'] != sha256(received / 'development_nli/predictions.json') or
            calibrator['development_gold_sha256'] != manifest['cohorts']['development']['gold_sha256'] or
            calibrator['cohort_manifest_sha256'] != COHORT_MANIFEST_SHA256):
        raise ValueError('Calibrator/development input binding differs')
    y_dev = [TARGET_LABELS.index(LABEL_MAP[r['label']]) for r in golds['development']]
    refit = fit_temperature([p['probabilities'] for p in predictions['development']], y_dev)
    temperature_error = abs(refit['temperature'] - calibrator['temperature'])
    if temperature_error > 1e-6:
        raise ValueError('Development-only temperature replay differs')
    for k in ('development_nll_before', 'development_nll_after'):
        if not np.isclose(refit[k], calibrator[k], rtol=0, atol=1e-10):
            raise ValueError('Development fit objective replay differs')
    verify_manifest(received, 'confirmation_prediction_manifest.json')
    replay, transformed = compare_confirmation(predictions['confirmation'], golds['confirmation'], calibrator)
    replay['development_claims'] = 600
    replay['calibrator_sha256'] = sha256(received / 'calibrator.json')
    if replay != read_json(received / 'summary.json'):
        raise ValueError('Full confirmation metric replay differs')
    saved = read_json(received / 'confirmation_calibrated_predictions.json')
    if (len(saved) != 300 or [r['id'] for r in saved] != [r['id'] for r in transformed] or
            [r['predicted_label'] for r in saved] != [r['predicted_label'] for r in transformed]):
        raise ValueError('Calibrated ID/label replay differs')
    probability_error = float(np.max(np.abs(np.array([r['probabilities'] for r in saved]) -
                                             np.array([r['probabilities'] for r in transformed]))))
    if probability_error > 1e-12:
        raise ValueError('Calibrated probability replay differs')
    scoring = read_json(received / 'scoring_manifest.json')
    if (scoring['confirmation_gold_sha256'] != manifest['cohorts']['confirmation']['gold_sha256'] or
            scoring['frozen_prediction_manifest_sha256'] != sha256(received / 'confirmation_prediction_manifest.json') or
            scoring['confirmation_fitting'] is not False):
        raise ValueError('Confirmation scoring receipt differs')
    p = np.asarray([r['probabilities'] for r in predictions['confirmation']])
    q = np.asarray([r['probabilities'] for r in saved])
    y = np.asarray([TARGET_LABELS.index(LABEL_MAP[r['label']]) for r in golds['confirmation']])
    onehot = np.eye(3)[y]
    losses = {
        'nll_calibrated_minus_raw': -np.log(np.maximum(q[np.arange(300), y], PROBABILITY_FLOOR)) +
                                    np.log(np.maximum(p[np.arange(300), y], PROBABILITY_FLOOR)),
        'brier_calibrated_minus_raw': ((q-onehot)**2).sum(1) - ((p-onehot)**2).sum(1)}
    rng = np.random.default_rng(369)
    samples = rng.integers(0, 300, size=(2000, 300))
    intervals = {name: {'observed_mean': float(v.mean()),
                        'paired_claim_bootstrap_95_percentile_interval':
                        np.quantile(v[samples].mean(1), [0.025, 0.975]).tolist()}
                 for name, v in losses.items()}
    audit = {'verification': 'passed', 'development_claims': 600, 'confirmation_claims': 300,
             'cached_pairs_checked': pair_count, 'confirmation_metric_replay': 'exact',
             'calibrated_probability_max_abs_error': probability_error,
             'probability_tolerance': 1e-12, 'temperature_replay_abs_error': temperature_error,
             'temperature_tolerance': 1e-6, 'temperature_fit_replay_uses_development_only': True,
             'producer_scipy': identity['packages']['scipy'], 'audit_scipy': version('scipy'),
             'paired_loss_intervals': intervals, 'bootstrap_seed': 369, 'bootstrap_replicates': 2000,
             'limitations': ['Claim resampling assumes claim units; shared pages/near duplicates may understate uncertainty.',
                            'Same public FEVER development dataset; no new independent dataset.',
                            'Saved model identity/cache verified; neural inference and full index not rerun.',
                            'Confidence improved here; verdict accuracy and NEI errors are not repaired.',
                            'Integrated APV-RAG generation/gate calibration remains open.']}
    output.mkdir(parents=True)
    write_json_atomic(output / 'verification.json', audit)
    write_json_atomic(output / 'verified_summary.json', replay)
    write_json_atomic(output / 'input_manifest.json', {
        'received_output_manifest_sha256': sha256(received / 'output_manifest.json'),
        'cohort_manifest_sha256': COHORT_MANIFEST_SHA256,
        'script_sha256': sha256(Path(__file__)), 'numpy': np.__version__, 'scipy': version('scipy')})
    text = ('# Verified FEVER calibration confirmation\n\n'
            f'All 900 development/confirmation predictions and {pair_count} cached premise '
            'scores bind to frozen inputs. Claim IDs and normalized text are disjoint between '
            'cohorts and prior project FEVER/XFEVER outcomes. All confirmation metrics replay '
            'exactly. The development-only temperature fit replays within 1e-6 across the '
            'recorded SciPy versions; transformed probabilities differ by at most '
            f'{probability_error:.3g}, below 1e-12. Full neural inference/index retrieval '
            'were not independently rerun.\n\n'
            f'The fitted temperature is {calibrator["temperature"]:.6f}.\n\n'
            '| Metric on 300 confirmation claims | Raw | Calibrated |\n'
            '|---|---:|---:|\n'
            f'| Accuracy | {replay["raw_accuracy"]:.2%} | {replay["calibrated_accuracy"]:.2%} |\n'
            f'| Macro F1 | {replay["raw_metrics"]["macro_f1"]:.6f} | {replay["calibrated_metrics"]["macro_f1"]:.6f} |\n'
            f'| Mean confidence | {replay["raw_mean_confidence"]:.2%} | {replay["calibrated_mean_confidence"]:.2%} |\n'
            f'| ECE, 15 bins | {replay["raw_metrics"]["expected_calibration_error"]:.6f} | {replay["calibrated_metrics"]["expected_calibration_error"]:.6f} |\n'
            f'| Brier | {replay["raw_metrics"]["multiclass_brier"]:.6f} | {replay["calibrated_metrics"]["multiclass_brier"]:.6f} |\n'
            f'| NLL | {replay["raw_nll"]:.6f} | {replay["calibrated_nll"]:.6f} |\n\n'
            'No verdict changes: accuracy is 159/300. At 50% coverage, accuracy falls '
            'from 97/150 (64.67%) to 94/150 (62.67%). At 80%, it falls from 137/240 '
            '(57.08%) to 134/240 (55.83%). Better probability calibration does not '
            'establish better selective ranking. Gold sentence-group recall is 106/209 '
            '(50.72%) among verifiable claims.\n\n'
            'The verification JSON contains paired claim-bootstrap intervals for calibrated '
            'minus raw NLL and Brier. These are descriptive, conditional on the fixed '
            'development-fitted scalar, and do not account for shared-page/near-duplicate '
            'dependence or uncertainty from refitting on another development sample.\n\n'
            'This closes the confidence-calibration experiment for the English NLI baseline. '
            'It does not establish calibrated accuracy of the integrated APV-RAG generator '
            'or gate. Preserve the scalar and observed confirmation results. Any new '
            'settings chosen after these outcomes require a fresh confirmation cohort. '
            'No manuscript or GitHub push was performed.\n')
    (output / 'RESULTS.md').write_text(text, encoding='utf-8')
    write_json_atomic(output / 'output_manifest.json', {
        x.name: sha256(x) for x in output.iterdir() if x.is_file()})
    print(f'Calibration verification passed: 900 predictions; {pair_count} cached pairs; exact summary replay.')
    print(json.dumps(intervals, indent=2))
    print(output / 'RESULTS.md')


if __name__ == '__main__':
    main()
