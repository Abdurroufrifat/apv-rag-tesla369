"""Separate FEVER development calibration and fresh confirmation."""

import hashlib
import math

import numpy as np
from scipy.optimize import minimize_scalar

SEED = 'apv-rag-fever-calibration-confirmation-v1'
PROBABILITY_FLOOR = 1e-12


def normalized(text):
    return ' '.join(text.casefold().split())


def partition_claims(rows, *, exclude_ids, exclude_texts, development_count=600,
                     confirmation_count=300):
    if any(type(n) is not int or n < 1 for n in (development_count, confirmation_count)):
        raise ValueError('Positive integer cohort counts required')
    ids = set()
    for row in rows:
        if (type(row.get('id')) is not int or not isinstance(row.get('claim'), str) or
                not row['claim'].strip() or row['id'] in ids):
            raise ValueError('Invalid or duplicate claim identity')
        ids.add(row['id'])
    excluded_texts = {normalized(t) for t in exclude_texts}
    chosen, texts = [], set()
    count = development_count + confirmation_count
    for row in sorted(rows, key=lambda r: (
            hashlib.sha256(f"{SEED}:{r['id']}".encode('ascii')).hexdigest(), r['id'])):
        text = normalized(row['claim'])
        if row['id'] in exclude_ids or text in excluded_texts or text in texts:
            continue
        chosen.append(row)
        texts.add(text)
        if len(chosen) == count:
            return chosen[:development_count], chosen[development_count:]
    raise ValueError('Insufficient distinct unseen claims')


def probability_matrix(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) < 1 or not np.isfinite(p).all() or
            (p < 0).any() or (p > 1).any() or
            not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Normalized three-label probabilities required')
    return p


def apply_temperature(probabilities, temperature):
    p = probability_matrix(probabilities)
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    z = np.log(np.maximum(p, PROBABILITY_FLOOR)) / temperature
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def negative_log_likelihood(probabilities, labels):
    p = probability_matrix(probabilities)
    y = np.asarray(labels)
    if y.shape != (len(p),) or y.dtype.kind not in 'iu' or (y < 0).any() or (y > 2).any():
        raise ValueError('Aligned integer class indices required')
    return float(-np.log(np.maximum(p[np.arange(len(p)), y], PROBABILITY_FLOOR)).mean())


def fit_temperature(probabilities, labels):
    p = probability_matrix(probabilities)
    before = negative_log_likelihood(p, labels)
    result = minimize_scalar(lambda t: negative_log_likelihood(apply_temperature(p, float(t)), labels),
                             bounds=(0.05, 20.0), method='bounded', options={'xatol': 1e-8})
    if not result.success or not math.isfinite(result.fun):
        raise ValueError('Temperature optimization failed')
    # Include the identity transform so calibration cannot worsen development NLL.
    temperature = float(result.x) if result.fun < before else 1.0
    return {'method': 'temperature scaling of pooled probabilities; one positive scalar',
            'temperature': temperature, 'bounds': [0.05, 20.0],
            'probability_floor': PROBABILITY_FLOOR, 'development_claims': len(p),
            'development_nll_before': before,
            'development_nll_after': negative_log_likelihood(apply_temperature(p, temperature), labels),
            'argmax_preserved': True, 'no_threshold_fitted': True}


def transform_predictions(predictions, temperature):
    from apv_rag.direct_nli import TARGET_LABELS

    q = apply_temperature([r['probabilities'] for r in predictions], temperature)
    return [{'id': row['id'], 'predicted_label': TARGET_LABELS[int(p.argmax())],
             'probabilities': p.tolist()} for row, p in zip(predictions, q, strict=True)]


def compare_confirmation(predictions, gold, calibrator):
    from apv_rag.direct_nli import TARGET_LABELS
    from apv_rag.fever_nli import LABEL_MAP, score_predictions
    from apv_rag.metrics import classification_metrics

    raw = score_predictions(predictions, gold)
    transformed = transform_predictions(predictions, calibrator['temperature'])
    truth = [LABEL_MAP[r['label']] for r in gold]
    labels = [TARGET_LABELS.index(t) for t in truth]
    q = np.asarray([r['probabilities'] for r in transformed])
    pred = [r['predicted_label'] for r in transformed]
    changed = sum(a['predicted_label'] != b['predicted_label']
                  for a, b in zip(predictions, transformed, strict=True))
    if changed:
        raise ValueError('Temperature scaling unexpectedly changed argmax')
    return {'scope': 'fresh FEVER claim confirmation of one development-fitted temperature',
            'confirmation_claims': len(predictions), 'temperature': calibrator['temperature'],
            'raw_accuracy': raw['accuracy'], 'calibrated_accuracy': raw['accuracy'],
            'argmax_changed': changed, 'raw_metrics': raw['metrics'],
            'calibrated_metrics': classification_metrics(truth, pred, q, TARGET_LABELS,
                ece_bins=15, coverages=[0.5, 0.8, 1.0]),
            'raw_nll': negative_log_likelihood([r['probabilities'] for r in predictions], labels),
            'calibrated_nll': negative_log_likelihood(q, labels),
            'raw_mean_confidence': float(np.asarray([r['probabilities'] for r in predictions]).max(1).mean()),
            'calibrated_mean_confidence': float(q.max(1).mean()), 'retrieval': raw['retrieval'],
            'no_confirmation_fitting': True,
            'limitations': ['Same public FEVER development dataset, not official blind test.',
                           'Claim IDs and normalized text are disjoint; pages and near duplicates may overlap.',
                           'Temperature scaling preserves verdicts and does not repair accuracy.',
                           'This calibrates the NLI baseline, not the integrated APV-RAG generator or gate.']}, transformed
