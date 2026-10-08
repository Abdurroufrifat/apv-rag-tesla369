"""Correctness confidence for final accepted outputs of the fixed FEVER pipeline."""

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from apv_rag.direct_nli import TARGET_LABELS, direct_probabilities
from apv_rag.fever_nli import LABEL_MAP


POLICIES = ('no_gate', 'nli', 'embedding', 'combined')


def logit_scores(scores):
    p = np.asarray(scores, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Finite scores in [0,1] required')
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def fit_correctness(scores, correct):
    x = logit_scores(scores)
    y = np.asarray(correct)
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned binary correctness outcomes required')
    y = y.astype(int)
    base = {'accepted_development_answers': len(x), 'correct_development_answers': int(y.sum()),
            'probability_clip': 1e-6, 'threshold_fitted': False}
    if not len(x):
        return dict(base, method='unavailable_no_accepted_development_answers')
    if len(set(y)) == 1:
        return dict(base, method='constant_beta_1_1_prevalence', probability=float((y.sum()+1)/(len(y)+2)))
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=369).fit(x, y)
    return dict(base, method='logistic_correctness_from_generated_label_nli_score', C=1.0,
                coefficient=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]))


def correctness_probability(score, calibrator):
    x = float(logit_scores([score])[0, 0])
    method = calibrator['method']
    if method == 'unavailable_no_accepted_development_answers':
        return None
    if method == 'constant_beta_1_1_prevalence':
        return calibrator['probability']
    if method != 'logistic_correctness_from_generated_label_nli_score':
        raise ValueError('Unknown correctness calibrator')
    z = x*calibrator['coefficient']+calibrator['intercept']
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def answer_score(row, feature):
    if row['candidate_label'] is None:
        return None
    if row['candidate_label'] not in TARGET_LABELS or feature is None:
        raise ValueError('Accepted answer lacks bound NLI features')
    p = direct_probabilities(feature['scores_cen'])
    return float(p[TARGET_LABELS.index(row['candidate_label'])])


def reliability(probabilities, correct):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(correct, dtype=float)
    if p.shape != y.shape or not len(p):
        raise ValueError('Aligned nonempty reliability inputs required')
    logit_scores(p)
    ece = 0.0
    for i in range(15):
        mask = (p >= i/15) & (p <= 1 if i == 14 else p < (i+1)/15)
        if mask.any():
            ece += float(mask.mean())*abs(float(p[mask].mean()-y[mask].mean()))
    clipped = np.clip(p, 1e-6, 1-1e-6)
    return {'answers': len(p), 'mean_confidence': float(p.mean()), 'accuracy': float(y.mean()),
            'ece_15_bins': ece, 'binary_brier': float(((p-y)**2).mean()),
            'binary_nll': float(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean())}


def summarize_policy(rows, gold, calibrator):
    if (not rows or len(rows) != len(gold) or len({r['claim_id'] for r in rows}) != len(rows) or
            [r['claim_id'] for r in rows] != [r['id'] for r in gold]):
        raise ValueError('Pipeline prediction/gold alignment mismatch')
    truth = [LABEL_MAP[r['label']] for r in gold]
    labels = [r['candidate_label'] or 'abstain' for r in rows]
    accepted = [i for i, r in enumerate(rows) if r['candidate_label'] is not None]
    correct = [labels[i] == truth[i] for i in accepted]
    result = {'claims': len(rows), 'accepted_answers': len(accepted), 'coverage': len(accepted)/len(rows),
              'accuracy_all_claims_abstentions_as_errors': sum(correct)/len(rows),
              'macro_f1_all_claims_abstentions_as_errors': float(f1_score(truth, labels,
                  labels=list(TARGET_LABELS), average='macro', zero_division=0)),
              'covered_accuracy': sum(correct)/len(correct) if correct else None}
    if not accepted:
        result['correctness_calibration'] = {'status': 'no_accepted_confirmation_answers', 'answers': 0}
        return result
    scores = [rows[i]['score'] for i in accepted]
    raw = reliability(scores, correct)
    confidence = [correctness_probability(s, calibrator) for s in scores]
    calibration = {'answers': len(accepted), 'raw_score_brier': raw['binary_brier'], 'raw_score_metrics': raw}
    if any(p is None for p in confidence):
        calibration['status'] = 'unavailable_no_accepted_development_answers'
    else:
        calibration['status'] = 'evaluated'
        calibration['calibrated_metrics'] = reliability(confidence, correct)
        order = np.argsort(-np.asarray(confidence), kind='stable')
        calibration['risk_coverage_among_accepted'] = {}
        for fraction in (0.5, 0.8, 1.0):
            n = math.ceil(len(accepted)*fraction)
            accuracy = float(np.asarray(correct)[order[:n]].mean())
            calibration['risk_coverage_among_accepted'][str(fraction)] = {
                'selected': n, 'coverage_all_claims': n/len(rows), 'accuracy': accuracy, 'risk': 1-accuracy}
    result['correctness_calibration'] = calibration
    return result
