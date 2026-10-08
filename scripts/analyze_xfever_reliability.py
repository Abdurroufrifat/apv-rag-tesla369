"""Descriptive reliability audit of frozen multilingual NLI outputs."""
import json
import math
from pathlib import Path

from apv_rag.splits import sha256, write_json_atomic
from run_multilingual_agreement_replay import LABELS, align, read, receipt, require


ROOT = Path(__file__).resolve().parents[1]


def reliability(rows, bins=10):
    if type(bins) is not int or bins < 1:
        raise ValueError('Positive bin count required')
    groups = {}
    seen = set()
    for row in rows:
        key = (row['file'], row['row'])
        if key in seen:
            raise ValueError('Duplicate prediction row')
        seen.add(key)
        probabilities = row['probabilities']
        if (type(probabilities) is not list or len(probabilities) != len(LABELS)
                or any(type(x) not in (int, float) or not math.isfinite(x)
                       or x < 0 or x > 1 for x in probabilities)
                or abs(sum(probabilities) - 1) > 1e-6):
            raise ValueError('Invalid probability vector')
        if row['true_label'] not in LABELS or row['predicted_label'] not in LABELS:
            raise ValueError('Unknown verdict label')
        predicted = LABELS[max(range(3), key=lambda i: probabilities[i])]
        if row['predicted_label'] != predicted:
            raise ValueError('Prediction differs from probability order')
        confidence = max(probabilities)
        correct = int(predicted == row['true_label'])
        brier = sum((p - int(i == LABELS.index(row['true_label']))) ** 2
                    for i, p in enumerate(probabilities))
        bucket = min(int(confidence * bins), bins - 1)
        groups.setdefault(row['file'], []).append((confidence, correct, brier, bucket))
    if not groups:
        raise ValueError('No predictions')
    results = {}
    for file, entries in sorted(groups.items()):
        n = len(entries)
        buckets = []
        ece = 0.0
        for i in range(bins):
            selected = [r for r in entries if r[3] == i]
            count = len(selected)
            mean_confidence = sum(r[0] for r in selected) / count if count else None
            accuracy = sum(r[1] for r in selected) / count if count else None
            if count:
                ece += count / n * abs(accuracy - mean_confidence)
            buckets.append({'lower': i / bins, 'upper': (i + 1) / bins,
                            'count': count, 'correct': sum(r[1] for r in selected),
                            'mean_confidence': mean_confidence, 'accuracy': accuracy})
        results[file] = {'rows': n, 'accuracy': sum(r[1] for r in entries) / n,
                         'mean_confidence': sum(r[0] for r in entries) / n,
                         'multiclass_brier': sum(r[2] for r in entries) / n,
                         'top_label_ece': ece, 'reliability_bins': buckets}
    return results


def main():
    source = ROOT / 'artifacts'
    generated = source / 'xfever_generation_received'
    nli = source / 'xfever_multilingual_received'
    qdigest, ndigest = receipt(generated), receipt(nli)
    previous = read(source / 'multilingual_agreement_replay_v1/audit_manifest.json')
    require(previous['source_output_manifests_sha256'] ==
            {'generation': qdigest, 'multilingual_nli': ndigest},
            'Agreement replay source identity changed')
    for name, digest in previous['files'].items():
        require(sha256(source / 'multilingual_agreement_replay_v1' / name) == digest,
                f'Agreement replay receipt mismatch: {name}')
    qrows, nrows = read(generated / 'predictions.json'), read(nli / 'predictions.json')
    aligned = align(qrows, nrows)
    require(len(aligned) == 6600, 'Unexpected aligned row count')
    results = reliability(nrows, bins=10)
    require(len(results) == 11 and all(v['rows'] == 600 for v in results.values()),
            'Unexpected reliability cohort size')
    output = source / 'xfever_reliability_audit_v1'
    output.mkdir(exist_ok=True)
    write_json_atomic(output / 'summary.json', {
        'scope': 'Fixed-bin descriptive reliability of frozen multilingual NLI on supplied XFEVER evidence; no fitted calibration or independent holdout.',
        'label_probability_order': LABELS, 'bins': 10,
        'source_output_manifests_sha256': {'generation': qdigest, 'multilingual_nli': ndigest},
        'agreement_replay_audit_sha256': sha256(source / 'multilingual_agreement_replay_v1/audit_manifest.json'),
        'results': results,
    })
    lines = ['# XFEVER multilingual NLI reliability audit', '',
             'Saved probabilities are evaluated against existing labels without fitting or changing the model. Ten fixed equal-width confidence bins include the right edge only in the last bin. ECE measures absolute top-label confidence gaps, and multiclass Brier is the sum of squared errors over Supported, Refuted and Not Enough Evidence. Each file has 600 rows. Eleven files include translations of overlapping English claims, so their results must not be pooled as independent samples.', '',
             '| File | Accuracy | Mean confidence | Top-label ECE | Multiclass Brier |',
             '|---|---:|---:|---:|---:|']
    for file, r in results.items():
        lines.append(f"| {file} | {r['accuracy']:.3f} | {r['mean_confidence']:.3f} | {r['top_label_ece']:.3f} | {r['multiclass_brier']:.3f} |")
    lines += ['', 'Bin counts, observed accuracy and mean confidence are in summary.json. Different languages and human/machine translations share underlying claims; differences are descriptive. The evidence is supplied, not retrieved by the multilingual pipeline. These scores measure the existing NLI model, not Qwen confidence, APV-RAG calibration or the correctness of its explanations. A calibrated deployed system and an untouched, provenance-independent evaluation are still open.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json_atomic(output / 'audit_manifest.json', {
        'source_output_manifests_sha256': {'generation': qdigest, 'multilingual_nli': ndigest},
        'agreement_replay_audit_sha256': sha256(source / 'multilingual_agreement_replay_v1/audit_manifest.json'),
        'script_sha256': sha256(Path(__file__)),
        'files': {p.name: sha256(p) for p in output.iterdir() if p.name != 'audit_manifest.json'},
    })
    print('XFEVER reliability audit passed: 6600 predictions, 11 supplied-evidence files.')


if __name__ == '__main__':
    main()
