"""Check saved experiment hashes and metrics without loading fitted models."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/trained_copy_robustness_v1'
LABELS = ('Supported', 'Refuted', 'Not Enough Evidence', 'Conflicting Evidence/Cherrypicking')

def main():
    receipt = json.loads((OUT / 'receipt.json').read_text())
    for section in ('inputs', 'code', 'outputs'):
        for name, expected in receipt[section].items():
            path = (OUT if section == 'outputs' else ROOT) / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'SHA-256 mismatch: {name}')
    rows = json.loads((OUT / 'predictions.json').read_text())
    indices = json.loads((ROOT / 'data/processed/averitec/phase2b_split_v0_1/validation_indices.json').read_text())
    source = json.loads((ROOT / 'data/external/averitec/official_7c62d1e/train.json').read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    if len(rows) != 6090:
        raise ValueError('Prediction coverage differs')
    for model in ('control', 'copy_augmented'):
        clean = {r['upstream_index']: r['probabilities'][0] for r in rows if r['model'] == model and r['copies'] == 0}
        for copies in (0, 1, 5, 10, 25):
            selected = [r for r in rows if r['model'] == model and r['copies'] == copies]
            if [r['upstream_index'] for r in selected] != indices:
                raise ValueError('Validation ordering/coverage differs')
            cm = [[0]*4 for _ in LABELS]
            shifts = []
            for row in selected:
                probs = row['probabilities']
                if len(probs) != 4 or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs) or abs(sum(probs)-1) > 1e-10:
                    raise ValueError('Invalid probabilities')
                if row['true_label'] != source[row['upstream_index']]['label']:
                    raise ValueError('Upstream label differs')
                cm[LABELS.index(row['true_label'])][max(range(4), key=lambda i: probs[i])] += 1
                shifts.append(probs[0]-clean[row['upstream_index']])
            f1 = []
            for i in range(4):
                denom = sum(cm[i]) + sum(r[i] for r in cm)
                f1.append(2*cm[i][i]/denom if denom else 0)
            actual = {'macro_f1': sum(f1)/4, 'accuracy': sum(cm[i][i] for i in range(4))/609,
                      'mean_absolute_support_probability_shift': sum(map(abs, shifts))/609,
                      'mean_signed_support_probability_shift': sum(shifts)/609}
            saved = summary[f'{model}:copies_{copies}']
            if any(abs(saved[k]-v)>1e-12 for k,v in actual.items()):
                raise ValueError('Metric differs')
    print('Trained-copy results verified: hashes, 6,090 predictions, upstream labels and all metrics.')
    print('Exploratory internal validation only; independent confirmation remains open.')

if __name__ == '__main__':
    main()
