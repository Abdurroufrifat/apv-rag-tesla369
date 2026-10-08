"""Describe correct/incorrect candidates withheld by the saved display guard.

Uses existing benchmark labels on already observed English contexts. No model
inference, policy selection, independent confirmation or truth annotation.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from apv_rag.generative_rag import LABELS
from audit_evidence_display import verify as verify_display

OUTPUT = ROOT / 'artifacts/guard_tradeoffs_v1'


def compare(pairs):
    if not pairs:
        raise ValueError('Empty comparison')
    for before, after, gold in pairs:
        if gold not in LABELS or before not in (*LABELS, None) or after not in (*LABELS, None):
            raise ValueError('Invalid benchmark/candidate label')
        if after is not None and after != before:
            raise ValueError('Display guard revived or relabeled a candidate')
    n = len(pairs)
    result = {'records': n,
              'before_accepted': sum(b is not None for b, a, g in pairs),
              'after_accepted': sum(a is not None for b, a, g in pairs),
              'before_correct': sum(b == g for b, a, g in pairs),
              'after_correct': sum(a == g for b, a, g in pairs),
              'withheld_correct': sum(b == g and a is None for b, a, g in pairs),
              'withheld_incorrect': sum(b is not None and b != g and a is None for b, a, g in pairs)}
    for stage in ('before', 'after'):
        accepted, correct = result[f'{stage}_accepted'], result[f'{stage}_correct']
        result[f'{stage}_accuracy_all'] = correct / n
        result[f'{stage}_coverage'] = accepted / n
        result[f'{stage}_covered_accuracy'] = correct / accepted if accepted else None
    return result


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def calculate():
    verify_display()
    folder = ROOT / 'artifacts/evidence_display_audit_v1'
    displays = {}
    for line in (folder / 'displays.jsonl').read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        if record['key'] in displays:
            raise ValueError('Duplicate display key')
        displays[record['key']] = record['display']['candidate_label']
    groups, pooled, used = defaultdict(list), defaultdict(list), set()
    for name in ('fresh_pipeline_received', 'text_stress_received'):
        predictions = json.loads((ROOT / 'artifacts' / name / 'predictions.json').read_text(encoding='utf-8'))
        for row in predictions:
            condition = row.get('condition', 'original')
            key = f"{row['cohort']}:{row['claim_id']}:{condition}:{row['policy']}"
            if key in used or key not in displays:
                raise ValueError('Comparison identity/coverage mismatch')
            used.add(key)
            pair = (row['candidate_label'], displays[key], row['true_label'])
            groups[f"{row['cohort']}:{condition}:{row['policy']}"].append(pair)
            pooled[f"{condition}:{row['policy']}"].append(pair)
    if used != set(displays) or len(used) != 3840:
        raise ValueError('Comparison source coverage changed')
    identity = {'display_audit_manifest_sha256': digest(folder / 'audit_manifest.json'),
                'script_sha256': digest(Path(__file__))}
    summary = {'records': len(used), 'groups': {k: compare(v) for k, v in sorted(groups.items())},
               'pooled_descriptive': {k: compare(v) for k, v in sorted(pooled.items())},
               'scope': 'Already observed benchmark-labelled original/stress contexts. Four policies '
                        'and perturbations share claims. Descriptive admission trade-offs only; '
                        'no neural inference, significance, semantic explanation truth, '
                        'publisher authentication or general robustness benefit established.'}
    return identity, summary


def report(summary):
    lines = ['# Cited-evidence guard: correctness and coverage trade-offs', '', summary['scope'], '',
        'Before uses the saved accepted candidate; after uses the separate cited-evidence display candidate. '
        'Abstention counts as an error in all-record accuracy. These are replay measurements, not a new '
        'model result. Original source outputs remain unchanged.', '',
        '| Condition / policy | Rows | Accuracy before | Accuracy after | Coverage before | Coverage after | Correct withheld | Incorrect withheld |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, value in summary['pooled_descriptive'].items():
        lines.append(f"| {key} | {value['records']} | {value['before_accuracy_all']:.2%} | "
            f"{value['after_accuracy_all']:.2%} | {value['before_coverage']:.2%} | "
            f"{value['after_coverage']:.2%} | {value['withheld_correct']} | {value['withheld_incorrect']} |")
    lines += ['', 'All original candidates are preserved. Altered contexts can lose correct as well as incorrect '
        'answers. Refusing benign OCR changes demonstrates the cost of exact matching. Removing wrong '
        'answers does not imply better all-record accuracy; a filtering guard cannot add correct answers. '
        'A source-bound swapped context can still pass without establishing relevance.', '',
        'The table pools two cohorts for compact descriptive reporting. Cohort-specific counts and covered '
        'accuracy, including null when nothing is accepted, are in summary.json. Original rows contain '
        '600 claims per policy; stress rows contain the same sixty underlying claims per condition/policy. '
        'Baseline stress rows overlap originals. Do not treat rows, conditions or policies as independent '
        'samples or tune the guard on these observed labels.', '',
        'This closes the saved-output trade-off analysis. Source/date/family/trained-robustness experiments, '
        'historical authentication and semantic explanation truth remain outside its scope. No new human '
        'annotation, manuscript or GitHub push is involved.']
    return '\n'.join(lines) + '\n'


def run(verify=False):
    identity, summary = calculate()
    if verify:
        manifest = json.loads((OUTPUT / 'audit_manifest.json').read_text(encoding='utf-8'))
        if manifest['identity'] != identity:
            raise ValueError('Trade-off input/code binding changed')
        if set(manifest['files']) != {'summary.json', 'RESULTS.md'}:
            raise ValueError('Trade-off file inventory changed')
        for name, expected in manifest['files'].items():
            if digest(OUTPUT / name) != expected:
                raise ValueError('Trade-off output hash changed')
        if json.loads((OUTPUT / 'summary.json').read_text(encoding='utf-8')) != summary:
            raise ValueError('Trade-off counts/metrics do not replay')
        if (OUTPUT / 'RESULTS.md').read_text(encoding='utf-8') != report(summary):
            raise ValueError('Trade-off report does not replay')
    else:
        if (OUTPUT / 'audit_manifest.json').exists():
            raise ValueError('Frozen analysis exists; use --verify')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        (OUTPUT / 'RESULTS.md').write_text(report(summary), encoding='utf-8')
        (OUTPUT / 'audit_manifest.json').write_text(json.dumps({'identity': identity, 'files': {
            n: digest(OUTPUT / n) for n in ('summary.json', 'RESULTS.md')}}, indent=2) + '\n', encoding='utf-8')
    print(f"Guard trade-off analysis verified: {summary['records']} saved records; no model inference.")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    run(parser.parse_args().verify)
