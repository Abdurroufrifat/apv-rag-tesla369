"""Build or replay exact cited-excerpt displays for saved English outputs."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.evidence_display import build_evidence_display, verify_evidence_display

OUTPUT = ROOT / 'artifacts/evidence_display_audit_v1'
DATASETS = {'scifact': 'data/external/scifact/sealed_v1',
            'climate_retrieved': 'data/external/climate_fever/frozen_v1'}
CODE = ('src/apv_rag/evidence_display.py', 'src/apv_rag/source_snapshot_guard.py',
        'src/apv_rag/sentence_context.py', 'src/apv_rag/integrated_gate.py',
        'src/apv_rag/generative_rag.py', 'scripts/audit_evidence_display.py',
        'docs/EVIDENCE_DISPLAY_PROTOCOL.md')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked_outputs(folder):
    manifest = load(folder / 'output_manifest.json')
    for name, expected in manifest.items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError(f'Saved output changed: {folder.name}/{name}')
    return digest(folder / 'output_manifest.json')


def inputs():
    corpora, identity = {}, {'source_receipts_sha256': {}, 'corpus_sha256': {},
                             'code_sha256': {name: digest(ROOT / name) for name in CODE}}
    for cohort, relative in DATASETS.items():
        folder = ROOT / relative
        expected = load(folder / 'manifest.json')['files']['corpus.jsonl']
        observed = digest(folder / 'corpus.jsonl')
        if observed != expected:
            raise ValueError('Frozen corpus changed')
        docs = [json.loads(line) for line in (folder / 'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
        corpora[cohort] = {str(d['doc_id']): d for d in docs}
        if len(corpora[cohort]) != len(docs):
            raise ValueError('Duplicate corpus document IDs')
        identity['corpus_sha256'][cohort] = observed
    rows = {}
    for source, verifier in (('fresh_pipeline_received', 'fresh_pipeline_verification_v1'),
                             ('text_stress_received', 'text_stress_verification_v1')):
        folder = ROOT / 'artifacts' / source
        receipt_sha = checked_outputs(folder)
        verification = load(ROOT / 'artifacts' / verifier / 'audit_manifest.json')
        if verification['source_output_manifest_sha256'] != receipt_sha:
            raise ValueError('Saved prediction verifier binding changed')
        identity['source_receipts_sha256'][source] = receipt_sha
        contexts = load(folder / 'contexts.json') if source == 'text_stress_received' else None
        predictions = load(folder / 'predictions.json')
        if len(predictions) != (1440 if contexts is not None else 2400):
            raise ValueError('Saved policy coverage changed')
        for row in predictions:
            condition = row.get('condition', 'original')
            if contexts is not None:
                context = contexts[f"{row['cohort']}:{row['claim_id']}:{condition}"]
                if row['evidence'] != context['evidence']:
                    raise ValueError('Saved stress context differs')
                row = {**context, **row}
            key = f"{row['cohort']}:{row['claim_id']}:{condition}:{row['policy']}"
            if key in rows:
                raise ValueError('Duplicate display source key')
            # No gold labels or free-form explanations enter the renderer.
            rows[key] = {name: row[name] for name in ('claim', 'claim_id', 'cohort',
                'retrieved_evidence', 'evidence', 'candidate_label', 'policy')}
            rows[key]['condition'] = condition
    if len(rows) != 3840:
        raise ValueError('Display source coverage changed')
    return rows, corpora, identity


def outputs(rows, corpora):
    records, groups = [], defaultdict(Counter)
    for key, row in sorted(rows.items()):
        display = build_evidence_display(row, corpora[row['cohort']])
        verify_evidence_display(display, row, corpora[row['cohort']])
        records.append({'key': key, 'display': display})
        count = groups[f"{row['cohort']}:{row['condition']}:{row['policy']}"]
        count['records'] += 1
        count['upstream_candidates'] += row['candidate_label'] is not None
        count['displayed_candidates'] += display['candidate_label'] is not None
        count['display_abstentions'] += display['status'] == 'abstain'
        count['unbound_contexts'] += not display['source_snapshot_bound']
        count['suppressed_upstream_candidates'] += (row['candidate_label'] is not None and
                                                   display['candidate_label'] is None)
        count['cited_excerpts'] += len(display['items'])
    totals = Counter()
    for group in groups.values():
        totals.update(group)
    summary = {'stage': 'evidence_display_v1', 'records': len(records),
               'unique_claim_contexts': len({key.rsplit(':', 1)[0] for key in rows}),
               'groups': {key: dict(value) for key, value in sorted(groups.items())},
               'totals': dict(totals), 'support_relationships_verified': 0,
               'neural_model_calls': 0, 'new_human_labels': 0,
               'scope': 'Exact cited frozen-snapshot excerpts only. Original candidates and free-form '
                        'outputs remain unchanged. No semantic rationale, explanation truth, publisher '
                        'authentication or model accuracy improvement is established.'}
    return records, summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def report(summary):
    total = summary['totals']
    lines = ['# Exact cited-excerpt display audit', '', summary['scope'], '',
        f"Replayed {summary['records']} policy records from {summary['unique_claim_contexts']} distinct "
        'claim/condition contexts. Four policies share the contexts and are not independent samples.', '',
        f"Displayed candidates: {total['displayed_candidates']}; display abstentions: "
        f"{total['display_abstentions']}; source-bound cited excerpts: {total['cited_excerpts']}; "
        f"suppressed upstream candidates: {total['suppressed_upstream_candidates']}.", '',
        '| Cohort / condition / policy | Rows | Upstream candidates | Displayed candidates | Unbound contexts | Cited excerpts |',
        '|---|---:|---:|---:|---:|---:|']
    for key, group in summary['groups'].items():
        lines.append(f"| {key} | {group['records']} | {group['upstream_candidates']} | "
                     f"{group['displayed_candidates']} | {group['unbound_contexts']} | {group['cited_excerpts']} |")
    lines += ['', 'Every displayed item records its source document ID, copied excerpt, original sentence-selection '
        'indices, excerpt SHA-256 and full frozen document SHA-256. A candidate is shown only when the existing '
        'snapshot check admits every retrieved passage and the collapsed context matches. Upstream abstentions '
        'remain abstentions. No generated rationale text is copied into the new display.', '',
        'This display is an evidence packet, not a natural-language explanation of why the verdict follows. '
        'Literal occurrence cannot establish entailment, relevance, completeness or source truth. The prior '
        'NLI explanation diagnostic remains separate and has not selected a gate threshold. A swapped context '
        'that happens to satisfy the existing selection rule can still pass. Benign OCR edits fail the strict '
        'snapshot rule. A Not Enough Evidence candidate does not prove absence of evidence.', '',
        'No new accuracy, significance, semantic truth or explanation quality score is reported. These are '
        'deterministic display/admission counts on already observed outputs. The source, generated verdicts, '
        'free-form explanations, model caches and original benchmark results are unchanged. No manuscript '
        'or GitHub push was performed.']
    return '\n'.join(lines) + '\n'


def build(out):
    if (out / 'audit_manifest.json').exists():
        raise ValueError('Frozen display audit exists; verify it or use a separate output directory')
    rows, corpora, identity = inputs()
    records, summary = outputs(rows, corpora)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'displays.jsonl').write_text(''.join(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
                                              for record in records), encoding='utf-8')
    write_json(out / 'summary.json', summary)
    (out / 'RESULTS.md').write_text(report(summary), encoding='utf-8')
    write_json(out / 'audit_manifest.json', {'identity': identity, 'files': {
        name: digest(out / name) for name in ('displays.jsonl', 'summary.json', 'RESULTS.md')}})
    return verify(out)


def verify(out=OUTPUT):
    rows, corpora, identity = inputs()
    manifest = load(out / 'audit_manifest.json')
    if manifest['identity'] != identity:
        raise ValueError('Display source/code/protocol binding changed')
    if set(manifest['files']) != {'displays.jsonl', 'summary.json', 'RESULTS.md'}:
        raise ValueError('Display audit inventory changed')
    if {p.name for p in out.iterdir()} != set(manifest['files']) | {'audit_manifest.json'}:
        raise ValueError('Unexpected display audit file')
    for name, expected in manifest['files'].items():
        if digest(out / name) != expected:
            raise ValueError(f'Display output file changed: {name}')
    expected_records, expected_summary = outputs(rows, corpora)
    actual = [json.loads(line) for line in (out / 'displays.jsonl').read_text(encoding='utf-8').splitlines()]
    if actual != expected_records or load(out / 'summary.json') != expected_summary:
        raise ValueError('Evidence display records or summary do not replay')
    if (out / 'RESULTS.md').read_text(encoding='utf-8') != report(expected_summary):
        raise ValueError('Evidence display report does not replay')
    return expected_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.build and args.verify:
        parser.error('Choose build or offline verify')
    summary = build(args.output) if args.build else verify(args.output)
    print(f"Evidence display audit verified: {summary['records']} records; "
          f"{summary['totals']['cited_excerpts']} exact cited excerpts.")
    print('Neural model calls: 0. Semantic explanation truth remains unverified.')


if __name__ == '__main__':
    main()
