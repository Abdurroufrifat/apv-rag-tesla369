"""Replay the opt-in four-policy snapshot controller with saved responses."""
import json
from collections import Counter
from pathlib import Path

from apv_rag.bound_policy_controller import execute_bound_policies
from apv_rag.fresh_pipeline import POLICIES
from apv_rag.splits import sha256, write_json_atomic
from audit_source_trace import DATASETS
from run_fresh_pipeline import ROOT, check_receipt, load, require
from run_snapshot_guard_replay import ENGINE_FIELDS, backend


def cohort_corpora():
    corpora = {}
    hashes = {}
    for cohort, relative in DATASETS.items():
        folder = ROOT / relative
        manifest = load(folder, 'manifest.json')
        require(sha256(folder / 'corpus.jsonl') == manifest['files']['corpus.jsonl'],
                'Frozen corpus mismatch')
        docs = [json.loads(line) for line in (folder / 'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
        corpora[cohort] = {str(d['doc_id']): d for d in docs}
        require(len(corpora[cohort]) == len(docs), 'Duplicate source ID')
        hashes[cohort] = sha256(folder / 'corpus.jsonl')
    return corpora, hashes


def grouped(rows, conditions):
    mapping = {}
    for row in rows:
        key = (row['cohort'], str(row['claim_id']),
               row['condition'] if conditions else 'original')
        policies = mapping.setdefault(key, {})
        policy = row['policy']
        require(policy in POLICIES and policy not in policies, 'Duplicate or unknown policy row')
        policies[policy] = row
    require(all(set(v) == set(POLICIES) for v in mapping.values()), 'Incomplete policy coverage')
    return mapping


def same_engine_output(result, prior, fields):
    for field in fields:
        if field == 'gate_probability' and result[field] is not None and prior[field] is not None:
            if abs(result[field] - prior[field]) > 1e-12:
                return False
        elif result[field] != prior[field]:
            return False
    return True


def replay(rows, contexts, features, corpus, responses, models, conditions):
    groups = grouped(rows, conditions)
    used = set()
    generate = backend(responses, used)
    decisions = []
    for (cohort, claim_id, condition), expected in sorted(groups.items()):
        key = f'{cohort}:{claim_id}' + (f':{condition}' if conditions else '')
        if conditions:
            require(key in contexts, 'Missing stress context')
            prepared = contexts[key]
            for prior in expected.values():
                require(prior['evidence'] == prepared['evidence'] and
                        prior['true_label'] == prepared['true_label'],
                        'Stress row/context mismatch')
        else:
            prior = expected['no_gate']
            prepared = {k: prior[k] for k in ('claim', 'shown_claim', 'retrieved_evidence', 'evidence')}
            for other in expected.values():
                require(all(other[k] == prepared[k] for k in prepared),
                        'Original policy contexts disagree')
        entry = features.get(key)
        results = execute_bound_policies(prepared, entry, models, corpus[cohort], generate)
        for result in results:
            prior = expected[result['policy']]
            if result['snapshot_source_bound']:
                require(same_engine_output(result, prior, ENGINE_FIELDS),
                        f'Bound policy changed saved output: {key}/{result["policy"]}')
            else:
                require(not result['generation_requests'] and result['candidate_label'] is None,
                        'Unbound context reached generation')
            decisions.append({'cohort': cohort, 'claim_id': claim_id, 'condition': condition,
                              'policy': result['policy'], 'snapshot_source_bound': result['snapshot_source_bound'],
                              'source_trace': result['source_trace'], 'reasons': result['reasons'],
                              'candidate_label': result['candidate_label'],
                              'original_candidate_label': prior['candidate_label'],
                              'generation_requests': result['generation_requests'],
                              'original_generation_requests': prior['generation_requests']})
    require(set(contexts) == {f'{a}:{b}:{c}' for a, b, c in groups} if conditions else True,
            'Unexpected stress context keys')
    return decisions, used


def main():
    artifact = ROOT / 'artifacts'
    frozen = artifact / 'fresh_pipeline_received'
    stress = artifact / 'text_stress_received'
    learned = artifact / 'semantic_sufficiency_received'
    receipts = {p.name: sha256(p / 'output_manifest.json') for p in (frozen, stress, learned)}
    for folder in (frozen, stress, learned):
        check_receipt(folder)
    corpus, corpus_hashes = cohort_corpora()
    models = load(learned, 'models.json')
    original, used_original = replay(load(frozen, 'predictions.json'), {},
        load(frozen, 'feature_cache.json'), corpus, load(frozen, 'responses.json'),
        models, False)
    modified, used_stress = replay(load(stress, 'predictions.json'),
        load(stress, 'contexts.json'), load(stress, 'feature_cache.json'),
        corpus, load(stress, 'responses.json'), models, True)
    require(len(original) == 2400 and len(modified) == 1440, 'Unexpected replay coverage')
    require(all(row['snapshot_source_bound'] for row in original), 'Original context rejected')
    counts = {}
    for row in original + modified:
        key = f"{row['cohort']}:{row['condition']}:{row['policy']}"
        group = counts.setdefault(key, Counter())
        group['records'] += 1
        group['bound'] += row['snapshot_source_bound']
        group['refused'] += not row['snapshot_source_bound']
        group['prior_verdict_requests_suppressed'] += int(
            not row['snapshot_source_bound'] and 'verdict' in row['original_generation_requests'])
    out = artifact / 'bound_policy_integration_v1'
    out.mkdir(exist_ok=True)
    write_json_atomic(out / 'decisions.json', original + modified)
    summary = {'scope': 'Opt-in four-policy controller on frozen corpus snapshots, saved features and exact cached responses; no fresh neural inference or publisher authentication.',
               'source_output_manifests_sha256': receipts, 'corpus_sha256': corpus_hashes,
               'original_policy_records': len(original), 'stress_policy_records': len(modified),
               'original_contexts': len(original) // len(POLICIES),
               'stress_contexts': len(modified) // len(POLICIES),
               'distinct_cached_responses_used': {'original': len(used_original),
                                                   'stress': len(used_stress)},
               'counts': {k: dict(v) for k, v in sorted(counts.items())}}
    write_json_atomic(out / 'summary.json', summary)
    total = Counter()
    for row in modified:
        if row['policy'] == 'no_gate':
            total['bound'] += row['snapshot_source_bound']
            total['refused'] += not row['snapshot_source_bound']
    (out / 'RESULTS.md').write_text(
        '# Bound four-policy controller replay\n\n'
        'The opt-in controller checks every retrieved passage against its selected text in the frozen benchmark corpus before any feature evaluation or generator call. The 600 original contexts produce the same saved outputs across all four policies (2400 policy records). The 360 saved stress contexts are replayed across all four policies (1440 records).\n\n'
        f"Stress contexts: {total['bound']} bound, {total['refused']} refused per policy. "
        'Refused contexts do not request generation. The rule deliberately refuses modified OCR text and can pass an unchanged excerpt under a swapped context. Saved features and exact answers were reused; this is an integration replay, not new model inference or a gain in independent accuracy. It does not authenticate publishers or verify explanation truth. Per-condition and per-policy counts are in summary.json.\n',
        encoding='utf-8')
    write_json_atomic(out / 'audit_manifest.json', {'source_output_manifests_sha256': receipts,
        'corpus_sha256': corpus_hashes, 'code_sha256': {
            name: sha256(ROOT / name) for name in (
                'src/apv_rag/bound_policy_controller.py', 'scripts/replay_bound_policies.py')},
        'files': {p.name: sha256(p) for p in out.iterdir() if p.name != 'audit_manifest.json'}})
    print('Bound policy replay passed: 2400 original and 1440 stress policy records.')


if __name__ == '__main__':
    main()
