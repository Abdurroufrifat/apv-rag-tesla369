"""Exact-copy/order gate replay; all neural scores and responses are received data."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from apv_rag.fresh_pipeline import POLICIES
from apv_rag.gated_stress import transform_entry, execute_stress_policy
from apv_rag.generation_robustness import CONDITIONS, context_variants, explanation_ids, question
from apv_rag.generative_rag import LABELS
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2
from apv_rag.splits import sha256, write_json_atomic
from run_fresh_pipeline import ROOT, CODE as FRESH_CODE, check_receipt, load, require, response_key
from run_gated_generation import qwen_prompt
from run_sentence_rag_scifact import summarize
from verify_fresh_pipeline import verify_export

CODE = tuple(dict.fromkeys(('scripts/run_gated_stress_replay.py', 'scripts/verify_gated_stress_replay.py',
    'src/apv_rag/gated_stress.py', 'src/apv_rag/generation_robustness.py', *FRESH_CODE)))
SETTINGS = {'copy_count': 3, 'conditions': list(CONDITIONS), 'policies': list(POLICIES),
            'gate_threshold': .5, 'explanation_claims_per_cohort': 60,
            'features': 'exact-text per-document score transformation; no new neural inference',
            'generation': 'verified prior responses only; no live fallback',
            'copies_raw': 'normalization-disabled diagnostic ablation',
            'copies_collapsed': 'normalize before features and generation; forced baseline equality',
            'verdict_metrics': 'all 300 per cohort; gate only',
            'guard_metrics': 'fixed 60 per cohort; gate plus numeric explanation guard'}


def prepare(root=ROOT):
    fresh = root / 'artifacts/fresh_pipeline_received'
    verify_export(fresh, root)
    stress = root / 'artifacts/generation_robustness_received'
    check_receipt(stress)
    fresh_meta = load(fresh, 'input_manifest.json')
    stress_meta = load(stress, 'input_manifest.json')
    require(stress_meta['generator'] == fresh_meta['generator'], 'Generator identity mismatch')
    require(all(fresh_meta['packages'][n] == v for n, v in stress_meta['packages'].items()), 'Provider package mismatch')
    for name, digest in stress_meta['code_sha256'].items():
        require(name in {'scripts/run_generation_robustness.py','src/apv_rag/generation_robustness.py',
                        'scripts/run_gated_generation.py','scripts/verify_gated_generation.py',
                        'src/apv_rag/integrated_gate.py','src/apv_rag/generative_interface.py',
                        'src/apv_rag/numeric_integrity_v2.py','src/apv_rag/input_numeric_integrity.py'} and
                sha256(root / name) == digest, 'Stress provider code changed')
    require(sha256(root / 'docs/GENERATION_ROBUSTNESS.md') == stress_meta['protocol_sha256'], 'Stress provider protocol changed')
    baseline_rows = [r for r in load(fresh, 'predictions.json') if r['policy'] == 'no_gate']
    baseline = {f"{r['cohort']}:{r['claim_id']}": r for r in baseline_rows}
    require(len(baseline_rows) == len(baseline) == 600, 'Unexpected baseline coverage')
    selected = {cohort: explanation_ids([r for r in baseline_rows if r['cohort'] == cohort], cohort)
                for cohort in ('scifact', 'climate_retrieved')}
    require(stress_meta['explanation_claim_ids'] == selected, 'Explanation subset changed')
    rows = load(stress, 'predictions.json')
    lookup = {(r['cohort'], str(r['claim_id']), r['condition']): r for r in rows}
    expected = {(r['cohort'], str(r['claim_id']), c) for r in baseline_rows for c in CONDITIONS}
    require(len(rows) == len(lookup) == 2400 and set(lookup) == expected, 'Stress provider cohort changed')
    responses = {}
    for key, original in baseline.items():
        cohort, claim_id = key.split(':', 1)
        explained = claim_id in selected[cohort]
        for condition, evidence in context_variants(original['evidence']).items():
            r = lookup[cohort, claim_id, condition]
            require(r['shown_claim'] == original['shown_claim'] and r['evidence'] == evidence and
                    r['true_label'] == original['true_label'] and r['explanation_selected'] == explained,
                    'Stress context, label or selection mismatch')
            require(r['generated_verdict'] in LABELS and r['raw_candidate_label'] == r['generated_verdict'], 'Invalid stress verdict')
            kinds = ('verdict', 'explanation') if explained else ('verdict',)
            for kind in kinds:
                prompt = qwen_prompt(question(original['shown_claim'], evidence,
                                             r['generated_verdict'] if kind == 'explanation' else None))
                require(r[kind + '_prompt'] == prompt, 'Prior response prompt binding changed')
                response = {'answer': r['generated_' + kind], 'prompt': prompt,
                            'prompt_tokens': r[kind + '_prompt_tokens']}
                require(isinstance(response['answer'], str) and type(response['prompt_tokens']) is int and
                        0 < response['prompt_tokens'] <= 1024, 'Invalid prior response')
                response_id = response_key(kind, prompt)
                require(response_id not in responses or responses[response_id] == response, 'Conflicting prior response')
                responses[response_id] = response
                if condition in ('baseline', 'copies_collapsed'):
                    require(response == {n: original[kind + '_prompt'] if n == 'prompt' else
                            original[kind + '_prompt_tokens'] if n == 'prompt_tokens' else original['generated_' + kind]
                            for n in ('answer', 'prompt', 'prompt_tokens')}, 'Normalized prior differs from fresh baseline')
            numeric, reasons = None, []
            if explained:
                numeric = numeric_provenance_v2(r['generated_explanation'], original['shown_claim'], evidence)
                if not r['generated_explanation']:reasons.append('empty_explanation')
                if numeric['absent_from_inputs']:reasons.append('numeric_value_absent')
            else:
                require(all(r[n] is None for n in ('generated_explanation','explanation_prompt','explanation_prompt_tokens')), 'Unexpected explanation')
            require(r['numeric_provenance'] == numeric and r['reasons'] == reasons and
                    r['explanation_guarded_label'] == (r['generated_verdict'] if explained and not reasons else None),
                    'Prior numeric guard changed')
    source_files = {}
    for folder in (fresh, stress):
        for p in folder.glob('*.json'):source_files[p.relative_to(root).as_posix()] = sha256(p)
    models_path = root / 'artifacts/semantic_sufficiency_received/models.json'
    source_files[models_path.relative_to(root).as_posix()] = sha256(models_path)
    return {'baseline': baseline, 'selected': selected, 'responses': responses,
            'features': load(fresh, 'feature_cache.json'), 'models': json.loads(models_path.read_text(encoding='utf-8')),
            'source_files': source_files, 'generator': fresh_meta['generator'],
            'feature_models': fresh_meta['feature_models'], 'provider_packages': fresh_meta['packages']}


def replay(inputs):
    used = {}
    def generate(kind, q):
        prompt = qwen_prompt(q);key = response_key(kind, prompt)
        require(key in inputs['responses'], 'Unrecorded prompt; refusing new generation')
        response = inputs['responses'][key]
        used[key] = {'kind': kind, **response, 'origin': 'verified_generation_robustness_export'}
        return response
    records, features = [], {}
    for key, original in inputs['baseline'].items():
        cohort, claim_id = key.split(':', 1)
        explained = claim_id in inputs['selected'][cohort]
        for condition, evidence in context_variants(original['evidence']).items():
            feature_key = key + ':' + condition
            entry = transform_entry(original['shown_claim'], original['evidence'], inputs['features'][key], evidence)
            features[feature_key] = entry
            for policy in POLICIES:
                model = None if policy == 'no_gate' else inputs['models'][policy]
                r = execute_stress_policy(original['shown_claim'], evidence, entry, model, generate, explained)
                r.update(cohort=cohort, claim_id=claim_id, condition=condition, policy=policy,
                         context_sha256=entry['context_sha256'], true_label=original['true_label'])
                records.append(r)
    return records, features, used


def summarize_replay(rows, responses):
    lookup = {(r['cohort'], r['claim_id'], r['condition'], r['policy']): r for r in rows}
    result = {}
    for cohort in ('scifact', 'climate_retrieved'):
        result[cohort] = {}
        for condition in CONDITIONS:
            result[cohort][condition] = {}
            for policy in POLICIES:
                subset = [r for r in rows if r['cohort'] == cohort and r['condition'] == condition and r['policy'] == policy]
                explained = [r for r in subset if r['explanation_selected']]
                base = [lookup[cohort, r['claim_id'], 'baseline', policy] for r in subset]
                accepted = [r['gate_probability'] is None or r['gate_probability'] >= .5 for r in subset]
                old_accepted = [r['gate_probability'] is None or r['gate_probability'] >= .5 for r in base]
                shifts = [r['gate_probability'] - b['gate_probability'] for r, b in zip(subset, base, strict=True)] if policy != 'no_gate' else []
                result[cohort][condition][policy] = {
                    'rows': len(subset), 'explained_subset_rows': len(explained),
                    'gate_only_verdict_metrics': summarize(subset, 'gate_verdict_label'),
                    'explanation_subset_guarded_metrics': summarize(explained, 'explanation_guarded_label'),
                    'gate_rejections_before_replay': sum('gate_below_threshold' in r['reasons'] for r in subset),
                    'verdict_response_requests': sum('verdict' in r['generation_requests'] for r in subset),
                    'explanation_response_requests': sum('explanation' in r['generation_requests'] for r in subset),
                    'gate_acceptance_changes': sum(a != b for a, b in zip(accepted, old_accepted, strict=True)),
                    'gate_verdict_or_abstention_changes': sum(r['gate_verdict_label'] != b['gate_verdict_label'] for r, b in zip(subset, base, strict=True)),
                    'mean_completeness_probability_shift': float(np.mean(shifts)) if shifts else None,
                    'max_absolute_completeness_probability_shift': float(max(map(abs, shifts))) if shifts else None,
                    'numeric_subset_rejections': sum('numeric_value_absent' in r['reasons'] for r in explained)}
    return {'policy_records': len(rows), 'claims': 600, 'conditions': list(CONDITIONS), 'cohorts': result,
            'distinct_prior_responses_used': len(responses), 'new_neural_inference_calls': 0,
            'scope': 'Post-hoc policy replay on observed English claims. Exact score transforms and prior answers; no new model run or calibration evidence.'}


def identity(inputs, root=ROOT):
    return {'schema_version': 1, 'source_files': inputs['source_files'], 'generator': inputs['generator'],
            'feature_models': inputs['feature_models'], 'provider_packages': inputs['provider_packages'],
            'analysis_packages': {n: version(n) for n in ('numpy','scikit-learn')},
            'settings': SETTINGS, 'code_sha256': {n: sha256(root/n) for n in CODE},
            'protocol_sha256': sha256(root/'docs/GATED_STRESS_REPLAY.md')}


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--preflight', action='store_true');args = parser.parse_args()
    inputs = prepare()
    if args.preflight:
        print(f"Gated stress inputs checked: 600 claims, 2400 contexts, {len(inputs['responses'])} verified responses; no model calls.")
        return
    out = ROOT/'artifacts/gated_stress_replay_v1'
    require(not (out/'output_manifest.json').exists(), 'Completed replay exists; verify it instead of overwriting')
    out.mkdir(exist_ok=True)
    rows, features, responses = replay(inputs)
    write_json_atomic(out/'input_manifest.json', identity(inputs))
    write_json_atomic(out/'transformed_features.json', features)
    write_json_atomic(out/'predictions.json', rows)
    write_json_atomic(out/'responses.json', responses)
    write_json_atomic(out/'summary.json', summarize_replay(rows, responses))
    write_json_atomic(out/'output_manifest.json', {p.name: sha256(p) for p in out.glob('*.json') if p.name != 'output_manifest.json'})
    print(f'Gated stress replay completed: {len(rows)} policy records; no new inference.')
    print(out)


if __name__ == '__main__':main()
