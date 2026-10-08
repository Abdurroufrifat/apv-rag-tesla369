"""Replay frozen copy/order exports and exploratory paired group statistics."""
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np

from apv_rag.generative_rag import LABELS
from apv_rag.generation_robustness import CONDITIONS, context_variants, question
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2
from apv_rag.paired_statistics import holm_adjust
from apv_rag.scifact import normalized_claim
from apv_rag.splits import sha256, write_json_atomic
from run_gated_generation import qwen_prompt
from run_generation_robustness import prepare_sources
from run_sentence_rag_scifact import summarize
from verify_integrated_gate import check_hashes, load, require
from verify_xfever_generation import compare, paired_result


def context_groups(rows):
    """Connect identical normalized claims or shared ORIGINAL retrieved sources."""
    parent = list(range(len(rows)))
    owners = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(rows):
        keys = [('claim', normalized_claim(row['claim']))]
        keys += [('source', str(e['id'])) for e in row['evidence']]
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    groups = {}
    membership = []
    for i in range(len(rows)):
        membership.append(groups.setdefault(find(i), len(groups)))
    return np.array(membership), len(groups)


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'artifacts/generation_robustness_received'
    check_hashes(folder)
    require(set(load(folder, 'output_manifest.json')) == {'input_manifest.json', 'predictions.json', 'summary.json'},
            'Unexpected output manifest schema')
    sources, baseline, selected, provider = prepare_sources(root)
    identity = load(folder, 'input_manifest.json')
    require(identity['generator'] == provider['generator'] and identity['packages'] == provider['packages'],
            'Generator/package identity mismatch')
    require(identity['provider_output_manifest_sha256'] == sha256(root / 'artifacts/gated_generation_received/output_manifest.json'),
            'Baseline receipt mismatch')
    paths = {'scifact': 'sentence_rag_received', 'climate_retrieved': 'climate_rag_received'}
    require(identity['source_sha256'] == {c: sha256(root / 'artifacts' / d / 'predictions.json') for c, d in paths.items()},
            'Source identity mismatch')
    require(identity['explanation_claim_ids'] == selected, 'Explanation selection mismatch')
    require(identity['settings'] == {'copy_count': 3, 'conditions': list(CONDITIONS), 'explanation_claims_per_cohort': 60,
            'seed': 369, 'threads': 4, 'dtype': 'float32', 'max_input_tokens': 1024, 'max_new_tokens': 128,
            'do_sample': False, 'num_beams': 1, 'learned_gate': 'none; source normalization control only',
            'context': 'original frozen clipped text'}, 'Settings mismatch')
    code_paths = {'scripts/run_generation_robustness.py', 'src/apv_rag/generation_robustness.py',
                  'scripts/run_gated_generation.py', 'scripts/verify_gated_generation.py', 'src/apv_rag/integrated_gate.py',
                  'src/apv_rag/generative_interface.py', 'src/apv_rag/numeric_integrity_v2.py',
                  'src/apv_rag/input_numeric_integrity.py'}
    require(set(identity['code_sha256']) == code_paths, 'Code manifest schema mismatch')
    for name, digest in identity['code_sha256'].items():
        require(sha256(root / name) == digest, f'Code mismatch: {name}')
    require(identity['protocol_sha256'] == sha256(root / 'docs/GENERATION_ROBUSTNESS.md'), 'Protocol mismatch')
    rows = load(folder, 'predictions.json')
    lookup = {(r['cohort'], str(r['claim_id']), r['condition']): r for r in rows}
    expected = {(c, str(s['claim_id']), condition) for c, values in sources.items() for s in values for condition in CONDITIONS}
    require(len(rows) == len(lookup) == 2400 and set(lookup) == expected, 'Output cohort mismatch')
    shared, seeded = {}, set()
    explanation_count = 0
    for cohort, originals in sources.items():
        require(len(originals) == 300, 'Unexpected original cohort size')
        for source in originals:
            claim_id = str(source['claim_id'])
            prior = baseline[cohort, claim_id]
            selected_row = claim_id in selected[cohort]
            for kind in ('verdict', 'explanation') if selected_row else ('verdict',):
                seeded.add(kind + ':' + hashlib.sha256(prior[kind + '_prompt'].encode('utf-8')).hexdigest())
            for condition, evidence in context_variants(source['evidence']).items():
                r = lookup[cohort, claim_id, condition]
                require(r['claim_id'] == source['claim_id'] and r['shown_claim'] == source['shown_claim']
                        and r['evidence'] == evidence and r['true_label'] == source['true_label'] in LABELS,
                        'Source, context or label mismatch')
                verdict = r['generated_verdict']
                require(verdict in LABELS and r['raw_candidate_label'] == verdict, 'Invalid verdict')
                require(r['verdict_prompt'] == qwen_prompt(question(source['shown_claim'], evidence)), 'Verdict prompt mismatch')
                require(r['explanation_selected'] == selected_row, 'Selection mismatch')
                numeric, reasons = None, []
                if selected_row:
                    explanation_count += 1
                    require(isinstance(r['generated_explanation'], str), 'Missing explanation')
                    require(r['explanation_prompt'] == qwen_prompt(question(source['shown_claim'], evidence, verdict)),
                            'Explanation prompt mismatch')
                    numeric = numeric_provenance_v2(r['generated_explanation'], source['shown_claim'], evidence)
                    if not r['generated_explanation']:
                        reasons.append('empty_explanation')
                    if numeric['absent_from_inputs']:
                        reasons.append('numeric_value_absent')
                else:
                    require(all(r[k] is None for k in ('generated_explanation', 'explanation_prompt', 'explanation_prompt_tokens')),
                            'Unexpected explanation outside subset')
                require(r['numeric_provenance'] == numeric and r['reasons'] == reasons, 'Numeric policy mismatch')
                require(r['explanation_guarded_label'] == (verdict if selected_row and not reasons else None), 'Guard mismatch')
                for kind in ('verdict', 'explanation') if selected_row else ('verdict',):
                    tokens = r[kind + '_prompt_tokens']
                    require(type(tokens) is int and 0 < tokens <= 1024, 'Reported prompt budget mismatch')
                    prompt = r[kind + '_prompt']
                    key = kind + ':' + hashlib.sha256(prompt.encode('utf-8')).hexdigest()
                    response = (r['generated_' + kind], tokens)
                    require(key not in shared or shared[key] == response, 'Conflicting shared responses')
                    shared[key] = response
                    if condition in ('baseline', 'copies_collapsed'):
                        require(prompt == prior[kind + '_prompt'] and response ==
                                (prior['generated_' + kind], prior[kind + '_prompt_tokens']), 'Frozen response mismatch')
    require(explanation_count == 480 and seeded <= set(shared), 'Explanation/cache key mismatch')
    summaries, changes, stats, group_details = {}, {}, [], {}
    mapping = {label: i for i, label in enumerate(LABELS)}
    for cohort, originals in sources.items():
        summaries[cohort], changes[cohort] = {}, {}
        membership, groups = context_groups(originals)
        sizes = Counter(membership.tolist())
        group_details[cohort] = {'groups': groups, 'sizes_descending': sorted(sizes.values(), reverse=True),
                                 'claim_ids_by_group': [[str(s['claim_id']) for i, s in enumerate(originals) if membership[i] == g]
                                                        for g in range(groups)]}
        rng = np.random.default_rng(369)
        bootstrap = rng.multinomial(groups, np.full(groups, 1 / groups), size=2000)
        swaps = rng.integers(0, 2, size=(10000, groups), dtype=np.int8)
        truth = np.array([mapping[s['true_label']] for s in originals])
        before = [lookup[cohort, str(s['claim_id']), 'baseline'] for s in originals]
        left = np.array([mapping[r['raw_candidate_label']] for r in before])
        for condition in CONDITIONS:
            subset = [lookup[cohort, str(s['claim_id']), condition] for s in originals]
            explained = [r for r in subset if r['explanation_selected']]
            flip = sum(r['raw_candidate_label'] != b['raw_candidate_label'] for r, b in zip(subset, before, strict=True))
            summaries[cohort][condition] = {'rows': len(subset), 'verdict_metrics': summarize(subset, 'raw_candidate_label'),
                'verdict_disagreement_with_baseline': flip / len(subset), 'explained_rows': len(explained),
                'explanation_subset_raw_verdict_metrics': summarize(explained, 'raw_candidate_label'),
                'explanation_subset_guarded_metrics': summarize(explained, 'explanation_guarded_label'),
                'numeric_rejections': sum('numeric_value_absent' in r['reasons'] for r in explained)}
            transitions = Counter(('correct' if b['raw_candidate_label'] == b['true_label'] else 'wrong') + '_to_' +
                                  ('correct' if r['raw_candidate_label'] == r['true_label'] else 'wrong')
                                  for r, b in zip(subset, before, strict=True))
            changes[cohort][condition] = {'verdict_flips': flip, 'correctness_transitions': dict(transitions),
                'wrong_to_wrong_verdict_flips': sum(r['raw_candidate_label'] != b['raw_candidate_label'] and
                    r['raw_candidate_label'] != r['true_label'] and b['raw_candidate_label'] != b['true_label']
                    for r, b in zip(subset, before, strict=True)),
                'selected_explanation_text_changes': sum(r['generated_explanation'] !=
                    lookup[cohort, str(r['claim_id']), 'baseline']['generated_explanation'] for r in explained)}
            if condition in ('copies_raw', 'reverse_order'):
                right = np.array([mapping[r['raw_candidate_label']] for r in subset])
                result = paired_result(truth, left, right, membership, groups, bootstrap, swaps)
                a = np.bincount(membership, weights=(left == truth), minlength=groups)
                b = np.bincount(membership, weights=(right == truth), minlength=groups)
                counts = np.bincount(membership, minlength=groups)
                effects = (bootstrap @ (b - a)) / (bootstrap @ counts)
                result.update(accuracy_difference=float((right == truth).mean() - (left == truth).mean()),
                              marginal_accuracy_group_bootstrap_95_interval=np.quantile(effects, [.025, .975]).tolist())
                stats.append({'cohort': cohort, 'condition': condition, **result})
    replay = {'cohorts': summaries, 'frozen_baseline_prompt_keys_seeded': len(seeded),
              'total_cached_prompt_keys': len(shared), 'new_live_prompt_keys': len(set(shared) - seeded),
              'scope': 'synthetic exact-copy/order generator stress on frozen retrieved contexts; no source authentication or learned gate'}
    compare(load(folder, 'summary.json'), replay)
    for r, adjusted in zip(stats, holm_adjust([r['group_swap_two_sided_p'] for r in stats]), strict=True):
        r['holm_p_four_macro_f1_comparisons'] = float(adjusted)
    out = root / 'artifacts/generation_robustness_verification'
    out.mkdir(exist_ok=True)
    write_json_atomic(out / 'verified_summary.json', replay)
    write_json_atomic(out / 'changes.json', changes)
    write_json_atomic(out / 'statistics.json', {'comparisons': stats, 'group_details': group_details,
        'group_rule': 'connected normalized original claim or shared original retrieved source ID within cohort',
        'analysis_scope': 'exploratory; grouping rule specified during export analysis, not preregistered',
        'seed': 369, 'bootstrap_samples': 2000, 'group_swap_samples': 10000,
        'packages': {n: version(n) for n in ('numpy', 'scikit-learn')}})
    write_json_atomic(out / 'verification_checks.json', {'verdict_records': len(rows), 'explanation_records': explanation_count,
        'original_claims': 600, 'unique_prompt_keys': len(shared), 'seeded_baseline_keys': len(seeded),
        'hashes_source_context_prompt_policy_metrics_verified': True, 'neural_inference_rerun': False,
        'tokenizer_counts_recomputed': False, 'sqlite_cache_inspected': False, 'local_model_bytes_verified': False})
    lines = ['# Generation robustness export verification', '',
        'All 2400 verdict records and 480 selected explanation records passed output/source/code/protocol hashes, prior generator/package identity declarations, exact source/context/prompt binding, deterministic subset selection, reported token budgets, shared-response consistency, numeric policy and metric replay. The 720 seeded and 2160 total prompt keys match the exported response records. New inference count is a receipt declaration consistent with 1440 new keys; local model bytes, inference, tokenizer counts and SQLite were not independently inspected or rerun.', '',
        '| Cohort | Condition | Correct / 300 | Accuracy | Macro F1 | Verdict flips | Wrong → correct | Correct → wrong |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in sources:
        for condition in CONDITIONS:
            m = summaries[c][condition]['verdict_metrics']
            t = changes[c][condition]['correctness_transitions']
            flips = changes[c][condition]['verdict_flips']
            lines.append(f"| {c} | {condition} | {round(m['covered_accuracy'] * 300)} | {m['covered_accuracy']:.2%} | {m['macro_f1_all_claims_abstentions_as_errors']:.4f} | {flips}/300 ({flips / 300:.2%}) | {t.get('wrong_to_correct', 0)} | {t.get('correct_to_wrong', 0)} |")
    lines += ['', 'Repetition changes about 16% of verdicts and improves aggregate accuracy in both received cohorts. Reversing order changes about 16% of verdicts while barely changing aggregate accuracy. Flips are sensitivity, not automatically errors. Deduplication restores baseline prompts/responses by construction and cache reuse; it does not demonstrate an accuracy improvement or independent neural robustness. Stable wrong answers remain wrong.', '',
        '| Cohort | Perturbation | Macro F1 difference | Marginal grouped 95% interval | Holm p |',
        '|---|---|---:|---|---:|']
    for r in stats:
        lo, hi = r['marginal_group_bootstrap_95_interval']
        lines.append(f"| {r['cohort']} | {r['condition']} | {r['macro_f1_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {r['holm_p_four_macro_f1_comparisons']:.4f} |")
    confirmed = sum(r['holm_p_four_macro_f1_comparisons'] < .05 for r in stats)
    lines += ['', f'{confirmed}/4 macro F1 differences meet the .05 threshold after Holm correction.', '',
        'These exploratory differences are perturbation minus baseline. Entire connected groups of normalized original claims or shared retrieved source IDs are resampled/swapped: 2000 bootstraps, 10000 swaps, seed 369. Holm corrects four macro F1 comparisons; intervals are marginal. Deduplicated controls are excluded from statistical tests because their equality is enforced.', '']
    for c, detail in group_details.items():
        lines.append(f"{c}: {detail['groups']} groups; largest group {detail['sizes_descending'][0]}/300 claims. Shared-source grouping is conservative but incomplete for other benchmark/pretraining dependence. Large connected components reduce the informativeness of resampling and prevent a strong generalization claim.")
    lines += ['', 'Numeric checks apply only to 60 explanation claims per cohort in each condition; they do not establish explanation correctness. New explanation text differences are saved separately. This ungated experiment evaluates exact copies and order on frozen clipped contexts, without fresh retrieval, paraphrase-copy detection, source authentication, multilingual gate validation or calibrated abstention. No manuscript or publication was created. Preserve mixed outcomes without retuning on the observed scores.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json_atomic(out / 'audit_manifest.json', {'source_output_manifest_sha256': sha256(folder / 'output_manifest.json'),
        'verifier_sha256': sha256(Path(__file__)), 'baseline_output_manifest_sha256': identity['provider_output_manifest_sha256'],
        'files': {p.name: sha256(p) for p in out.iterdir() if p.name != 'audit_manifest.json'}})
    print(f'Verified {len(rows)} verdicts, {explanation_count} explanations, {len(shared)} distinct prompt keys.')
    print(json.dumps({'comparisons': stats, 'group_counts': {c: d['groups'] for c, d in group_details.items()}}, indent=2))


if __name__ == '__main__':
    main()
