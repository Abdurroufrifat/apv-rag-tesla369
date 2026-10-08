"""Verify a fresh-pipeline export without neural models; no accuracy claim from caching."""
import argparse
import math
from pathlib import Path

from apv_rag.fresh_pipeline import POLICIES, execute_policies, validate_feature_entry
from apv_rag.sentence_context import select_sentences
from apv_rag.splits import sha256, write_json_atomic
from run_gated_generation import qwen_prompt
from run_fresh_pipeline import (ROOT, CODE, SETTINGS, prepare_sources, check_context, load, require,
                                check_receipt, response_key, seed_responses, build_summary)

REQUIRED = {'input_manifest.json', 'feature_cache.json', 'execution_progress.json',
            'predictions.json', 'responses.json', 'summary.json'}


def equal(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected, strict=True))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
    return type(actual) is type(expected) and actual == expected


def replay_contexts(inputs):
    """Use prior clipped bytes; independently recompute retrieval and sentence selection."""
    prepared = {}
    for cohort, source in inputs['sources'].items():
        texts = {}
        def remember(text, limit, clipped):
            key = (text, limit)
            require(key not in texts or texts[key] == clipped, 'Conflicting clipped source text')
            texts[key] = clipped
        corpus = {d['doc_id']: d for d in source['retriever'].corpus}
        for claim in source['claims']:
            original = inputs['baselines'][cohort][str(claim['id'])]
            remember(claim['claim'], 64, original['shown_claim'])
            for evidence in original['evidence']:
                selected = select_sentences(claim['claim'], corpus[evidence['id']]['abstract'])
                remember(selected['text'], 96, evidence['text'])
        for claim in source['claims']:
            r = source['retriever'].prepare(claim['claim'], lambda text, limit: texts[(text, limit)])
            check_context(r, inputs['baselines'][cohort][str(claim['id'])])
            prepared[f"{cohort}:{claim['id']}"] = r
    return prepared


def verify_rows(rows, prepared, cache, models, responses, baselines):
    # Prepared keys contain the cohort; no labels are passed into the controller.
    expected_keys = {(key.split(':', 1)[0], key.split(':', 1)[1], policy)
                     for key in prepared for policy in POLICIES}
    keyed = {}
    for row in rows:
        key = (row['cohort'], str(row['claim_id']), row['policy'])
        require(key not in keyed, 'Duplicate policy output')
        keyed[key] = row
    require(set(keyed) == expected_keys, 'Missing or unexpected policy outputs')
    require(set(cache) == {k for k, r in prepared.items() if r['evidence']}, 'Feature cache coverage mismatch')
    seeds = seed_responses(baselines)
    for key, response in responses.items():
        require(set(response) == {'kind', 'answer', 'prompt', 'prompt_tokens', 'origin'}, 'Response schema mismatch')
        require(response_key(response['kind'], response['prompt']) == key, 'Response prompt digest mismatch')
        require(isinstance(response['answer'], str) and type(response['prompt_tokens']) is int and
                0 < response['prompt_tokens'] <= 1024, 'Response token budget/answer invalid')
        if key in seeds:
            require(response['origin'] == 'prior_exact_prompt' and
                    {n: response[n] for n in ('answer', 'prompt', 'prompt_tokens')} == seeds[key],
                    'Prior response provenance mismatch')
        else:
            require(response['origin'] == 'current_run_live_or_resume', 'New response provenance mismatch')
    used = set()
    def generate(kind, question):
        prompt = qwen_prompt(question)
        key = response_key(kind, prompt)
        require(key in responses, 'Missing requested response')
        used.add(key)
        r = responses[key]
        require(r['kind'] == kind and r['prompt'] == prompt, 'Response binding mismatch')
        return {n: r[n] for n in ('answer', 'prompt', 'prompt_tokens')}
    expected_rows = []
    for key, r in prepared.items():
        cohort, claim_id = key.split(':', 1)
        if key in cache:
            validate_feature_entry(cache[key], r['shown_claim'], r['evidence'])
        for result in execute_policies(r, cache.get(key), models, generate):
            result.update(cohort=cohort, claim_id=claim_id, claim=r['claim'], shown_claim=r['shown_claim'],
                          retrieved_evidence=r['retrieved_evidence'],
                          context_sha256=cache[key]['context_sha256'] if key in cache else None,
                          true_label=baselines[cohort][claim_id]['true_label'])
            require(equal(keyed[(cohort, claim_id, result['policy'])], result),
                    f'Controller/text/label binding mismatch: {key}/{result["policy"]}')
            expected_rows.append(result)
    require(used == set(responses), 'Unrequested response in export')
    return expected_rows


def verify_export(folder, root=ROOT):
    hashes = check_receipt(folder)
    require(set(hashes) == REQUIRED, 'Output manifest coverage mismatch')
    inputs = prepare_sources(root)
    identity = load(folder, 'input_manifest.json')
    expected_identity = {'schema_version': 1, 'files': inputs['files'], 'generator': inputs['generator'],
        'feature_models': inputs['feature_models'], 'packages': inputs['packages'], 'policies': list(POLICIES),
        'settings': SETTINGS, 'code_sha256': {n: sha256(root / n) for n in CODE},
        'protocol_sha256': sha256(root / 'docs/FRESH_PIPELINE.md')}
    require(identity == expected_identity, 'Source/model/code/protocol identity mismatch')
    prepared = replay_contexts(inputs)
    cache = load(folder, 'feature_cache.json')
    responses = load(folder, 'responses.json')
    expected_rows = verify_rows(load(folder, 'predictions.json'), prepared, cache, inputs['models'],
                                responses, inputs['baselines'])
    summary = build_summary(expected_rows, cache, load(root / 'artifacts/integrated_gate_received', 'feature_cache.json'),
                            inputs['models'], responses)
    require(equal(load(folder, 'summary.json'), summary), 'Summary replay mismatch')
    progress = load(folder, 'execution_progress.json')
    require(set(progress) == {'new_feature_records_this_invocation', 'total_feature_records', 'retrieval_claims'} and
            type(progress['new_feature_records_this_invocation']) is int and
            0 <= progress['new_feature_records_this_invocation'] <= len(cache) and
            progress['total_feature_records'] == len(cache) and progress['retrieval_claims'] == 600,
            'Execution progress inconsistency')
    return {'status': 'non_neural_export_checks_passed', 'summary': summary,
            'source_output_manifest_sha256': sha256(folder / 'output_manifest.json'),
            'checks': ['600 raw claim/corpus bindings', 'BM25 IDs/scores and sentence selection',
                       'context collapse and feature input digests', '13 feature aggregates and frozen heads',
                       '2400 gate-before-generation decisions', 'exact response provenance and prompts',
                       'numeric checks and all policy metrics'],
            'limitations': ['No neural inference, model bytes, tokenizer recount or SQLite read here.',
                            'Cached generations are reused evidence, not a new quality evaluation.',
                            'Matched-rationale completeness is not general source authentication or explanation correctness.',
                            'Already observed English development cohorts; full project remains incomplete.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='artifacts/fresh_pipeline_v1')
    args = parser.parse_args()
    folder = (ROOT / args.folder).resolve()
    require(folder.is_relative_to((ROOT / 'artifacts').resolve()) and folder != (ROOT / 'artifacts').resolve(),
            'Export must be inside artifacts')
    result = verify_export(folder)
    out = ROOT / 'artifacts/fresh_pipeline_verification_v1'
    out.mkdir(exist_ok=True)
    write_json_atomic(out / 'verification.json', result)
    (out / 'RESULTS.md').write_text(
        '# Fresh pipeline integration verification\n\n'
        '600 claims and 2400 policy records passed non-neural export checks.\n\n'
        f"Distinct used responses: {result['summary']['distinct_responses_used']}; "
        f"prior exact responses reused: {result['summary']['distinct_prior_responses_reused']}; "
        f"current-run live/resumed responses: {result['summary']['distinct_current_run_responses']}.\n\n"
        + '\n'.join(result['limitations']) + '\n', encoding='utf-8')
    write_json_atomic(out / 'audit_manifest.json', {'source_output_manifest_sha256': result['source_output_manifest_sha256'],
        'verifier_sha256': sha256(Path(__file__)), 'files': {p.name: sha256(p) for p in out.iterdir() if p.name != 'audit_manifest.json'}})
    print('Fresh pipeline export verified: 600 claims, 2400 policy records.')
    print(out)


if __name__ == '__main__':
    main()
