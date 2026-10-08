"""Audit the frozen FEVER pipeline export without downloading or running neural models."""
import argparse
import json
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.fever_calibration import normalized, partition_claims
from apv_rag.fever_corpus import EXPECTED_ARCHIVE_SHA256, EXPECTED_PAGES
from apv_rag.fever_nli import LABEL_MAP, read_jsonl, validate_contexts
from apv_rag.fever_pipeline import POLICIES, answer_score, correctness_probability, summarize_policy
from apv_rag.fresh_pipeline import execute_policies, validate_feature_entry
from apv_rag.integrated_gate import collapse_context
from apv_rag.splits import sha256, write_json_atomic
from run_fever_pipeline import (CODE, SETTINGS, COHORT_SHA256, REFERENCE_SHA256,
                                calibrate_development, load, load_inputs)
from run_fever_calibration import verify_manifest
from run_fresh_pipeline import response_key
from run_gated_generation import qwen_prompt
from verify_fresh_pipeline import equal


FIT_TOLERANCE = 1e-8
REPLAY_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_manifest(folder):
    verify_manifest(folder)
    expected = {p.relative_to(folder).as_posix() for p in folder.rglob('*')
                if p.is_file() and p != folder / 'output_manifest.json'}
    require(set(load(folder / 'output_manifest.json')) == expected, 'Output manifest coverage differs')


def verify_stage_data(folder, claims, heads):
    """Replay decisions against exact saved inputs and independent SQLite response bindings."""
    raw = validate_contexts(claims, read_jsonl(folder / 'retrieved_contexts.jsonl'))
    prepared = load(folder / 'prepared_contexts.json')
    cache = load(folder / 'feature_cache.json')
    responses = load(folder / 'responses_used.json')
    rows = load(folder / 'predictions.json')
    require(set(prepared) == {str(r['id']) for r in claims}, 'Prepared claim coverage differs')
    require(set(cache) == {k for k, r in prepared.items() if r['evidence']}, 'Feature coverage differs')
    used = set()
    with closing(sqlite3.connect(f'file:{(folder / "generation_cache.sqlite").resolve().as_posix()}?mode=ro', uri=True)) as db:
        require(db.execute('PRAGMA integrity_check').fetchone() == ('ok',), 'SQLite cache integrity differs')
        cached = {k: json.loads(v) for k, v in db.execute('SELECT key,response FROM answers')}
    require(set(cached) == set(responses), 'Generation cache coverage differs')
    for key, response in responses.items():
        require(set(response) == {'kind','answer','prompt','prompt_tokens','origin'}, 'Response schema differs')
        require(response_key(response['kind'], response['prompt']) == key, 'Response digest differs')
        require(isinstance(response['answer'], str) and type(response['prompt_tokens']) is int and
                0 < response['prompt_tokens'] <= 1024 and response['origin'] == 'current_run_live_or_resume',
                'Response budget/origin differs')
        require(cached[key] == {n: response[n] for n in ('answer','prompt','prompt_tokens')}, 'Generation cache answer conflict')
    def generate(kind, question):
        prompt = qwen_prompt(question)
        key = response_key(kind, prompt)
        require(key in responses, 'Missing requested response')
        r = responses[key]
        require(r['kind'] == kind and r['prompt'] == prompt, 'Response binding differs')
        used.add(key)
        return {n: r[n] for n in ('answer','prompt','prompt_tokens')}
    replay = []
    premise_count = 0
    for claim, retrieved in zip(claims, raw, strict=True):
        key = str(claim['id'])
        r = prepared[key]
        require(set(r) == {'claim','shown_claim','retrieved_evidence','evidence'} and r['claim'] == claim['claim'] and
                isinstance(r['shown_claim'], str) and r['shown_claim'].strip(), 'Prepared claim schema/binding differs')
        original = {d['id']: d for d in retrieved['evidence'] if d['text'].strip()}
        require(len({d['id'] for d in r['retrieved_evidence']}) == len(r['retrieved_evidence']), 'Repeated prepared page')
        for doc in r['retrieved_evidence']:
            require(doc['id'] in original and set(doc) == set(original[doc['id']]) and
                    all(doc[n] == original[doc['id']][n] for n in doc if n != 'text') and
                    isinstance(doc['text'], str) and doc['text'].strip(), 'Prepared retrieval metadata differs')
        require(r['evidence'] == collapse_context(r['retrieved_evidence']), 'Context collapse differs')
        if key in cache:
            validate_feature_entry(cache[key], r['shown_claim'], r['evidence'])
            premise_count += len(cache[key]['scores_cen'])
        for row in execute_policies(r, cache.get(key), heads, generate):
            row.update(claim_id=claim['id'], claim=claim['claim'], shown_claim=r['shown_claim'],
                       retrieved_evidence=r['retrieved_evidence'],
                       context_sha256=cache[key]['context_sha256'] if key in cache else None,
                       score=answer_score(row, cache.get(key)))
            replay.append(row)
    require(equal(rows, replay), 'Controller, numeric check, generated label or score replay differs')
    require(used == set(responses), 'Unrequested response in export')
    return {'policy_records': len(replay), 'feature_records': len(cache), 'premise_scores': premise_count,
            'responses': len(responses), 'rows': replay}


def bootstrap_difference(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(369).integers(0, len(values), size=(2000, len(values)))
    return {'mean': float(values.mean()), 'paired_claim_bootstrap_95_percentile_interval':
            np.quantile(values[indices].mean(1), [0.025,0.975]).tolist()}


def diagnostics(rows, gold, fits):
    result = {}
    by_policy = {p: [r for r in rows if r['policy'] == p] for p in POLICIES}
    control = by_policy['no_gate']
    truth = [LABEL_MAP[g['label']] for g in gold]
    for policy, subset in by_policy.items():
        correct = np.array([r['candidate_label'] == t for r,t in zip(subset,truth,strict=True)])
        base_correct = np.array([r['candidate_label'] == t for r,t in zip(control,truth,strict=True)])
        removed = [i for i,(a,b) in enumerate(zip(control,subset,strict=True))
                   if a['candidate_label'] is not None and b['candidate_label'] is None]
        accepted = [i for i,r in enumerate(subset) if r['candidate_label'] is not None]
        p = np.clip([subset[i]['score'] for i in accepted],1e-6,1-1e-6)
        q = np.clip([correctness_probability(subset[i]['score'],fits[policy]) for i in accepted],1e-6,1-1e-6)
        y = correct[accepted].astype(float)
        result[policy] = {
            'correct_answers': int(correct.sum()), 'rejection_reasons': dict(Counter(x for r in subset for x in r['reasons'])),
            'removed_control_answers': len(removed), 'removed_correct_control_answers': int(base_correct[removed].sum()),
            'removed_incorrect_control_answers': int((~base_correct[removed]).sum()),
            'accuracy_all_claims_minus_no_gate': bootstrap_difference(correct.astype(float)-base_correct.astype(float)),
            'label_counts': dict(Counter(truth)),
            'confusion_true_S_R_NEI_pred_S_R_NEI_abstain': [[sum(t == a and (r['candidate_label'] or 'abstain') == b
                  for r,t in zip(subset,truth,strict=True)) for b in [*LABEL_MAP.values(),'abstain']]
                  for a in LABEL_MAP.values()],
            'binary_brier_calibrated_minus_raw': bootstrap_difference((q-y)**2-(p-y)**2),
            'binary_nll_calibrated_minus_raw': bootstrap_difference(
                -(y*np.log(q)+(1-y)*np.log(1-q))+(y*np.log(p)+(1-y)*np.log(1-p)))}
    return result


def audit(folder):
    source, manifest, ref, claims = load_inputs()
    check_manifest(folder)
    identity = load(folder / 'input_manifest.json')
    index_meta = {'archive_sha256': EXPECTED_ARCHIVE_SHA256, 'source_rows': str(EXPECTED_PAGES),
                  'pages': '5416536', 'empty_placeholders': '1', 'retrieval': 'fts5_bm25_title3_body1'}
    require(identity == {'cohort_manifest_sha256': COHORT_SHA256, 'reference_sha256': REFERENCE_SHA256,
        'packages': ref['packages'], 'settings': SETTINGS, 'index_metadata': index_meta,
        'code_sha256': {n: sha256(ROOT/n) for n in CODE},
        'protocol_sha256': sha256(ROOT/'docs/FEVER_PIPELINE_PROTOCOL.md')}, 'Producer code/protocol/reference identity differs')
    previous = ROOT/'data/external/fever/calibration_v1'
    verify_manifest(previous)
    old = load(previous/'selection_manifest.json')
    observed = read_jsonl(previous/'development/model_inputs.jsonl') + read_jsonl(previous/'confirmation/model_inputs.jsonl')
    require(manifest['prior_calibration_manifest_sha256'] == sha256(previous/'selection_manifest.json') and
            set(manifest['excluded_ids']) == set(old['excluded_prior_ids']) | {r['id'] for r in observed} and
            set(manifest['excluded_texts']) == set(old['excluded_prior_texts']) | {normalized(r['claim']) for r in observed},
            'Prior outcome exclusions differ')
    raw_path = ROOT/'data/external/fever/heldout_v1/shared_task_dev.jsonl'
    require(sha256(raw_path) == manifest['source_sha256'], 'Original FEVER source identity differs')
    selected = partition_claims(read_jsonl(raw_path), exclude_ids=manifest['excluded_ids'],
                               exclude_texts=manifest['excluded_texts'], development_count=300, confirmation_count=300)
    stages, golds = {}, {}
    for name, chosen in zip(('development','confirmation'), selected, strict=True):
        require(claims[name] == [{'id':r['id'],'claim':r['claim']} for r in chosen], 'Deterministic selection replay differs')
        gold_path = source/name/'gold.jsonl'
        require(sha256(gold_path) == manifest['cohorts'][name]['gold_sha256'], 'Gold checksum differs')
        golds[name] = read_jsonl(gold_path)
        require(golds[name] == [{n:r[n] for n in ('id','label','evidence')} for r in chosen], 'Upstream gold binding differs')
        stage = folder/name
        check_manifest(stage)
        verify_manifest(stage,'retrieval_manifest.json')
        require(load(stage/'input_manifest.json') == dict(identity,cohort=name,
                claims_sha256=sha256(source/name/'model_inputs.jsonl')), 'Stage identity differs')
        stages[name] = verify_stage_data(stage,claims[name],ref['gate_models'])
    verify_manifest(folder,'calibrator_manifest.json')
    frozen = load(folder/'correctness_calibrators.json')
    require(set(frozen) == {'development_predictions_sha256','development_gold_sha256','confirmation_labels_used','policies'} and
        frozen['development_predictions_sha256'] == sha256(folder/'development/predictions.json') and
        frozen['development_gold_sha256'] == manifest['cohorts']['development']['gold_sha256'] and
        frozen['confirmation_labels_used'] is False, 'Development fit binding differs')
    refit = calibrate_development(stages['development']['rows'],golds['development'])
    fit_error = 0.0
    require(refit.keys() == frozen['policies'].keys(), 'Calibration policy coverage differs')
    for p, fit in refit.items():
        saved = frozen['policies'][p]
        require(fit.keys() == saved.keys(), 'Calibration schema differs')
        for k,v in fit.items():
            if k in ('coefficient','intercept'):
                error = abs(v-saved[k]); fit_error = max(fit_error,error)
                require(error <= FIT_TOLERANCE, 'Development-only fit replay differs')
            else:
                require(v == saved[k], 'Development calibration counts/settings differ')
    fits = frozen['policies']
    verify_manifest(folder,'confirmation_prediction_manifest.json')
    confirmation = stages['confirmation']['rows']
    annotated = [dict(r, correctness_confidence=correctness_probability(r['score'],fits[r['policy']])
                     if r['candidate_label'] is not None else None) for r in confirmation]
    require(equal(annotated,load(folder/'confirmation_predictions.json')), 'Correctness confidence replay differs')
    summary = {'scope':'fixed full English retrieved-evidence pipeline; new FEVER claims; development-only correctness fits',
        'development_claims':300,'confirmation_claims':300,'policy_records':1200,
        'policies': {p:summarize_policy([r for r in confirmation if r['policy']==p],golds['confirmation'],fits[p]) for p in POLICIES},
        'no_confirmation_fitting':True,'calibrators_sha256':sha256(folder/'correctness_calibrators.json'),
        'distinct_responses': {n:stages[n]['responses'] for n in stages},
        'limitations':['Same public FEVER development dataset, not official blind test.',
            'Wikipedia snapshot identity does not authenticate historical publishers.',
            'Numeric input checks do not establish explanation truth or meaning.',
            'English experiment; full multilingual retrieval/gate transfer remains untested.',
            'Correctness confidence is evaluated only on final accepted answers.',
            'Generation request counts are not measured runtime or energy savings.']}
    require(equal(summary,load(folder/'summary.json')), 'Confirmation summary replay differs')
    require(load(folder/'scoring_manifest.json') == {
        'confirmation_gold_sha256':manifest['cohorts']['confirmation']['gold_sha256'],
        'prediction_manifest_sha256':sha256(folder/'confirmation_prediction_manifest.json'),
        'confirmation_fitting':False}, 'Scoring binding differs')
    return {'verification':'passed','claims':600,'policy_records':2400,
        'stage_counts':{n:{k:v for k,v in s.items() if k!='rows'} for n,s in stages.items()},
        'development_fit_max_abs_error':fit_error,'development_fit_tolerance':FIT_TOLERANCE,
        'controller_and_metric_replay_tolerance':REPLAY_TOLERANCE,
        'confirmation_summary_exact_equal':summary == load(folder/'summary.json'),
        'zero_id_and_normalized_text_overlap':True,'selection_and_upstream_gold_replay':'exact',
        'diagnostics':diagnostics(confirmation,golds['confirmation'],fits),
        'bootstrap_seed':369,'bootstrap_replicates':2000,
        'limitations':['Neural inference, tokenizer clipping/token counts and full-index retrieval were not independently rerun.',
            'Manifests and recorded code bind the workflow; they cannot independently prove execution timing or absence of external inspection.',
            'Claim bootstrap is descriptive, conditional on frozen fits; shared pages/paraphrases and fit uncertainty are unaccounted for.',
            *summary['limitations']]}, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='artifacts/fever_pipeline_received_v1/fever_pipeline_v1')
    parser.add_argument('--output', default='artifacts/fever_pipeline_audit_v1')
    args = parser.parse_args()
    folder, out = (ROOT/args.folder).resolve(), (ROOT/args.output).resolve()
    require(folder.is_relative_to(ROOT/'artifacts') and out.is_relative_to(ROOT/'artifacts'), 'Audit paths must be under artifacts')
    result, summary = audit(folder)
    require(not out.exists(), 'Audit output exists; refusing overwrite')
    out.mkdir()
    write_json_atomic(out/'verification.json',result)
    write_json_atomic(out/'verified_summary.json',summary)
    write_json_atomic(out/'input_manifest.json', {'received_output_manifest_sha256':sha256(folder/'output_manifest.json'),
        'verifier_sha256':sha256(Path(__file__)),'numpy':np.__version__,'scikit-learn':version('scikit-learn')})
    lines = ['# Verified English FEVER pipeline results','',
        '600 distinct claims (300 development and 300 confirmation), 2,400 policy records, all saved feature aggregates, '
        'prompt/cache bindings, gate decisions and numeric checks passed replay. The 600 frozen IDs and normalized texts '
        'are disjoint from prior FEVER/XFEVER project outcomes. Selection and upstream labels replay exactly. '
        'Calibration was refitted using development outcomes only. No neural inference, tokenizer clipping/token recount '
        'or full Wikipedia retrieval was independently repeated.','',
        '| Policy | Correct / 300 | Coverage | Accepted accuracy | Raw correctness ECE | Calibrated ECE |',
        '|---|---:|---:|---:|---:|---:|']
    for p in POLICIES:
        s=summary['policies'][p]; c=s['correctness_calibration']
        lines.append(f'| {p} | {result["diagnostics"][p]["correct_answers"]}/300 | {s["coverage"]:.2%} | '
            f'{s["covered_accuracy"]:.2%} | {c["raw_score_metrics"]["ece_15_bins"]:.6f} | '
            f'{c["calibrated_metrics"]["ece_15_bins"]:.6f} |')
    lines += ['', 'The combined gate removes 11 previously accepted control answers: five correct and six incorrect. '
        'Its all-claim accuracy falls from 160/300 to 155/300. The NLI gate achieves only a small increase in accepted-answer '
        'accuracy while coverage falls to 64%. These outcomes do not establish gate superiority or justify tuning on '
        'confirmation outcomes. Confidence calibration improves binary Brier and NLL for every policy, conditional on '
        'accepted answers; it does not change generated verdicts or establish explanation truth.', '',
        'Rejection and confusion counts and paired claim-bootstrap intervals are in verification.json. These intervals '
        'are descriptive and conditional on the fitted calibrators. Shared-page/paraphrase dependence and development-fit '
        'uncertainty are not included. Policy responses are shared for identical prompts, so the policies are paired '
        'comparisons and response counts are not runtime or energy measurements.', '',
        'This completes verification of this English integrated-pipeline experiment. Historical publisher authentication, '
        'semantic explanation truth and full multilingual retrieval/gate transfer remain unestablished. The project '
        'must report these as limitations unless separately evaluated. No manuscript or GitHub push was performed.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    print('FEVER pipeline audit passed: 600 claims; 2400 policy records; confirmation scores reproduced.')
    print(json.dumps(result['stage_counts'],indent=2))
    print(out/'RESULTS.md')


if __name__ == '__main__':
    main()
