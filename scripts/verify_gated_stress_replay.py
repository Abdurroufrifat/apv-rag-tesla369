"""Independently bind saved gate-policy replays to verified provider data."""
import argparse
from importlib.metadata import version
from pathlib import Path

from apv_rag.splits import sha256, write_json_atomic
from run_fresh_pipeline import ROOT, check_receipt, load, require
from run_gated_stress_replay import prepare, replay, summarize_replay, identity
from verify_fresh_pipeline import equal

REQUIRED = {'input_manifest.json','transformed_features.json','predictions.json','responses.json','summary.json'}


def verify_payloads(actual, expected):
    rows, features, responses = expected
    predicted = actual['predictions']
    key = lambda r: (r['cohort'], str(r['claim_id']), r['condition'], r['policy'])
    keyed = {key(r): r for r in predicted}
    require(len(predicted) == len(keyed) == len(rows), 'Duplicate or missing policy records')
    require(set(keyed) == {key(r) for r in rows}, 'Unexpected policy identity')
    for r in rows:require(equal(keyed[key(r)], r), f'Policy/context/response mismatch: {key(r)}')
    require(equal(actual['features'], features), 'Transformed feature/source mismatch')
    require(equal(actual['responses'], responses), 'Prior response binding mismatch')
    require(equal(actual['summary'], summarize_replay(rows, responses)), 'Summary mismatch')


def verify(folder, root=ROOT):
    hashes = check_receipt(folder)
    require(set(hashes) == REQUIRED, 'Output receipt coverage mismatch')
    inputs = prepare(root)
    received = load(folder,'input_manifest.json')
    expected = identity(inputs, root)
    require(isinstance(received.get('analysis_packages'), dict) and
            set(received['analysis_packages']) == {'numpy','scikit-learn'} and
            all(isinstance(v,str) and v for v in received['analysis_packages'].values()), 'Analysis package schema mismatch')
    expected['analysis_packages'] = received['analysis_packages']
    require(received == expected, 'Provider/code/settings/protocol mismatch')
    expected_rows, features, responses = replay(inputs)
    verify_payloads({'predictions':load(folder,'predictions.json'), 'features':load(folder,'transformed_features.json'),
                     'responses':load(folder,'responses.json'), 'summary':load(folder,'summary.json')},
                    (expected_rows, features, responses))
    summary = summarize_replay(expected_rows, responses)
    return {'status':'posthoc_gate_stress_replay_verified', 'source_output_manifest_sha256':sha256(folder/'output_manifest.json'),
            'policy_records':len(expected_rows), 'transformed_feature_records':len(features),
            'distinct_prior_responses':len(responses), 'new_neural_inference_calls':0,
            'analysis_package_differences':{n:{'recorded':v,'current':version(n)} for n,v in received['analysis_packages'].items() if version(n)!=v},
            'summary':summary,
            'limitations':['Exact-text score copying/reordering is a mathematical replay, not fresh feature inference.',
                           'All generator responses were produced earlier without these gates; request counts are not compute savings.',
                           'All-claim verdict metrics use gate only; gate plus numeric explanation metrics use fixed 60-claim subsets.',
                           'Collapsed equality follows identical contexts/scores/prompts and reused answers; it is not an independent accuracy gain.',
                           'Completeness probability is not correctness confidence; no calibration or statistical superiority claim.',
                           'Observed English cohorts and synthetic copies/order only; full charter remains unfinished.']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',default='artifacts/gated_stress_replay_v1');args=parser.parse_args()
    folder=(ROOT/args.folder).resolve()
    require(folder.is_relative_to((ROOT/'artifacts').resolve()) and folder!=(ROOT/'artifacts').resolve(),'Export must be inside artifacts')
    result=verify(folder)
    out=ROOT/'artifacts/gated_stress_replay_verification_v1';out.mkdir(exist_ok=True)
    write_json_atomic(out/'verification.json',result)
    lines=['# Post-hoc gate copy/order policy results','',
           '9600 policy records and 2400 transformed feature records passed provider, code, context, score, response, gate, numeric and metric replay checks. No new neural inference was performed.','',
           '| Cohort | Policy | Raw-copy gate changes / 300 | Reversal gate changes / 300 | Baseline covered accuracy | Raw-copy covered accuracy |','|---|---|---:|---:|---:|---:|']
    for cohort,conditions in result['summary']['cohorts'].items():
        for policy in ('nli','embedding','combined'):
            base=conditions['baseline'][policy];raw=conditions['copies_raw'][policy];reverse=conditions['reverse_order'][policy]
            fmt=lambda v:'n/a' if v is None else f'{v:.2%}'
            lines.append(f"| {cohort} | {policy} | {raw['gate_acceptance_changes']} | {reverse['gate_acceptance_changes']} | {fmt(base['gate_only_verdict_metrics']['covered_accuracy'])} | {fmt(raw['gate_only_verdict_metrics']['covered_accuracy'])} |")
    lines+=['','These covered-accuracy comparisons change both source presentation and which claims are answered. They are descriptive and do not isolate a causal accuracy benefit. Coverage, all-claim accuracy/F1 and the fixed-subset explanation metrics are preserved separately in verification.json.','']+result['limitations']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json_atomic(out/'audit_manifest.json',{'source_output_manifest_sha256':result['source_output_manifest_sha256'],
        'verifier_sha256':sha256(Path(__file__)), 'files':{p.name:sha256(p) for p in out.iterdir() if p.name!='audit_manifest.json'}})
    print(f"Verified {result['policy_records']} gate-stress policy records; no new model calls.")
    print(out)


if __name__=='__main__':main()
