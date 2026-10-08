"""Descriptive selective-risk ranking for frozen, observed generator outputs.

Gate scores target constructed evidence completeness, not verdict correctness.
No threshold selection or model fit is performed here.
"""
import hashlib
import json
import math
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
POLICIES=('no_gate','nli','embedding','combined')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose_rows(rows):
    lookup={}
    for row in rows:
        key=(row['cohort'],str(row['claim_id']),row['policy'])
        if key in lookup or row['policy'] not in POLICIES:
            raise ValueError('Duplicate or unexpected policy row')
        lookup[key]=row
    cohorts=sorted({k[0] for k in lookup})
    out={}
    for cohort in cohorts:
        ids={claim_id for c,claim_id,p in lookup if c==cohort and p=='no_gate'}
        if not ids:raise ValueError('Missing ungated cohort')
        for claim_id in ids:
            if any((cohort,claim_id,p) not in lookup for p in POLICIES):
                raise ValueError('Missing policy counterpart')
        if sum(c==cohort for c,_,_ in lookup)!=len(ids)*len(POLICIES):
            raise ValueError('Unexpected policy counterpart')
        out[cohort]={}
        for policy in POLICIES[1:]:
            ranked=[]
            for claim_id in ids:
                base=lookup[cohort,claim_id,'no_gate']
                gate=lookup[cohort,claim_id,policy]
                score=gate['gate_probability']
                if base['true_label']!=gate['true_label']:
                    raise ValueError('Truth label mismatch')
                if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
                    raise ValueError('Invalid frozen gate score')
                passed=score>=.5
                if passed:
                    if gate['candidate_label']!=base['candidate_label']:
                        raise ValueError('Accepted outcome differs from ungated output')
                elif gate['candidate_label'] is not None:
                    raise ValueError('Rejected gate returned a label')
                ranked.append((score,claim_id,base['candidate_label']==base['true_label'],passed,
                               gate['candidate_label'] is not None))
            ranked.sort(key=lambda x:(-x[0],x[1]))
            n=len(ranked)
            curve={}
            for fraction in (.2,.4,.6,.8,1.0):
                size=max(1,math.ceil(n*fraction))
                curve[f'top_{round(fraction*100)}_percent']={
                    'count':size,'correct':sum(x[2] for x in ranked[:size]),
                    'accuracy':sum(x[2] for x in ranked[:size])/size,
                    'actual_coverage':size/n}
            correct=0
            risks=[]
            for i,row in enumerate(ranked,1):
                correct+=row[2]
                risks.append(1-correct/i)
            passed=[x for x in ranked if x[3]]
            emitted=[x for x in passed if x[4]]
            out[cohort][policy]={
                'n':n,
                'ungated_correct':sum(x[2] for x in ranked),
                'fixed_threshold':{
                    'threshold':.5,'gate_passed':len(passed),
                    'accepted':len(emitted),
                    'accepted_correct':sum(x[2] for x in emitted),
                    'accepted_accuracy':sum(x[2] for x in emitted)/len(emitted) if emitted else None},
                'ranking':{**curve,'mean_prefix_risk':sum(risks)/n}}
    return out


def main():
    source=ROOT/'artifacts/fresh_pipeline_received'
    manifest=json.loads((source/'output_manifest.json').read_text(encoding='utf-8'))
    if any(digest(source/name)!=sha for name,sha in manifest.items()):
        raise ValueError('Frozen output digest mismatch')
    receipt=ROOT/'artifacts/fresh_pipeline_verification_v1/audit_manifest.json'
    audit=json.loads(receipt.read_text(encoding='utf-8'))
    if audit['source_output_manifest_sha256']!=digest(source/'output_manifest.json'):
        raise ValueError('Verified source receipt mismatch')
    rows=json.loads((source/'predictions.json').read_text(encoding='utf-8'))
    result=diagnose_rows(rows)
    if set(result)!={'scifact','climate_retrieved'} or any(
            any(result[c][p]['n']!=300 for p in POLICIES[1:]) for c in result):
        raise ValueError('Unexpected evaluated cohort')
    folder=ROOT/'artifacts/selective_diagnostic_v1'
    folder.mkdir(exist_ok=True)
    payload={'scope':'Observed development cohorts, descriptive only. Score targets constructed evidence completeness, not verdict correctness.',
             'frozen_threshold':.5,'source_output_manifest_sha256':digest(source/'output_manifest.json'),
             'source_predictions_sha256':digest(source/'predictions.json'),
             'results':result}
    (folder/'diagnostic.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# Frozen gate-score ranking diagnostic','','Six hundred previously observed claims were scored, 300 SciFact and 300 retrieved climate. This is a descriptive replay with no model fitting, threshold tuning or new neural inference.','',
           'Gate probability estimates a constructed completeness target. Ranking it against verdict correctness is a diagnostic, not probability calibration or an independent test of retrieved-evidence sufficiency. A gate passing at the frozen 0.5 threshold can still produce a numeric-guard abstention.','',
           '| Cohort | Gate | Gate passed / 300 | Output accepted / 300 | Correct / accepted | Ungated correct / 300 | Top 20% correct / sampled | Mean prefix risk |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for cohort in ('scifact','climate_retrieved'):
        for policy in POLICIES[1:]:
            x=result[cohort][policy];fix=x['fixed_threshold'];top=x['ranking']['top_20_percent']
            lines.append(f"| {cohort} | {policy} | {fix['gate_passed']} | {fix['accepted']} | {fix['accepted_correct']}/{fix['accepted']} | {x['ungated_correct']} | {top['correct']}/{top['count']} | {x['ranking']['mean_prefix_risk']:.4f} |")
    lines+=['','Top 20% is selected by each frozen gate score for the same cohort, with claim ID breaking exact score ties; it does not set a deployable threshold. Mean prefix risk averages error fractions across all sorted prefixes and depends on this observed sample. Other cumulative points and exact counts are in diagnostic.json.','',
             'The same observed claim labels and generated answers informed earlier development work. Large source-sharing groups, selection history and target mismatch limit inference. These numbers do not validate a confidence threshold, probability calibration, source authentication or improvement on new data. Keep the frozen 0.5 policy unchanged.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (folder/'audit_manifest.json').write_text(json.dumps({
        'source_output_manifest_sha256':digest(source/'output_manifest.json'),
        'source_verification_sha256':digest(receipt),
        'script_sha256':digest(Path(__file__)),
        'files':{name:digest(folder/name) for name in ('diagnostic.json','RESULTS.md')}},indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('Frozen selective diagnostic completed for 600 claims.')


if __name__=='__main__':main()
