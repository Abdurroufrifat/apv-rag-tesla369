"""Reproduce exploratory grouped correctness calibration from audited multilingual outputs."""
import argparse
import contextlib
import io
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.multilingual_calibration import claim_folds,fit_folds,predict_folds,summarize,require
from apv_rag.fever_pipeline import POLICIES
from apv_rag.splits import sha256,write_json_atomic
from run_multilingual_retrieved_pipeline import load_inputs,audit,load
from run_fever_calibration import verify_manifest
from verify_fresh_pipeline import equal

SOURCE='artifacts/multilingual_retrieved_pipeline_received_v1/multilingual_retrieved_pipeline_v1'
CODE=('scripts/analyze_multilingual_calibration.py','src/apv_rag/multilingual_calibration.py',
      'src/apv_rag/fever_pipeline.py','src/apv_rag/fever_nli.py','src/apv_rag/splits.py',
      'scripts/verify_fresh_pipeline.py')
SETTINGS={'folds':5,'fold_rule':'SHA256 of apv-multilingual-calibration:369:<claim_id>; sorted positions modulo five',
    'score':'raw English NLI probability assigned to the final accepted generated label',
    'target':'upstream benchmark label equals final accepted generated label',
    'fit':'one pooled logistic C=1 per policy and held-out fold; existing single-class/empty fallbacks',
    'control':'Beta(1,1) development-only correctness prevalence using identical folds',
    'ece_bins':15,'ranking_fractions':[0.5,0.8,1.0],'refit_parameter_absolute_tolerance':1e-8,
    'prediction_metric_absolute_tolerance':1e-12,'deployment_refit':False,'independent_confirmation':False}


def inputs():
    source,manifest,ref,queries=load_inputs();folder=ROOT/SOURCE
    # Audit exact saved outputs again, without flooding the short analysis output with replay progress.
    with contextlib.redirect_stdout(io.StringIO()):audit(folder,source,manifest,ref,queries)
    prior=ROOT/'artifacts/multilingual_retrieved_pipeline_audit_v1';verify_manifest(prior)
    verification=load(prior/'verification.json')
    require(verification['received_output_manifest_sha256']==sha256(folder/'output_manifest.json') and
        verification['verifier_sha256']==sha256(ROOT/'scripts/verify_multilingual_retrieved_pipeline.py'),
        'Verified source receipt differs')
    rows=load(folder/'predictions.json');gold=load(source/'gold.json')
    identity={'received_output_manifest_sha256':sha256(folder/'output_manifest.json'),
        'source_audit_output_manifest_sha256':sha256(prior/'output_manifest.json'),
        'gold_sha256':sha256(source/'gold.json'),'settings':SETTINGS,
        'code_sha256':{n:sha256(ROOT/n) for n in CODE},
        'protocol_sha256':sha256(ROOT/'docs/MULTILINGUAL_CORRECTNESS_CALIBRATION.md')}
    return rows,gold,identity


def fits_equal(saved,expected):
    if isinstance(expected,dict):
        return isinstance(saved,dict) and saved.keys()==expected.keys() and all(fits_equal(saved[k],v) for k,v in expected.items())
    if isinstance(expected,list):
        return isinstance(saved,list) and len(saved)==len(expected) and all(fits_equal(a,b) for a,b in zip(saved,expected,strict=True))
    if isinstance(expected,float):return isinstance(saved,(int,float)) and not isinstance(saved,bool) and math.isclose(saved,expected,rel_tol=0,abs_tol=1e-8)
    return type(saved) is type(expected) and saved==expected


def prediction_binding(out):
    return {n:sha256(out/n) for n in ('folds.json','calibrators.json','cross_fitted_predictions.json')}


def report(summary):
    lines=['# Multilingual correctness calibration: grouped exploratory validation','',
        'Five label-blind folds keep all translations of each claim together. Every evaluated confidence comes from a '
        'policy-specific logistic fit excluding that claim and its parallel variants. The source sixty claims were already '
        'observed; this is exploratory validation, not independent confirmation. No neural model was rerun.','',
        '| Policy | Accepted variant rows / 660 | Raw Brier | Cross-fitted Brier | Raw NLL | Cross-fitted NLL | Raw ECE | Cross-fitted ECE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for policy,m in summary['pooled_descriptive'].items():
        if m['status']!='evaluated':continue
        a=m['raw_score_metrics'];b=m['cross_fitted_metrics']
        lines.append(f"| {policy} | {m['accepted_answers']}/660 | {a['binary_brier']:.4f} | {b['binary_brier']:.4f} | "
            f"{a['binary_nll']:.4f} | {b['binary_nll']:.4f} | {a['ece_15_bins']:.4f} | {b['ece_15_bins']:.4f} |")
    lines+=['', '| Policy | Prevalence Brier | Logistic minus prevalence Brier | Prevalence NLL | Logistic minus prevalence NLL | Prevalence ECE | Logistic minus prevalence ECE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for policy,m in summary['pooled_descriptive'].items():
        if m['status']!='evaluated':continue
        c=m['cross_fitted_prevalence_control_metrics'];d=m['calibrated_minus_prevalence']
        lines.append(f"| {policy} | {c['binary_brier']:.4f} | {d['binary_brier']:+.4f} | {c['binary_nll']:.4f} | {d['binary_nll']:+.4f} | {c['ece_15_bins']:.4f} | {d['ece_15_bins']:+.4f} |")
    lines+=['','These pooled losses are descriptive summaries of accepted variant rows. There are sixty underlying claims, '
        'not 660 independent samples. Verdict accuracy and acceptance remain unchanged. All raw/calibrated per-file results '
        'and top-confidence ranking outcomes are preserved in summary.json; do not select a policy using this observed cohort.','',
        '| File | Policy | Accepted / 60 | Brier change (calibrated minus raw) | NLL change | ECE change |',
        '|---|---|---:|---:|---:|---:|']
    for file,policies in summary['per_file'].items():
        for policy,m in policies.items():
            if m['status']!='evaluated':continue
            d=m['calibrated_minus_raw']
            lines.append(f"| {file} | {policy} | {m['accepted_answers']}/60 | {d['binary_brier']:+.4f} | "
                f"{d['binary_nll']:+.4f} | {d['ece_15_bins']:+.4f} |")
    evaluated=[m for policies in summary['per_file'].values() for m in policies.values() if m['status']=='evaluated']
    lines+=['',f"Brier decreases in {sum(m['calibrated_minus_raw']['binary_brier']<0 for m in evaluated)}/{len(evaluated)} "
        f"file/policy cells, NLL in {sum(m['calibrated_minus_raw']['binary_nll']<0 for m in evaluated)}/{len(evaluated)}, "
        f"and ECE in {sum(m['calibrated_minus_raw']['ece_15_bins']<0 for m in evaluated)}/{len(evaluated)}. "
        'These are descriptive counts, without significance testing.', '',
        f"Against the development-prevalence control, logistic Brier decreases in {sum(m['calibrated_minus_prevalence']['binary_brier']<0 for m in evaluated)}/{len(evaluated)} cells, "
        f"and NLL in {sum(m['calibrated_minus_prevalence']['binary_nll']<0 for m in evaluated)}/{len(evaluated)}. "
        f"ECE decreases in {sum(m['calibrated_minus_prevalence']['ece_15_bins']<0 for m in evaluated)}/{len(evaluated)} cells; the pooled logistic ECE is higher than the prevalence control for all four policies. "
        'Retain this simple control when judging whether the score-based logistic model adds value.', '',
        'The fold allocation, twenty fitted models and held-out predictions passed non-neural replay. No original experiment '
        'prediction file was changed. Fits target final accepted-answer correctness; abstentions remain unavailable. No '
        'deployment calibrator is refitted on all sixty claims. Independent new-claim confirmation, historical authentication '
        'and explanation truth remain outside this result. No new human review, manuscript or GitHub push was performed.','',
        *summary['limitations']]
    return '\n'.join(lines)+'\n'


def verify(out,rows,gold,identity):
    verify_manifest(out)
    require(set(load(out/'output_manifest.json'))=={p.name for p in out.iterdir() if p.is_file() and p.name!='output_manifest.json'},
            'Output receipt coverage differs')
    require(load(out/'input_manifest.json')==identity,'Calibration input/code/protocol identity differs')
    folds=claim_folds(rows);require(load(out/'folds.json')==folds,'Label-blind claim grouping differs')
    saved=load(out/'calibrators.json');require(fits_equal(saved,fit_folds(rows,gold,folds)),'Development-fold refit differs')
    verify_manifest(out,'prediction_manifest.json')
    require(load(out/'prediction_manifest.json')==prediction_binding(out),'Prediction receipt coverage differs')
    predicted=predict_folds(rows,folds,saved)
    require(equal(load(out/'cross_fitted_predictions.json'),predicted),'Out-of-fold confidence replay differs')
    summary=summarize(predicted,gold)
    require(equal(load(out/'summary.json'),summary),'Fold-held-out reliability/ranking metrics differ')
    require((out/'RESULTS.md').read_text(encoding='utf-8')==report(summary),'Result report differs')
    require(load(out/'scoring_manifest.json')=={'gold_sha256':identity['gold_sha256'],
        'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),'independent_confirmation':False},'Scoring binding differs')
    runtime=load(out/'producer_runtime.json')
    require(set(runtime)=={'python','numpy','scikit-learn'} and all(isinstance(v,str) for v in runtime.values()),'Producer runtime receipt differs')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--output',default='artifacts/multilingual_correctness_calibration_v1')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    require(out.is_relative_to(ROOT/'artifacts'),'Use output inside artifacts')
    rows,gold,identity=inputs()
    if args.verify:
        verify(out,rows,gold,identity);print('Grouped calibrator fits, held-out confidence and all metrics replayed. No neural inference.')
        return
    require(not out.exists(),'Output exists; use --verify or a new --output path')
    folds=claim_folds(rows);fits=fit_folds(rows,gold,folds)
    out.mkdir(parents=True);write_json_atomic(out/'input_manifest.json',identity)
    write_json_atomic(out/'producer_runtime.json',{'python':sys.version,'numpy':version('numpy'),'scikit-learn':version('scikit-learn')})
    write_json_atomic(out/'folds.json',folds);write_json_atomic(out/'calibrators.json',fits)
    # Freeze fold models before writing and scoring evaluation confidences.
    predictions=predict_folds(rows,folds,fits)
    write_json_atomic(out/'cross_fitted_predictions.json',predictions)
    write_json_atomic(out/'prediction_manifest.json',prediction_binding(out))
    summary=summarize(predictions,gold);write_json_atomic(out/'summary.json',summary)
    write_json_atomic(out/'scoring_manifest.json',{'gold_sha256':identity['gold_sha256'],
        'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),'independent_confirmation':False})
    (out/'RESULTS.md').write_text(report(summary),encoding='utf-8')
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    verify(out,rows,gold,identity)
    print('\n'.join(report(summary).splitlines()[:11]));print(out/'RESULTS.md')


if __name__=='__main__':main()
