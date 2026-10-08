"""Reproduce fixed-coverage controls on the independently audited FEVER pipeline."""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.fever_nli import read_jsonl
from apv_rag.fever_selective_analysis import analyze
from apv_rag.splits import sha256,write_json_atomic
from run_fever_calibration import verify_manifest
from run_fever_pipeline import load
from verify_fever_pipeline import audit,require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='artifacts/fever_selective_analysis_v1')
    args=parser.parse_args()
    out=(ROOT/args.output).resolve()
    require(out.is_relative_to(ROOT/'artifacts') and not out.exists(),'Use a new audit output under artifacts')
    source=ROOT/'artifacts/fever_pipeline_received_v1/fever_pipeline_v1'
    audit(source)
    rows=load(source/'confirmation/predictions.json')
    goldpath=ROOT/'data/external/fever/pipeline_v1/confirmation/gold.jsonl'
    result=analyze(rows,read_jsonl(goldpath),load(source/'confirmation/feature_cache.json'))
    out.mkdir()
    write_json_atomic(out/'analysis.json',result)
    write_json_atomic(out/'input_manifest.json',{
        'received_output_manifest_sha256':sha256(source/'output_manifest.json'),
        'confirmation_gold_sha256':sha256(goldpath),
        'analysis_code_sha256':{n:sha256(ROOT/n) for n in ('scripts/analyze_fever_selective.py','src/apv_rag/fever_selective_analysis.py')},
        'protocol_sha256':sha256(ROOT/'docs/FEVER_MATCHED_COVERAGE.md')})
    lines=['# FEVER matched-coverage gate comparison','',
        'This is an exploratory analysis specified after the confirmation results were observed. It reuses the frozen '
        'generator responses and compares each gate with simple rankings of the no-gate accepted answers at the same '
        'answer count. Neither control is fitted on gold labels. No threshold, model or deployed policy is changed.','',
        '| Gate | Answers / 300 | Gate accepted accuracy | NLI-ranked control | Cosine-ranked control |',
        '|---|---:|---:|---:|---:|']
    for policy in ('nli','embedding','combined'):
        a,b=[r for r in result['comparisons'] if r['gate']==policy]
        lines.append(f'| {policy} | {a["accepted_answers"]}/300 | {a["gate_accepted_accuracy"]:.2%} | '
            f'{a["control_accepted_accuracy"]:.2%} | {b["control_accepted_accuracy"]:.2%} |')
    lines+=['',f'The 300 claims form {result["groups"]} connected page/claim groups; the largest contains '
        f'{result["largest_group"]} claims. All six group-based paired comparisons, intervals, adjusted p-values and '
        'selected IDs are recorded in analysis.json. The effect used for tests is all-claim accuracy difference, counting '
        'unanswered claims as errors. At matched answer count, its sign agrees with the difference in accepted-answer accuracy.','',
        '| Gate minus control | All-claim difference | Group bootstrap 95% interval | Holm adjusted p |',
        '|---|---:|---:|---:|']
    for r in result['comparisons']:
        s=r['paired_statistics']; lo,hi=s['group_bootstrap_95_percentile_interval']
        lines.append(f'| {r["gate"]} minus {r["control"]} | {s["all_claim_accuracy_difference"]:+.2%} | '
            f'[{lo:+.2%}, {hi:+.2%}] | {s["holm_adjusted_p_within_this_family"]:.4f} |')
    lines+=['',*result['limitations'],'',
        'This comparison measures answer selection. It does not establish source authentication, factual explanation '
        'truth, calibrated multilingual deployment or journal readiness. Preserve all outcomes and do not use this '
        'observed cohort for policy tuning. No manuscript or GitHub push was performed.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    verify_manifest(out)
    print('\n'.join(lines[:11]))
    print(out/'RESULTS.md')


if __name__=='__main__':main()
