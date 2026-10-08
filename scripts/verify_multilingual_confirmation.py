"""Audit a received frozen multilingual confirmation export and save scoped results."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.splits import sha256,write_json_atomic
from run_multilingual_confirmation import load_inputs,audit,load,require
from run_fever_calibration import verify_manifest
from verify_fresh_pipeline import equal

SOURCE='artifacts/multilingual_confirmation_received_v1/multilingual_confirmation_v1'


def diagnostics(folder,rows,gold):
    en=load(folder/'feature_cache.json');ml=load(folder/'multilingual_feature_cache.json')
    confidence=load(folder/'correctness_predictions.json')
    return {'queries':len(gold),'underlying_claim_ids':len({r['claim_id'] for r in gold}),
        'policy_records':len(load(folder/'predictions.json')),
        'english_feature_records':len(en),'multilingual_feature_records':len(ml),
        'english_passage_scores':sum(len(v['scores_cen']) for v in en.values()),
        'multilingual_passage_scores':sum(len(v['scores_cen']) for v in ml.values()),
        'distinct_responses':len(load(folder/'responses_used.json')),
        'label_counts_per_file':{name:dict(Counter(g['label'] for g in gold if g['file']==name))
            for name in sorted({g['file'] for g in gold})},
        'accepted_with_unavailable_confidence':sum(r['candidate_label'] is not None and
            r['correctness_confidence'] is None for r in confidence),
        'abstentions_with_confidence':sum(r['candidate_label'] is None and
            (r['correctness_confidence'] is not None or r['prevalence_control_confidence'] is not None) for r in confidence),
        'inference_rerun':False,'confirmation_fitting':False,'significance_testing':False}


def report(summary,checked):
    lines=['# Verified new-claim multilingual confirmation','',
        f"Saved outputs passed non-neural replay for {checked['queries']} queries, {checked['policy_records']} policy records "
        f"and {checked['distinct_responses']} distinct model responses. The cohort has {checked['underlying_claim_ids']} "
        'project-held-out claim IDs, each represented in six parallel variants. The four logistic correctness fits and four '
        'prevalence controls were frozen on prior development outcomes; no confirmation labels were used to fit or tune them.','',
        'Each file has 100 queries: 29 SUPPORTS, 27 REFUTES and 44 NOT ENOUGH INFO. The majority-class descriptive accuracy '
        'is therefore 44%. The five non-English files contain upstream machine translations.','',
        '| File | English NLI accuracy | Multilingual NLI accuracy | No-gate all-query accuracy | Combined all-query accuracy | Combined coverage |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in summary['controller_metrics_by_file'].items():
        direct=m['direct_baselines'];policies=m['policies'];combined=policies['combined']
        lines.append(f"| {name} | {direct['english_nli']['accuracy']:.2%} | {direct['multilingual_nli']['accuracy']:.2%} | "
            f"{policies['no_gate']['correct']/100:.2%} | {combined['correct']/100:.2%} | {combined['accepted']/100:.2%} |")
    lines+=['','Multilingual NLI is numerically higher than English NLI on all five translated sets and lower on English. '
        'No gate improves all-query accuracy over the common no-gate outputs in any file. Policies select or reject common '
        'generated answers; this is not an equal-coverage test of gate superiority. No statistical superiority is claimed.','',
        '## Correctness confidence on accepted answers','',
        'The following summaries pool translated rows descriptively. They do not represent 600 independent observations. '
        'Calibration leaves verdicts, acceptance and all-query accuracy unchanged. Abstentions retain null confidence.','',
        '| Policy | Accepted / 600 | Raw Brier | Frozen logistic Brier | Prevalence Brier | Raw NLL | Logistic NLL | Prevalence NLL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for policy,m in summary['correctness_calibration']['pooled_descriptive'].items():
        a=m['raw_score_metrics'];b=m['frozen_calibrator_metrics'];c=m['frozen_prevalence_control_metrics']
        lines.append(f"| {policy} | {m['accepted_answers']}/600 | {a['binary_brier']:.4f} | {b['binary_brier']:.4f} | "
            f"{c['binary_brier']:.4f} | {a['binary_nll']:.4f} | {b['binary_nll']:.4f} | {c['binary_nll']:.4f} |")
    lines+=['','| Policy | Raw ECE | Frozen logistic ECE | Prevalence ECE |','|---|---:|---:|---:|']
    for policy,m in summary['correctness_calibration']['pooled_descriptive'].items():
        a=m['raw_score_metrics'];b=m['frozen_calibrator_metrics'];c=m['frozen_prevalence_control_metrics']
        lines.append(f"| {policy} | {a['ece_15_bins']:.4f} | {b['ece_15_bins']:.4f} | {c['ece_15_bins']:.4f} |")
    cells=[v for values in summary['correctness_calibration']['per_file'].values() for v in values.values()]
    counts=lambda comparison,n:sum(v[comparison][n]<0 for v in cells)
    lines+=['',f"Against raw generated-label NLI scores, frozen logistic Brier, NLL and ECE decrease in "
        f"{counts('calibrated_minus_raw','binary_brier')}/{len(cells)}, {counts('calibrated_minus_raw','binary_nll')}/{len(cells)} "
        f"and {counts('calibrated_minus_raw','ece_15_bins')}/{len(cells)} file/policy cells. Against the fixed prevalence control, "
        f"logistic Brier and NLL decrease in {counts('calibrated_minus_prevalence','binary_brier')}/{len(cells)} and "
        f"{counts('calibrated_minus_prevalence','binary_nll')}/{len(cells)}, while ECE decreases in "
        f"{counts('calibrated_minus_prevalence','ece_15_bins')}/{len(cells)}. The prevalence control has lower pooled ECE "
        'for every policy. Retain this mixed result; ECE alone does not establish that the score-based model adds value.','',
        '| File | Policy | Accepted / 100 | Logistic minus raw Brier | Logistic minus prevalence Brier | Logistic minus prevalence NLL | Logistic minus prevalence ECE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,values in summary['correctness_calibration']['per_file'].items():
        for policy,m in values.items():
            a=m['calibrated_minus_raw'];b=m['calibrated_minus_prevalence']
            lines.append(f"| {name} | {policy} | {m['accepted_answers']}/100 | {a['binary_brier']:+.4f} | "
                f"{b['binary_brier']:+.4f} | {b['binary_nll']:+.4f} | {b['ece_15_bins']:+.4f} |")
    lines+=['','## Verification and scope','',
        f"Both feature caches have {checked['english_feature_records']} context-bound entries and "
        f"{checked['english_passage_scores']} passage scores. File coverage and checksums, frozen sample/code/model identity, "
        'prior-input exclusions, source selection, retrieval, feature shapes and aggregates, prompt/SQLite response equality, '
        'controller decisions, confidence application, abstentions and metric replay passed. Exact development-fit bytes are '
        'preserved. The audit did not rerun neural models, independently recount tokenizer budgets or judge explanation truth.','',
        'This closes the fixed new-claim confirmation experiment within its machine-translated closed excerpt-pool scope. '
        'Do not tune or rerun it on these now observed claims. Full-page/open-web transfer, historical source authentication '
        'and factual explanation verification remain unestablished. The full project charter is unfinished. No new human '
        'annotation, manuscript or GitHub push was performed.','',*summary['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default=SOURCE)
    p.add_argument('--output',default='artifacts/multilingual_confirmation_audit_v1')
    p.add_argument('--verify',action='store_true');args=p.parse_args()
    folder=(ROOT/args.source).resolve();out=(ROOT/args.output).resolve()
    require(folder.is_relative_to(ROOT/'artifacts') and out.is_relative_to(ROOT/'artifacts'),'Use paths inside artifacts')
    manifest,rows,ref,fits=load_inputs();summary=audit(folder,manifest,rows,ref,fits)
    gold=load(ROOT/'data/processed/xfever/confirmation_v1/gold.json');checked=diagnostics(folder,rows,gold)
    receipt={'received_output_manifest_sha256':sha256(folder/'output_manifest.json'),
        'sample_manifest_sha256':sha256(ROOT/'data/processed/xfever/confirmation_v1/selection_manifest.json'),
        'frozen_calibrators_sha256':sha256(ROOT/'data/processed/xfever/confirmation_v1/calibrators.json'),
        'verifier_sha256':sha256(Path(__file__)),
        'verification_scope':'non-neural source, response, controller, confidence and metric replay',
        'inference_rerun':False,'confirmation_fitting':False}
    if args.verify:
        verify_manifest(out)
        require(set(load(out/'output_manifest.json'))=={q.name for q in out.iterdir() if q.is_file() and q.name!='output_manifest.json'},'Audit receipt coverage differs')
        require(equal(load(out/'verified_summary.json'),summary) and load(out/'diagnostics.json')==checked and
            load(out/'verification.json')==receipt and (out/'RESULTS.md').read_text(encoding='utf-8')==report(summary,checked),'Saved audit differs')
    else:
        require(not out.exists(),'Audit exists; use --verify');out.mkdir(parents=True)
        write_json_atomic(out/'verified_summary.json',summary);write_json_atomic(out/'diagnostics.json',checked)
        write_json_atomic(out/'verification.json',receipt);(out/'RESULTS.md').write_text(report(summary,checked),encoding='utf-8')
        write_json_atomic(out/'output_manifest.json',{q.name:sha256(q) for q in out.iterdir() if q.is_file()})
    print('Confirmation export passed: 600 queries, 2400 policy records, 1200 responses. No refit or neural rerun.')
    print(out/'RESULTS.md')


if __name__=='__main__':main()
