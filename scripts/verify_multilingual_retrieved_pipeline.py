"""Audit the received multilingual controller and save scoped results without neural inference."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from apv_rag.fever_nli import LABEL_MAP
from apv_rag.fresh_pipeline import POLICIES
from apv_rag.splits import sha256,write_json_atomic
from run_multilingual_retrieved_pipeline import load_inputs,audit,load,require
from run_fever_calibration import verify_manifest

RECEIVED='artifacts/multilingual_retrieved_pipeline_received_v1/multilingual_retrieved_pipeline_v1'


def diagnostics(rows,gold,prepared,en,ml):
    truth={g['key']:LABEL_MAP[g['label']] for g in gold}
    require(len(truth)==len(gold),'Unique gold keys required')
    files=list(dict.fromkeys(g['file'] for g in gold));details={}
    for name in files:
        targets=[g for g in gold if g['file']==name]
        control=[r for r in rows if r['file']==name and r['policy']=='no_gate']
        require([r['key'] for r in control]==[g['key'] for g in targets],'Diagnostic alignment differs')
        bykey={r['key']:r for r in control};policies={}
        for policy in POLICIES:
            selected=[r for r in rows if r['file']==name and r['policy']==policy]
            require([r['key'] for r in selected]==[g['key'] for g in targets],'Policy alignment differs')
            removed=[]
            for r in selected:
                base=bykey[r['key']]
                if r['candidate_label'] is not None:
                    require(r['candidate_label']==base['candidate_label'],'Gate changed shared accepted answer')
                elif base['candidate_label'] is not None:removed.append(base)
            policies[policy]={'removed_correct_control_answers':sum(r['candidate_label']==truth[r['key']] for r in removed),
                'removed_incorrect_control_answers':sum(r['candidate_label']!=truth[r['key']] for r in removed),
                'removed_control_keys':[r['key'] for r in removed],
                'rejection_reasons':dict(Counter(reason for r in selected for reason in r['reasons']))}
        verifiable=[g for g in targets if g['label']!='NOT ENOUGH INFO']
        hits=sum(g['target_id'] in {d['id'] for d in prepared[g['key']]['retrieved_evidence']} for g in verifiable)
        details[name]={'queries':len(targets),'gold_label_counts':dict(Counter(g['label'] for g in targets)),
            'verifiable_pairs':len(verifiable),'paired_target_excerpt_id_retrieved':hits,
            'paired_target_excerpt_id_hit_at_3':hits/len(verifiable) if verifiable else None,
            'clipped_claims':sum(prepared[g['key']]['clipping']['claim']['original']>64 for g in targets),
            'clipped_passages':sum(c['original']>96 for g in targets for c in prepared[g['key']]['clipping']['passages']),
            'policies':policies}
    return {'queries':len(gold),'policy_records':len(rows),'english_feature_records':len(en),
        'multilingual_feature_records':len(ml),'english_premise_scores':sum(len(v['scores_cen']) for v in en.values()),
        'multilingual_premise_scores':sum(len(v['scores_cen']) for v in ml.values()),
        'correctness_confidence_unavailable_records':sum(r['correctness_confidence'] is None for r in rows),
        'by_file':details,'inference_rerun':False,'significance_test_performed':False,
        'target_hits_mean':'Original target excerpt identifier in clipped model context; not complete semantic evidence recall.'}


def report(summary,checked,resume):
    lines=['# Verified multilingual retrieved-controller results','',
        f"Saved outputs passed non-neural replay for {summary['queries']} queries, {summary['policy_records']} policy records "
        f"and {summary['distinct_responses']} distinct Qwen responses. There are {checked['english_feature_records']} English "
        f"feature records and {checked['multilingual_feature_records']} multilingual feature records, each with "
        f"{checked['english_premise_scores']} passage NLI scores. The Windows save-recovery receipt retains "
        f"{resume['validated_english_feature_records']} previously validated English records.",'',
        'Each row below has 60 queries: 23 support, 18 refute and 19 NEI. Translated variants share the same underlying '
        'sixty English claims. Do not pool them as independent observations. All-query generator accuracy counts abstentions '
        'as errors; direct NLI baselines always predict one of three labels.','',
        '| File | English NLI accuracy | Multilingual NLI accuracy | No-gate all-query accuracy | Combined all-query accuracy | Combined coverage |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in summary['metrics_by_file'].items():
        a=m['direct_baselines'];b=m['policies'];combined=b['combined']['metrics']
        lines.append(f"| {name} | {a['english_nli']['accuracy']:.2%} | {a['multilingual_nli']['accuracy']:.2%} | "
            f"{b['no_gate']['metrics']['accuracy_all_claims_abstentions_as_errors']:.2%} | "
            f"{combined['accuracy_all_claims_abstentions_as_errors']:.2%} | {combined['coverage']:.2%} |")
    lines+=['',
        'Multilingual NLI accuracy is numerically higher on each of the ten translated sets and lower on the English set. '
        'No gate has higher all-query accuracy than the no-gate control on any file. Gate policies select or reject common '
        'generated answers, so lower all-query accuracy alone is not a comparison at equal answer coverage. These descriptive '
        'outcomes do not establish statistical superiority or general gate benefit.','',
        '| File | Combined accepted / 60 | Accepted accuracy | Correct control answers removed | Wrong control answers removed |',
        '|---|---:|---:|---:|---:|']
    for name,m in summary['metrics_by_file'].items():
        c=m['policies']['combined'];d=checked['by_file'][name]['policies']['combined']
        accuracy=c['metrics']['covered_accuracy']
        lines.append(f"| {name} | {c['accepted']}/60 | {accuracy:.2%} | "
            f"{d['removed_correct_control_answers']} | {d['removed_incorrect_control_answers']} |")
    lines+=['','## Verification scope','',
        'File coverage/checksums, pinned sample/model/code receipts, original save-recovery identity, upstream gold bindings, '
        'clipping metadata, context collapse, English feature aggregates, multilingual CEN shapes, exact prompt/SQLite response '
        'bindings, all four controller decisions, direct predictions and all metrics passed. No neural forward pass, independent '
        'tokenizer recount or explanation truth assessment was performed here.','',
        'All 2640 generated-answer correctness confidences remain null. No multilingual calibrator was fitted and no English '
        'FEVER calibrator was reused. Detailed per-class F1, Brier and ECE are preserved in verified_summary.json; rejection '
        'diagnostics and target-excerpt-ID hits are in diagnostics.json.','',
        'This closes the bounded multilingual retrieved-controller experiment. It uses already observed claims and a target-derived '
        'excerpt corpus that contains all targets. Full-page/open-web retrieval, historical publisher authentication, factual '
        'explanation validation and multilingual correctness calibration remain unestablished. No new human annotation, manuscript '
        'or GitHub push was performed.','',*summary['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=RECEIVED)
    parser.add_argument('--output',default='artifacts/multilingual_retrieved_pipeline_audit_v1')
    args=parser.parse_args();folder=(ROOT/args.source).resolve();out=(ROOT/args.output).resolve()
    require(folder.is_relative_to(ROOT/'artifacts') and out.is_relative_to(ROOT/'artifacts') and not out.exists(),
            'Use an existing source and new audit output under artifacts')
    source,manifest,ref,inputs=load_inputs();summary=audit(folder,source,manifest,ref,inputs)
    checked=diagnostics(load(folder/'predictions.json'),load(source/'gold.json'),load(folder/'prepared_contexts.json'),
        load(folder/'feature_cache.json'),load(folder/'multilingual_feature_cache.json'))
    resume=load(folder/'io_resume_receipt.json')
    out.mkdir(parents=True)
    write_json_atomic(out/'verified_summary.json',summary);write_json_atomic(out/'diagnostics.json',checked)
    write_json_atomic(out/'verification.json',{'status':'saved_outputs_passed_non_neural_replay',
        'received_output_manifest_sha256':sha256(folder/'output_manifest.json'),
        'sample_manifest_sha256':sha256(source/'selection_manifest.json'),'gold_sha256':manifest['gold_sha256'],
        'verifier_sha256':sha256(Path(__file__)),'source_io_resume_receipt_sha256':sha256(folder/'io_resume_receipt.json'),
        'queries':summary['queries'],'policy_records':summary['policy_records'],'responses':summary['distinct_responses'],
        'neural_inference_rerun':False,'human_labels_created':False})
    (out/'RESULTS.md').write_text(report(summary,checked,resume),encoding='utf-8')
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    verify_manifest(out)
    print(out/'RESULTS.md')


if __name__=='__main__':main()
