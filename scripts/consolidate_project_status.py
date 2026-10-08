"""Compile observed generative results and open charter requirements."""
import json
from pathlib import Path
from apv_rag.splits import sha256, write_json_atomic


def main():
    root=Path(__file__).resolve().parents[1]
    runs=[('SciFact separated v1','separated_rag_scifact_received','exploratory'),
          ('SciFact constrained v2','constrained_rag_received','exploratory'),
          ('SciFact sentence selection v3','sentence_rag_received','exploratory'),
          ('CLIMATE-FEVER frozen retrieval','climate_rag_received','frozen new-domain limited-pool evaluation'),
          ('CLIMATE-FEVER supplied evidence','climate_supplied_received','post-hoc diagnostic')]
    entries=[]
    for name,folder,scope in runs:
        path=root/'artifacts'/folder
        hashes=json.loads((path/'output_manifest.json').read_text(encoding='utf-8'))
        for filename in ('generation_summary.json','predictions.json','input_manifest.json'):
            if sha256(path/filename)!=hashes[filename]:raise ValueError('Source hash mismatch')
        summary=json.loads((path/'generation_summary.json').read_text(encoding='utf-8'))
        rows=json.loads((path/'predictions.json').read_text(encoding='utf-8'))
        assert len(rows)==summary['rows']==300
        entries.append({'run':name,'scope':scope,'source_predictions_sha256':hashes['predictions.json'],
                        'raw':summary['raw_verdict_metrics'],'guarded':summary['structural_guard_metrics']})
    requirements=[
        {'requirement':'Benchmark adapters and split integrity','status':'implemented and checked','remaining':'Preserve frozen source/split receipts; use existing benchmark labels only.'},
        {'requirement':'Classical retrieval and verifier baselines','status':'evaluated','remaining':'Retain mixed and negative outcomes; evaluated baselines do not prove full APV-RAG efficacy.'},
        {'requirement':'Generative baseline','status':'evaluated with limitations','remaining':'SciFact adaptive development; climate weak transfer. Explanations not semantically verified.'},
        {'requirement':'Full integrated provenance-aware generative pipeline','status':'partial; fresh retrieval/feature integration checked','remaining':'900 frozen-context claims and a received 600-claim fresh retrieval/feature integration are checked. Its 2400 policy records show exact prior-feature agreement and 652 early rejections; all 1200 generator responses are reused. Changed-context live generation is now checked on a fixed sixty-claim stress sample. Authentication, calibrated abstention and full planned ablations remain open.'},
        {'requirement':'Source authentication and provenance benefit','status':'partial; optional snapshot controller integrated across four policies','remaining':'All 1800 baseline passages match frozen SciFact and climate snapshots. An opt-in controller checks excerpts before all four policies; cached replay preserves 2400 original outputs and refuses 299/360 stress contexts per policy, including 60 missing-evidence cases. It rejects benign OCR edits and passes one swapped context. Metadata, snapshot matching and family collapse do not authenticate publishers or original webpages. No deduplication accuracy benefit is established. Historical attribution remains unlabeled.'},
        {'requirement':'Actual learned retrieved-evidence sufficiency','status':'partial; constructed coverage target evaluated','remaining':'Training-side complete/missing-rationale targets and parent-group splits are implemented. NLI/embedding/combined heads, matched coverage and live transfer are checked. A descriptive frozen-score ranking diagnostic on 600 observed claims does not calibrate correctness confidence or establish universal retrieved-evidence sufficiency.'},
        {'requirement':'Evidence-linked explanation quality','status':'partial; saved NLI diagnostic verified','remaining':'Saved baseline explanations have no unknown bracket IDs or unseen URLs, but 157/158 SciFact and 195/197 retrieved-climate decisive explanations do not cite a known ID; the prompt did not require one. One stress explanation echoes a fabricated archive reference. The received 840-explanation/2520-pair English NLI diagnostic is verified with zero token-budget skips. These single-passage scores and literal references do not establish explanation truth.'},
        {'requirement':'Multilingual generative evaluation','status':'supplied-evidence experiment and agreement replay checked','remaining':'6600 verdicts across eleven aligned sets and 660 explanations are checked. Eight translated sets are worse than English after grouped correction. A post-hoc Qwen/multilingual-NLI agreement rule is replayed over all 6600 rows; it rejects many correct answers. Multilingual retrieval, gate transfer and confidence calibration are not established.'},
        {'requirement':'Integrated robustness, calibration and component ablations','status':'partial; stress and descriptive multilingual NLI reliability checked','remaining':'2400 earlier verdicts and 480 explanations now support 9600 gate-policy replays. Raw copies change acceptance and order changes verdicts; collapsed equality is enforced. This reuses fixed per-document scores and answers, not live neural stress. The received changed-context export passed checks for 60 claims, 360 contexts and 1440 policy records; it reports 240 new feature computations and includes 480 current-run/resumed responses plus 120 exact prior responses. Synthetic citation/authority controls still change some outputs; missing evidence always abstains. A fixed-bin descriptive audit finds overconfidence in the saved multilingual NLI scores. The later English FEVER controller has development-only final-answer correctness fits and fresh-claim confirmation; multilingual deployment confidence remains uncalibrated.'},
        {'requirement':'Reproducibility and final scope audit','status':'existing Windows audit cleared; clean neural reproduction unverified','remaining':'The received final Windows audit reports 172 tests, nine successful checks, 341 matching references and zero missing, including recovered aliases. Original large arrays remain outside this consolidated package. Clean neural installation, model weights and SQLite reproduction remain unverified.'},
        {'requirement':'Tesla archival stress test','status':'bounded qualitative only','remaining':'Keep machine_candidate or abstain; no new human review and no gold historical verdicts.'},
        {'requirement':'Manuscript and GitHub publication','status':'paused by user','remaining':'No manuscript without permission; no GitHub push before project completion.'}]
    out=root/'artifacts/project_status_consolidated';out.mkdir(exist_ok=True)
    evidence={}
    for dirname in ('integrated_gate_verification','gated_generation_verification',
                    'xfever_generation_verification','generation_robustness_verification','fresh_pipeline_verification_v1','gated_stress_replay_verification_v1','text_stress_verification_v1'):
        folder=root/'artifacts'/dirname
        receipt=json.loads((folder/'audit_manifest.json').read_text(encoding='utf-8'))
        for name,digest in receipt['files'].items():
            if sha256(folder/name)!=digest:raise ValueError(f'Audit output hash mismatch: {dirname}/{name}')
        source_dir={'integrated_gate_verification':'integrated_gate_received',
                    'gated_generation_verification':'gated_generation_received',
                    'xfever_generation_verification':'xfever_generation_received',
                    'generation_robustness_verification':'generation_robustness_received',
                    'fresh_pipeline_verification_v1':'fresh_pipeline_received',
                    'gated_stress_replay_verification_v1':'gated_stress_replay_v1',
                    'text_stress_verification_v1':'text_stress_received'}[dirname]
        if receipt['source_output_manifest_sha256']!=sha256(root/'artifacts'/source_dir/'output_manifest.json'):
            raise ValueError(f'Audit source changed: {source_dir}')
        evidence[dirname]={'audit_manifest_sha256':sha256(folder/'audit_manifest.json'),
                          'source_output_manifest_sha256':receipt['source_output_manifest_sha256']}
    selective=root/'artifacts/selective_diagnostic_v1';selective_receipt=json.loads((selective/'audit_manifest.json').read_text(encoding='utf-8'))
    if selective_receipt['source_output_manifest_sha256']!=sha256(root/'artifacts/fresh_pipeline_received/output_manifest.json'):
        raise ValueError('Selective diagnostic source mismatch')
    for name,digest in selective_receipt['files'].items():
        if sha256(selective/name)!=digest:raise ValueError(f'Selective diagnostic hash mismatch: {name}')
    evidence['selective_diagnostic_v1']={'audit_manifest_sha256':sha256(selective/'audit_manifest.json'),
                                         'source_output_manifest_sha256':selective_receipt['source_output_manifest_sha256']}
    trace=root/'artifacts/source_trace_audit_v1';trace_receipt=json.loads((trace/'audit_manifest.json').read_text(encoding='utf-8'))
    for name,dirname in (('fresh_pipeline_received','fresh_pipeline_received'),('text_stress_received','text_stress_received')):
        if trace_receipt['received_receipts_sha256'][name]!=sha256(root/'artifacts'/dirname/'output_manifest.json'):
            raise ValueError(f'Source trace input changed: {name}')
    for name,digest in trace_receipt['files'].items():
        if sha256(trace/name)!=digest:raise ValueError(f'Source trace hash mismatch: {name}')
    evidence['source_trace_audit_v1']={'audit_manifest_sha256':sha256(trace/'audit_manifest.json'),
                                       'received_receipts_sha256':trace_receipt['received_receipts_sha256']}
    guard=root/'artifacts/snapshot_guard_replay_v1';guard_receipt=json.loads((guard/'audit_manifest.json').read_text(encoding='utf-8'))
    if guard_receipt['source_receipts_sha256']!=trace_receipt['received_receipts_sha256']:
        raise ValueError('Snapshot guard source receipt mismatch')
    for name,digest in guard_receipt['files'].items():
        if sha256(guard/name)!=digest:raise ValueError(f'Snapshot guard hash mismatch: {name}')
    evidence['snapshot_guard_replay_v1']={'audit_manifest_sha256':sha256(guard/'audit_manifest.json'),
                                          'source_receipts_sha256':guard_receipt['source_receipts_sha256']}
    expl=root/'artifacts/explanation_diagnostic_preflight_v1';expl_receipt=json.loads((expl/'audit_manifest.json').read_text(encoding='utf-8'))
    if expl_receipt['preflight_sha256']!=sha256(expl/'preflight.json'):
        raise ValueError('Explanation preflight hash mismatch')
    evidence['explanation_diagnostic_preflight_v1']={'audit_manifest_sha256':sha256(expl/'audit_manifest.json'),
                                                      'preflight_sha256':expl_receipt['preflight_sha256']}
    explanation_received=root/'artifacts/explanation_diagnostic_received'
    explanation_analysis=root/'artifacts/explanation_diagnostic_analysis_v1'
    explanation_receipt=json.loads((explanation_analysis/'audit_manifest.json').read_text(encoding='utf-8'))
    if explanation_receipt['source_output_manifest_sha256']!=sha256(explanation_received/'output_manifest.json'):
        raise ValueError('Explanation analysis source changed')
    for name,digest in explanation_receipt['files'].items():
        if sha256(explanation_analysis/name)!=digest:raise ValueError(f'Explanation analysis mismatch: {name}')
    evidence['explanation_diagnostic_analysis_v1']={
        'audit_manifest_sha256':sha256(explanation_analysis/'audit_manifest.json'),
        'source_output_manifest_sha256':explanation_receipt['source_output_manifest_sha256'],
        'windows_verification_sha256':explanation_receipt['windows_verification_sha256'],
        'local_verification_sha256':explanation_receipt['local_verification_sha256']}
    multilingual=root/'artifacts/multilingual_agreement_replay_v1'
    multilingual_receipt=json.loads((multilingual/'audit_manifest.json').read_text(encoding='utf-8'))
    for name,digest in multilingual_receipt['source_output_manifests_sha256'].items():
        source={'generation':'xfever_generation_received','multilingual_nli':'xfever_multilingual_received'}[name]
        if sha256(root/'artifacts'/source/'output_manifest.json')!=digest:
            raise ValueError(f'Multilingual agreement source changed: {source}')
    if multilingual_receipt['script_sha256']!=sha256(root/'scripts/run_multilingual_agreement_replay.py'):
        raise ValueError('Multilingual agreement script changed')
    for name,digest in multilingual_receipt['files'].items():
        if sha256(multilingual/name)!=digest:raise ValueError(f'Multilingual agreement mismatch: {name}')
    evidence['multilingual_agreement_replay_v1']={'audit_manifest_sha256':sha256(multilingual/'audit_manifest.json'),
        'source_output_manifests_sha256':multilingual_receipt['source_output_manifests_sha256']}
    reliability=root/'artifacts/xfever_reliability_audit_v1'
    reliability_receipt=json.loads((reliability/'audit_manifest.json').read_text(encoding='utf-8'))
    if reliability_receipt['source_output_manifests_sha256']!=multilingual_receipt['source_output_manifests_sha256']:
        raise ValueError('Reliability source identity mismatch')
    if reliability_receipt['agreement_replay_audit_sha256']!=sha256(multilingual/'audit_manifest.json'):
        raise ValueError('Reliability alignment receipt changed')
    if reliability_receipt['script_sha256']!=sha256(root/'scripts/analyze_xfever_reliability.py'):
        raise ValueError('Reliability analysis script changed')
    for name,digest in reliability_receipt['files'].items():
        if sha256(reliability/name)!=digest:raise ValueError(f'Reliability output mismatch: {name}')
    evidence['xfever_reliability_audit_v1']={'audit_manifest_sha256':sha256(reliability/'audit_manifest.json'),
        'source_output_manifests_sha256':reliability_receipt['source_output_manifests_sha256']}
    bound=root/'artifacts/bound_policy_integration_v1'
    bound_receipt=json.loads((bound/'audit_manifest.json').read_text(encoding='utf-8'))
    for dirname,digest in bound_receipt['source_output_manifests_sha256'].items():
        if sha256(root/'artifacts'/dirname/'output_manifest.json')!=digest:
            raise ValueError(f'Bound controller source changed: {dirname}')
    for cohort,relative in (('scifact','data/external/scifact/sealed_v1'),
                            ('climate_retrieved','data/external/climate_fever/frozen_v1')):
        if sha256(root/relative/'corpus.jsonl')!=bound_receipt['corpus_sha256'][cohort]:
            raise ValueError(f'Bound controller corpus changed: {cohort}')
    for name,digest in bound_receipt['code_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError(f'Bound controller code changed: {name}')
    for name,digest in bound_receipt['files'].items():
        if sha256(bound/name)!=digest:raise ValueError(f'Bound controller output mismatch: {name}')
    evidence['bound_policy_integration_v1']={'audit_manifest_sha256':sha256(bound/'audit_manifest.json'),
        'source_output_manifests_sha256':bound_receipt['source_output_manifests_sha256']}
    fever=root/'data/external/fever/heldout_v1'
    fever_receipt=json.loads((fever/'output_manifest.json').read_text(encoding='utf-8'))
    for name,digest in fever_receipt.items():
        if name not in ('shared_task_dev.jsonl','selection_manifest.json','model_inputs.jsonl','gold.jsonl'):
            raise ValueError(f'Unexpected FEVER input file: {name}')
        if sha256(fever/name)!=digest:raise ValueError(f'Frozen FEVER input mismatch: {name}')
    if len(fever_receipt)!=4:raise ValueError('Incomplete frozen FEVER input')
    fever_meta=json.loads((fever/'selection_manifest.json').read_text(encoding='utf-8'))
    if fever_meta['prior_xfever_output_manifest_sha256']!=sha256(root/'artifacts/xfever_generation_received/output_manifest.json'):
        raise ValueError('FEVER exclusion source changed')
    evidence['fever_heldout_v1']={'output_manifest_sha256':sha256(fever/'output_manifest.json'),
        'prior_xfever_output_manifest_sha256':fever_meta['prior_xfever_output_manifest_sha256']}
    evidence['semantic_sufficiency']={'verified_metrics_sha256':sha256(root/'artifacts/semantic_sufficiency_verification/verified_metrics.json'),
                                     'source_manifest_sha256':sha256(root/'artifacts/semantic_sufficiency_received/output_manifest.json')}
    # Add verified FEVER outcomes without changing older experiments or their receipts.
    from run_fever_calibration import verify_manifest
    from run_fever_pipeline import load
    from verify_fever_pipeline import audit
    pipeline_source=root/'artifacts/fever_pipeline_received_v1/fever_pipeline_v1'
    _,pipeline_summary=audit(pipeline_source)
    fever_results={}
    for dirname,source_dir in (
        ('fever_calibration_audit_v1','fever_calibration_received_v1/fever_calibration_v1'),
        ('fever_pipeline_audit_v1','fever_pipeline_received_v1/fever_pipeline_v1'),
        ('fever_selective_analysis_v1','fever_pipeline_received_v1/fever_pipeline_v1')):
        folder=root/'artifacts'/dirname
        verify_manifest(folder)
        binding=load(folder/'input_manifest.json')
        received_sha=sha256(root/'artifacts'/source_dir/'output_manifest.json')
        if binding['received_output_manifest_sha256']!=received_sha:
            raise ValueError(f'FEVER audit source changed: {dirname}')
        evidence[dirname]={'output_manifest_sha256':sha256(folder/'output_manifest.json'),
                          'received_output_manifest_sha256':received_sha}
    baseline=root/'artifacts/fever_audit_v1'
    verify_manifest(baseline)
    original_source=root/'artifacts/fever_received_v1'
    for name,digest in load(baseline/'input_manifest.json')['inputs'].items():
        if sha256(original_source/name)!=digest:raise ValueError(f'FEVER baseline source changed: {name}')
    evidence['fever_audit_v1']={'output_manifest_sha256':sha256(baseline/'output_manifest.json')}
    fever_results['original_nli']=load(baseline/'audit_summary.json')
    fever_results['nli_calibration_confirmation']=load(root/'artifacts/fever_calibration_audit_v1/verified_summary.json')
    fever_results['integrated_pipeline_confirmation']=pipeline_summary
    fever_results['matched_coverage_analysis']=load(root/'artifacts/fever_selective_analysis_v1/analysis.json')
    write_json_atomic(out/'fever_results.json',fever_results)
    updates={
        'Full integrated provenance-aware generative pipeline':(
            'English retrieval/features/generation/gates evaluated; original full scope partial',
            'The complete frozen English FEVER controller is evaluated on 300 new development and 300 new confirmation claims, after exclusions of all prior FEVER/XFEVER IDs and texts. All 2400 policy records, 600 feature records, 1789 premise scores and 1200 distinct current-run/resumed responses passed replay. Confirmation accuracy is 53.33% no-gate and 51.67% combined. This does not establish historical authentication, open-web efficacy or all planned ablations.'),
        'Actual learned retrieved-evidence sufficiency':(
            'Constructed target evaluated; fresh FEVER matched-coverage benefit unsupported',
            'At 192 accepted FEVER answers, the NLI gate has 56.25% accuracy versus 63.02% for generated-label NLI ranking. No learned gate advantage is established by the six group-based comparisons after Holm correction. Heads and threshold stay frozen. Annotation-completeness targets do not establish universal sufficiency.'),
        'Integrated robustness, calibration and component ablations':(
            'English correctness calibration and three gate-head ablations evaluated; wider scope partial',
            'The original stress results are preserved. English FEVER development-only fits now assess final accepted-answer correctness on 300 new confirmation claims; every policy improves binary Brier/NLL versus its raw NLI label score. These fits do not change verdicts or calibrate multilingual confidence. Source/date/family and trained-robustness efficacy remain unestablished.'),
        'Reproducibility and final scope audit':(
            'Saved English FEVER caches/decisions/scores audited; clean neural reproduction unverified',
            'The historical Windows audit is preserved. FEVER pipeline verification checks every exported SQLite response against exact prompts, feature/context bindings, upstream labels and full scores. Full model inference, tokenizer clipping/token counts and Wikipedia retrieval were not independently rerun. Model weights and some earlier arrays remain outside the package.')}
    for r in requirements:
        if r['requirement'] in updates:r['status'],r['remaining']=updates[r['requirement']]
    pool_folder=root/'artifacts/xfever_retrieval_pool_v1'
    from run_xfever_retrieval import prepare as prepare_pool,verify as verify_pool
    pool_sets,pool_identity,pools,pool_queries=prepare_pool()
    pool_summary=verify_pool(pool_folder,pool_sets,pool_identity,pools,pool_queries)
    evidence['xfever_retrieval_pool_v1']={'output_manifest_sha256':sha256(pool_folder/'output_manifest.json'),
                                       'dataset_manifest_sha256':pool_identity['dataset_manifest_sha256']}
    for r in requirements:
        if r['requirement']=='Multilingual generative evaluation':
            r['status']='Supplied-evidence generation and closed excerpt-pool retrieval evaluated; full transfer partial'
            r['remaining']='The 6600 supplied-evidence verdicts and 660 explanations remain preserved. A separate claim-only retrieval diagnostic now evaluates 6600 aligned queries per analyzer across eleven files, representing 585 English claim IDs. Pools contain 538-571 target-derived excerpts; every target is present by construction. CJK character tokens improve target hit rates in Chinese/Japanese, but this does not establish full-page/open-web retrieval, end-to-end multilingual gates, factual explanations or confidence calibration. No new neural inference or human review was performed.'
    from run_multilingual_retrieved_pipeline import load_inputs as prepare_transfer,identity as transfer_identity
    from run_fever_calibration import verify_manifest
    _,_,transfer_ref,transfer_rows=prepare_transfer()
    transfer_source=root/'artifacts/multilingual_retrieved_pipeline_received_v1/multilingual_retrieved_pipeline_v1'
    transfer_audit=root/'artifacts/multilingual_retrieved_pipeline_audit_v1'
    verify_manifest(transfer_source);verify_manifest(transfer_audit)
    transfer_receipt=json.loads((transfer_audit/'verification.json').read_text(encoding='utf-8'))
    transfer_summary=json.loads((transfer_audit/'verified_summary.json').read_text(encoding='utf-8'))
    if (transfer_receipt['received_output_manifest_sha256']!=sha256(transfer_source/'output_manifest.json') or
            transfer_receipt['verifier_sha256']!=sha256(root/'scripts/verify_multilingual_retrieved_pipeline.py') or
            transfer_receipt['status']!='saved_outputs_passed_non_neural_replay' or
            json.loads((transfer_source/'input_manifest.json').read_text(encoding='utf-8'))!=transfer_identity(transfer_ref) or
            transfer_summary!=json.loads((transfer_source/'summary.json').read_text(encoding='utf-8'))):
        raise ValueError('Verified multilingual controller bindings differ')
    evidence['multilingual_retrieved_pipeline_audit_v1']={
        'output_manifest_sha256':sha256(transfer_audit/'output_manifest.json'),
        'received_output_manifest_sha256':transfer_receipt['received_output_manifest_sha256']}
    write_json_atomic(out/'multilingual_retrieved_pipeline_results.json',transfer_summary)
    pending_path=out/'multilingual_retrieved_pipeline_pending.json'
    if pending_path.exists():pending_path.unlink()
    for r in requirements:
        if r['requirement']=='Multilingual generative evaluation':
            r['status']='Supplied-evidence generation and closed-pool retrieved-controller transfer evaluated'
            r['remaining']='The supplied-evidence and claim-only excerpt-pool stages remain preserved. The received sixty-claim/eleven-variant controller run is now verified for 660 queries, 2640 policy decisions, 660 English and 660 multilingual feature records, 1978 premise scores per NLI model and 1316 distinct responses. Multilingual NLI is numerically better on ten translated sets but worse on English. English-trained gates select common generator answers and show no all-query accuracy gain. These are observed claims in target-derived pools, not independent full-page/open-web confirmation. Historical source authentication, explanation truth and multilingual correctness calibration remain open; correctness confidence is unavailable.'
    from analyze_multilingual_calibration import inputs as calibration_inputs,verify as verify_calibration
    calibration_rows,calibration_gold,calibration_identity=calibration_inputs()
    calibration_folder=root/'artifacts/multilingual_correctness_calibration_v1'
    calibration_summary=verify_calibration(calibration_folder,calibration_rows,calibration_gold,calibration_identity)
    evidence['multilingual_correctness_calibration_v1']={
        'output_manifest_sha256':sha256(calibration_folder/'output_manifest.json'),
        'received_output_manifest_sha256':calibration_identity['received_output_manifest_sha256']}
    write_json_atomic(out/'multilingual_calibration_results.json',calibration_summary)
    for r in requirements:
        if r['requirement']=='Multilingual generative evaluation':
            r['remaining']+=' A separate five-fold correctness-confidence analysis now evaluates all 2640 policy rows with translations grouped by sixty claim IDs. Twenty logistic fits exclude each evaluated claim and its variants. Raw-score Brier/NLL/ECE decrease in all 44 file/policy cells; the prevalence control has better ECE in pooled results. This is exploratory observed-data validation, not independent calibrated deployment confirmation.'
        elif r['requirement']=='Integrated robustness, calibration and component ablations':
            r['status']='English confirmation and grouped multilingual correctness calibration evaluated with scope limits'
            r['remaining']+=' The grouped multilingual confidence analysis is now checked and should not be repeated. It preserves verdicts, gate acceptance and abstentions. Compared with a Beta(1,1) development-prevalence control, logistic Brier/NLL improve in 33 of 44 file/policy cells, while ECE improves in only 6. Independent new-claim calibration confirmation and remaining charter ablations are still unestablished.'
    from prepare_multilingual_confirmation import verify as verify_confirmation_preparation
    confirmation_manifest,confirmation_inputs=verify_confirmation_preparation()
    confirmation_ready={'status':'prepared; confirmation neural inference NOT RUN','claims':100,'queries':len(confirmation_inputs),
        'selection_manifest_sha256':sha256(root/'data/processed/xfever/confirmation_v1/selection_manifest.json'),
        'frozen_calibrators_sha256':confirmation_manifest['calibrators_sha256'],
        'development_claims':confirmation_manifest['development_claims'],'pool_sizes':confirmation_manifest['pool_sizes'],
        'command':'.\\.venv\\Scripts\\python.exe scripts\\run_multilingual_confirmation.py',
        'result_zip':'multilingual_confirmation_outputs.zip'}
    confirmation_audit=root/'artifacts/multilingual_confirmation_audit_v1'
    confirmation_summary=None
    if (confirmation_audit/'output_manifest.json').exists():
        from run_fever_calibration import verify_manifest
        from run_multilingual_confirmation import load_inputs as confirmation_load,audit as confirmation_replay
        from verify_fresh_pipeline import equal
        verify_manifest(confirmation_audit)
        received=root/'artifacts/multilingual_confirmation_received_v1/multilingual_confirmation_v1'
        receipt=json.loads((confirmation_audit/'verification.json').read_text(encoding='utf-8'))
        if (receipt['received_output_manifest_sha256']!=sha256(received/'output_manifest.json') or
            receipt['verifier_sha256']!=sha256(root/'scripts/verify_multilingual_confirmation.py') or
            receipt['sample_manifest_sha256']!=confirmation_ready['selection_manifest_sha256'] or
            receipt['frozen_calibrators_sha256']!=confirmation_ready['frozen_calibrators_sha256']):
            raise ValueError('New-claim confirmation audit binding differs')
        manifest,queries,reference,fits=confirmation_load()
        confirmation_summary=confirmation_replay(received,manifest,queries,reference,fits)
        if not equal(confirmation_summary,json.loads((confirmation_audit/'verified_summary.json').read_text())):
            raise ValueError('New-claim confirmation metric replay differs')
        pending=out/'multilingual_confirmation_pending.json'
        if pending.exists():pending.unlink()
        write_json_atomic(out/'multilingual_confirmation_results.json',confirmation_summary)
        evidence['multilingual_confirmation_v1']={'status':'received and verified; fixed machine-translated closed-pool confirmation complete',
            'audit_output_manifest_sha256':sha256(confirmation_audit/'output_manifest.json'),**receipt}
        for r in requirements:
            if r['requirement'] in ('Multilingual generative evaluation','Integrated robustness, calibration and component ablations'):
                r['status']+='; fixed new-claim multilingual confidence confirmation verified'
                if r['requirement']=='Multilingual generative evaluation':
                    r['remaining']='Supplied-evidence generation, retrieved-controller transfer, grouped calibration and fixed 100-ID/600-query confidence confirmation are now verified within excerpt pools. Full-page/open-web multilingual retrieval, human-translated new-claim confirmation and general deployment transfer remain unestablished. Prior-data exclusions cannot establish absence of pretrained-model benchmark exposure. Authentication and explanation truth remain separate open requirements.'
                else:
                    r['remaining']='English held-out confidence calibration and fixed multilingual new-claim confirmation are verified within their reported benchmark scopes. Remaining source/date/family/trained-robustness ablations and full-page/open-web transfer remain unestablished. Logistic Brier/NLL beat raw scores in all 24 multilingual cells and prevalence controls in 21; prevalence has lower ECE in all 24. No broad gate superiority is established. Do not rerun or tune on these observed outcomes.'
    else:
        write_json_atomic(out/'multilingual_confirmation_pending.json',confirmation_ready)
        evidence['multilingual_confirmation_v1_preparation']=confirmation_ready
    display_summary=None
    display_audit=root/'artifacts/evidence_display_audit_v1'
    if (display_audit/'audit_manifest.json').exists():
        from audit_evidence_display import verify as verify_display
        display_summary=verify_display(display_audit)
        evidence['evidence_display_v1']={'audit_manifest_sha256':sha256(display_audit/'audit_manifest.json'),
            'status':'exact cited-snapshot evidence displays verified',
            'records':display_summary['records'],
            'support_relationships_verified':0}
        write_json_atomic(out/'evidence_display_results.json',display_summary)
        for r in requirements:
            if r['requirement']=='Evidence-linked explanation quality':
                r['status']+='; exact cited-excerpt display layer verified'
                r['remaining']+=' A separate display layer now replays all 3840 original/stress policy records, with 5637 exact cited excerpts. It excludes generated rationale text and withholds candidates on mismatched contexts; original outputs and results remain unchanged. These evidence packets fix literal citation traceability but do not certify the verdict-evidence relationship, explanation truth, relevance or completeness. The earlier NLI proxy has not selected a truth threshold.'
    tradeoff_summary=None
    if (root/'artifacts/guard_tradeoffs_v1/audit_manifest.json').exists():
        from analyze_guard_tradeoffs import run as verify_tradeoffs
        tradeoff_summary=verify_tradeoffs(verify=True)
        write_json_atomic(out/'guard_tradeoff_results.json',tradeoff_summary)
        evidence['guard_tradeoffs_v1']={'audit_manifest_sha256':sha256(root/'artifacts/guard_tradeoffs_v1/audit_manifest.json'),
            'status':'saved-output correctness/coverage trade-offs verified; no general robustness benefit established'}
        for r in requirements:
            if r['requirement']=='Integrated robustness, calibration and component ablations':
                r['status']+='; cited-evidence admission trade-offs measured'
                r['remaining']+=' Saved-output analysis preserves original candidates but withholds both correct and incorrect altered-context answers. The no-gate OCR control loses 28 correct and 32 incorrect answers among 60 cases. Strict source matching does not establish broad robustness or accuracy benefit.'
    origin_summary=None
    origin=root/'artifacts/source_origin_audit_v1'
    if (origin/'audit_manifest.json').exists():
        from audit_source_origin import verify as verify_origin
        origin_summary=verify_origin(origin)
        evidence['source_origin_v1']={'audit_manifest_sha256':sha256(origin/'audit_manifest.json'),
            'status':'fixed institutional metadata/navigation origin binding verified',
            'captured_sources':origin_summary['captured_sources'],
            'historical_attribution_authenticated':False}
        write_json_atomic(out/'source_origin_results.json',origin_summary)
        for r in requirements:
            if r['requirement']=='Source authentication and provenance benefit':
                r['status']+='; fixed institutional HTTPS origin/capture guard verified'
                r['remaining']+=' Three fixed institutional metadata/navigation pages now have trusted-collector HTTPS captures and exact-byte replay; eight altered/missing-source controls per captured page are refused before model scoring. Unsigned receipts are not third-party TLS attestations. Newspaper images, embedded content, quotation attribution, author/date truth and publisher honesty remain unverified; this does not establish provenance accuracy benefit.'
    from apv_rag.current_stage_evidence import current_evidence
    latest=current_evidence(root)
    write_json_atomic(out/'latest_stage_results.json',latest)
    evidence['latest_stage_evidence']={
        'english_clean_confirmation':'received 300-claim confirmation reproduction verified',
        'multilingual_clean_confirmation':'received 600-query confirmation reproduction verified',
        'capture_timing_ablation':'negative exploratory result; original classifier retained',
        'multilingual_pool_mismatch':'600 controlled swaps withheld; exact pool identity only'}
    for r in requirements:
        if r['requirement']=='Reproducibility and final scope audit':
            r['status']='scoped English and multilingual confirmation reproduction exports verified'
            r['remaining']='Received Windows clean English 300-claim and multilingual 600-query reruns match their frozen outputs with zero reported differences. Export hashes and original verification receipts are rechecked. This uses trusted-collector execution logs; neural inference is not repeated here. Development training, source-index reconstruction, earlier experiments and third-party execution attestation remain unverified.'
        if r['requirement']=='Integrated robustness, calibration and component ablations':
            r['remaining']+=' The capture-timing augmentation lowered observed internal-validation macro-F1 from 0.5193 to 0.4796 and was not adopted. Archive capture dates do not establish publication dates.'
        if r['requirement']=='Multilingual generative evaluation':
            r['remaining']+=' An optional pinned-pool guard binds 600 saved original contexts and withholds 600 same-claim foreign-language-pool substitutions before generation. This checks exact source-pool identity and does not detect language semantically or measure model accuracy on mismatches.'
    write_json_atomic(out/'multilingual_retrieval_results.json',pool_summary)
    write_json_atomic(out/'later_stage_evidence.json',evidence)
    write_json_atomic(out/'generative_results.json',entries)
    write_json_atomic(out/'requirements.json',requirements)
    lines=['# Project status and completion requirements', '', 'Status: unfinished against the project charter. This is a technical project audit, not a manuscript. No new human annotation is required; existing benchmark labels remain the quantitative basis.', '', '## Observed generative results', '', '| Run | Raw accuracy | Guarded accuracy | Guarded coverage | Guarded macro F1 |','|---|---:|---:|---:|---:|']
    for r in entries:
        raw,guard=r['raw'],r['guarded']
        lines.append(f"| {r['run']} | {raw['accuracy_all_claims_abstentions_as_errors']:.2%} | {guard['accuracy_all_claims_abstentions_as_errors']:.2%} | {guard['coverage']:.2%} | {guard['macro_f1_all_claims_abstentions_as_errors']:.4f} |")
    lines+=['', 'Numeric extractor v2 SciFact replay:54.33% accuracy,99.67% coverage,macroF1 .4991. This is post-hoc and not a new frozen model run. Decoder likelihood diagnostics underperformed greedy decoding in both climate contexts.', '', 'Climate majority-class descriptive accuracy46.67%; frozen raw38% and guarded36% are lower. No strong transfer or full-system improvement claim is supported. Stop incremental decoder tuning on these observed cohorts.', '', '## Remaining requirements', '', '| Requirement | Status | Remaining work |','|---|---|---|']
    for r in requirements:lines.append(f"| {r['requirement']} | {r['status']} | {r['remaining']} |")
    lines+=['', '## Later work now checked', '',
            'The previous checklist predates the completed constructed-sufficiency, matched-coverage, live-controller, multilingual-generation and copy/order experiments. These stages are now reflected above; they do not need to be repeated simply because the old checklist marked them open. Later-stage evidence hashes are recorded in later_stage_evidence.json.', '',
            'Synthetic raw repetition changes 47/300 SciFact and 48/300 climate verdicts. Reversing order changes 47/300 and 49/300. Raw repetition accuracy rises from 54.67% to 59.67% and from 38.00% to 40.00%. No macro F1 difference is confirmed after four-comparison Holm correction. Collapsed-control equality is enforced by identical prompts and reused baseline answers. It is not evidence of an accuracy benefit. Climate grouping has 28 components, including one with 262 claims, which limits inference.', '',
            'The live controller checks 3600 policy records and 934 early rejections without generation requests. The NLI gate raises selective accuracy at low coverage in two retrieved cohorts but is worse on supplied climate evidence; embedding/combined heads do not demonstrate a general transfer gain. Multilingual generation scores supplied evidence, not a full multilingual RAG system. Retain all negative and mixed results.', '',
            '## Concrete next work', '',
            'The existing Windows audit is cleared in artifacts/reproducibility_audit_windows_final_verification. The reproducibility command and current inventory are implemented in scripts/audit_project_reproducibility.py and artifacts/reproducibility_audit_v1. The Windows setup installs the editable package, and current instructions use the existing project folder. The optional sentence-transformers version range now permits the recorded feature-run version. Inspect audit logs for actual results; this does not establish clean neural reproduction.', '',
            'The received fresh retrieval/feature integration passed checks for 600 claims and 2400 policy records. Its progress record reports 600 new feature computations; exported features and probabilities exactly match the earlier cache, with zero threshold changes. All 1200 distinct generator responses are reused and 652 policies reject before generation. The received Windows verification and local replay agree. This stage is closed; do not rerun it. A subsequent post-hoc gate copy/order replay is now verified for 9600 policy records with 2400 transformed feature entries and 2160 prior responses. Eight new focused tests and the full 193-test local suite pass. Raw copies change acceptance in 3–17/300 cases per head/cohort; reversal preserves acceptance while some verdicts change. The fixed sixty-claim changed-context batch is now received and verified: 1440 policy records, 240 reported fresh-feature contexts, 480 current-run/resumed responses and 120 exact prior responses. Windows and local verification agree. The missing-evidence condition abstains for all policies without requests. Citation/authority changes still affect some outputs, including one distinct SciFact explanation that repeats a fabricated reference. This stress stage is closed; do not rerun it. Address authentication, factual explanations, multilingual integration and calibration next. Exact-copy score transforms and prior answers must not be presented as new neural inference or independent accuracy gains. Keep benchmark labels out of fitting/tuning and preserve all mixed outcomes. The Windows audit/recovery stage is also closed.', '',
            'The frozen-score diagnostic in artifacts/selective_diagnostic_v1 ranks the original ungated answers for 300 SciFact and 300 retrieved climate claims. At the fixed 0.5 threshold the climate NLI gate passes 52/300, while numeric checks reduce final accepted answers to 49/300. Scores target constructed evidence completeness, not correctness probability, so this earlier analysis is descriptive; later English FEVER correctness calibration is now verified, while multilingual calibration remains open.', '',
            'The source trace audit checks every saved passage against the frozen corpus snapshot and claim-specific sentence indices. All 1800 baseline passages match. On the ungated explanation rows, 157/158 SciFact and 195/197 retrieved-climate decisive outputs lack a known bracketed source ID; prompts did not require explicit citation. One fabricated archive reference is repeated under synthetic stress. An offline exact-snapshot pre-generation rule passes all 600 original contexts, rejects all modified SciFact contexts and 119/120 modified climate contexts; the remaining donor context shares the selected source excerpts with the target claim. This is synthetic replay and rejects legitimate OCR edits, leaving publisher authentication and explanation truth open.', '',
            'A separate executable source-snapshot controller now checks source IDs, claim-specific sentence selection and excerpt prefixes before asking for a verdict or explanation. Cached replay leaves all 600 original outputs unchanged, rejects 239/240 modified stress contexts, and prevents the one fabricated-reference echo. It is an optional controller with no live neural timing or new answers; it refuses benign OCR differences and passes one swapped donor context. It is not publisher authentication.', '',
            'The received explanation diagnostic passes local replay of all 840 saved English explanations and 2520 passage pairs with zero token-budget skips. The Windows and local verification contents agree after CRLF normalization, each with its own matching hash receipt. On the original 300 per cohort, mean maximum single-passage entailment is .2701 SciFact and .3306 retrieved climate; this score is not factual explanation accuracy. Literal quote checks find 202 spans: 58 in evidence, 87 only in claims, 57 unmatched under case/whitespace normalization. The paired 30-claim stress differences are descriptive and recorded in explanation_diagnostic_analysis_v1. Explanation truth remains open.', '',
            'The multilingual agreement replay aligns all 6600 saved Qwen and multilingual NLI rows on eleven supplied-evidence files. A verdict is retained only if the models agree, accepting 295–393 of 600 per file. This rejects many correct Qwen answers and lowers all-claim accuracy when abstentions count as errors. The exploratory replay uses no new model inference and cannot establish multilingual retrieval, evidence sufficiency, explanation truth or calibrated confidence; details and receipt are in artifacts/multilingual_agreement_replay_v1.', '',
            'The separate frozen XFEVER reliability audit computes fixed-bin top-label ECE and multiclass Brier on those same 6600 multilingual NLI probability rows, per file. Mean confidence ranges .903–.964 while accuracy ranges .663–.727; top-label ECE ranges .228–.267. The eleven translated files overlap by underlying English claim, so do not pool them as independent samples. This descriptive measurement does not fit a calibrator or show calibrated APV-RAG deployment. See artifacts/xfever_reliability_audit_v1.', '',
            'An opt-in four-policy source-snapshot wrapper is replayed on 2400 original and 1440 stress policy records using saved feature entries and exact responses. It preserves all 2400 original policy results, and refuses 299/360 stress contexts per policy before generation. Sixty refusals per policy reflect absent evidence, and one swapped climate context passes matching excerpts. The original completed runner and its receipt remain unchanged. This cached integration is not a live neural result or publisher authentication. See artifacts/bound_policy_integration_v1.', '',
            'The pinned Wikipedia archive was downloaded and indexed on Windows: 5416537 raw records, 5416536 pages and one audited empty placeholder. The first frozen 300-claim FEVER NLI evaluation is verified at 47.33% accuracy. A separate 600-development/300-confirmation NLI calibration experiment is verified: confirmation accuracy stays 53%, ECE falls from .2539 to .0227, and confidence ranking at 50% and 80% coverage gets worse. A further disjoint 300-development/300-confirmation integrated English pipeline is verified for 2400 policy records and 1200 distinct generated responses. Final-answer correctness calibration improves probability losses but cannot repair verdicts. These are separate cohorts and must not be pooled or compared as if the same claims were used.', '',
            'Do not repeat completed generation experiments or invent more tuning stages on observed cohorts. A final technical audit cannot establish source authentication or explanation truth. Any narrower research scope must be explicitly agreed before closing the full charter.', '',
            '## Verification limits', '', 'Received output hashes, non-neural retrieval/guard/scoring replay and diagnostic consistency checks were performed. Neural inference is executed on the user machine and was not independently rerun here. Source authentication and factual explanation grounding are not established. Earlier completion/readiness statements are limited to their named stage, not the whole project. No new human review, manuscript or GitHub push is authorized at this stage.']
    lines += ['', '## Verified FEVER confirmation and matched-coverage controls', '',
              '| Policy | All-claim accuracy | Coverage | Accepted accuracy | Correctness ECE |',
              '|---|---:|---:|---:|---:|']
    for policy in ('no_gate','nli','embedding','combined'):
        r=pipeline_summary['policies'][policy]
        ece=r['correctness_calibration']['calibrated_metrics']['ece_15_bins']
        lines.append(f"| {policy} | {r['accuracy_all_claims_abstentions_as_errors']:.2%} | {r['coverage']:.2%} | {r['covered_accuracy']:.2%} | {ece:.4f} |")
    lines += ['', 'Correctness ECE is measured only among accepted answers. Abstentions count as errors in all-claim accuracy. Gates are paired selections of common generated answers, not independent generators.', '',
              'The post-result equal-coverage controls do not establish a learned-gate advantage. At 192 answers, NLI gating gives 56.25% accepted accuracy versus 63.02% for simple NLI ranking. The three gates are numerically below that ranking control at their respective coverages; none of six group comparisons establishes superiority after Holm correction. The 300 confirmation claims form 269 shared-page/claim groups, largest size four. This is exploratory, conditional on frozen selections, and does not select or retune a policy. See artifacts/fever_selective_analysis_v1/RESULTS.md.', '',
              'Do not repeat completed English pipeline, calibration or matched-coverage stages. The original charter remains open for historical publisher authentication and general provenance efficacy, semantic explanation truth, full multilingual retrieval/gate/calibration transfer, and remaining source/date/family/trained-robustness ablations. Existing machine diagnostics cannot establish those stronger claims. No new human review or manuscript is performed.']
    lines += ['', '## Multilingual retrieval pool diagnostic', '',
              'The separately identified XFEVER retrieval stage is now completed and replayed for 6600 queries per analyzer (13200 total query results) across eleven files. Each query receives only a claim and searches the pooled evidence excerpts for that file. No target labels, page titles or query-specific target IDs enter retrieval. The pools contain 538-571 unique excerpts and include every target by construction, so the result is an optimistic limited-pool diagnostic on the already observed XFEVER cohorts.', '',
              'On 400 SUPPORTS/REFUTES pairs per file, the stock word tokenizer versus words plus CJK characters/bigrams yields 29.75% versus 68.50% paired target hit@3 for machine-translated Japanese, and 21.75% versus 71.50% for machine-translated Chinese. Human-translated targets yield 37.25% versus 66.50% Japanese and 34.25% versus 79.00% Chinese. No significance or general-system superiority claim is made. These are target-excerpt retrieval rates, not verdict accuracy. All translated variants share underlying English claims.', '',
              'This closes the bounded excerpt-pool retrieval diagnostic. Full-page/open-web multilingual search and calibrated deployment remain unevaluated. The bounded downstream controller experiment is reported below. Do not call this the completed multilingual APV-RAG pipeline. See artifacts/xfever_retrieval_pool_v1/RESULTS.md.']
    lines += ['', '## Verified multilingual retrieved controller', '',
              'The Windows multilingual retrieved-controller experiment is received and verified for 660 queries, 2640 policy records, 660 feature records per model, 1978 passage scores per NLI model and 1316 distinct Qwen responses. The save-recovery receipt preserves 422 previously validated English feature records. The sixty underlying claims are shared across eleven variants; do not pool translated rows as independent samples. No model rerun, source authentication or factual explanation judgment was performed locally.', '',
              'Multilingual NLI accuracy is numerically higher than English NLI on all ten translated sets (46.67-63.33% versus 33.33-51.67%) and lower on English (41.67% versus 46.67%). No gate improves all-query accuracy over common no-gate answers in any file; the combined gate removes both correct and incorrect answers. This is not an equal-coverage superiority comparison. Generated-answer correctness confidence remains null for all 2640 policy records. No target fitting or English FEVER confidence transfer was applied.', '',
              'This bounded retrieved-context transfer experiment is closed and should not be rerun. Results and non-neural verification receipts are in artifacts/multilingual_retrieved_pipeline_audit_v1. Independent multilingual correctness-calibration confirmation, full-page/open-web transfer and the stronger historical authentication/explanation requirements in the original charter remain open. The grouped exploratory confidence analysis is reported below. Manuscript writing still requires user permission; no GitHub push has been performed.']
    lines += ['', '## Grouped multilingual correctness-confidence validation', '',
              'Five label-blind folds divide the sixty observed claim IDs into twelve IDs per fold; all eleven variants of a claim stay together. For each policy and fold, an existing logistic C=1 calibrator is fitted only on the other forty-eight claim IDs. Twenty fits produce held-out confidences for the accepted answers in the 2640 original policy records. Original answers, gate decisions, thresholds and source files remain unchanged. No neural inference or new human annotation is performed.', '',
              'Compared with raw generated-label NLI scores, cross-fitted Brier, NLL and ECE decrease in all 44 file/policy cells. For the combined policy, pooled descriptive Brier falls from 0.3526 to 0.2411 and ECE from 0.3361 to 0.0981 among 534 accepted variant rows. A constant development-prevalence control gives Brier 0.2507 and ECE 0.0631: logistic scoring has lower Brier/NLL but higher ECE. This mixed comparison is retained in artifacts/multilingual_correctness_calibration_v1/RESULTS.md and summary.json.', '',
              'This closes the grouped exploratory calibration analysis, not a new independent confirmation test. These claims and outcomes were previously observed; shared excerpts across different claim IDs are not separated. Translated rows must not be treated as independent samples. No production calibrator is refitted on all sixty claims. Independent new-claim confidence confirmation, historical source authentication, explanation truth and full-page/open-web transfer remain unestablished. No manuscript or GitHub push was performed.']
    if confirmation_summary is None:
        lines += ['', '## New-claim multilingual confirmation: prepared, inference pending', '',
            'One hundred project-held-out claim IDs across six variants (600 queries) and development-only confidence models '
            'are frozen. Run scripts/run_multilingual_confirmation.py with the existing Windows models. No confirmation '
            'outcomes are available. Full-page/open-web transfer, authentication and explanation truth remain open.']
    else:
        lines += ['', '## Verified new-claim multilingual confidence confirmation', '',
            'The Windows export is received and replayed: 600 queries, 2400 policy records, 600 context-bound entries per '
            'feature model, 1800 passage scores per model and 1200 distinct responses. The fixed 100 underlying IDs are '
            'project-held-out; translated variants share those claims. Four prior-development logistic fits and prevalence '
            'controls are unchanged. No confirmation fitting, neural rerun, policy selection or threshold tuning was performed.', '',
            'Logistic Brier/NLL/ECE decrease versus raw scores in all 24 language/policy cells. Logistic Brier/NLL decrease '
            'versus the fixed prevalence control in 21 of 24 cells; prevalence has lower ECE in every cell and every pooled '
            'policy. For combined outputs, pooled descriptive Brier is 0.3326 raw, 0.2388 logistic and 0.2484 prevalence. '
            'ECE is 0.2892 raw, 0.0446 logistic and 0.0110 prevalence. Retain this mixed comparison. No gate improves '
            'all-query accuracy over common no-gate outputs in any language. No superiority test is claimed.', '',
            'This bounded confirmation experiment is complete and should not be rerun. See '
            'artifacts/multilingual_confirmation_audit_v1/RESULTS.md. Machine translations, larger target-derived excerpt '
            'pools and unknown pretrained-model benchmark exposure limit the result. Historical authentication, factual '
            'explanation verification and full-page/open-web multilingual transfer remain unestablished. The full charter '
            'is unfinished. No manuscript or GitHub push was performed.']
    if origin_summary is not None:
        lines += ['', '## Fixed institutional source origin and capture binding', '',
            f"The origin collector saved {origin_summary['captured_sources']}/{origin_summary['attempted_sources']} "
            'fixed institutional HTML pages with default certificate/hostname verification, exact URL policies, '
            'content anchors and byte receipts. Saved metadata excerpts and whitespace controls replay; eight '
            'altered or missing-source controls per captured page are refused before scoring. '
            'This is a deterministic admission diagnostic, not a model accuracy or provenance efficacy experiment.', '',
            'The pages contain catalogue/navigation text. No newspaper-image transcription, historical quotation, '
            'author, primary-source status, publication-date truth or embedded third-party resource is authenticated. '
            'Unsigned receipts assume a trusted collector and cannot independently attest past TLS exchanges. '
            'Historical authentication remains open. See artifacts/source_origin_audit_v1/RESULTS.md. '
            'No neural model, human annotation, manuscript or GitHub publication is involved.']
    if display_summary is not None:
        lines += ['', '## Exact cited-excerpt evidence display layer', '',
            f"The display layer reconstructs {display_summary['records']} saved policy records from "
            f"{display_summary['unique_claim_contexts']} claim/condition contexts and "
            f"{display_summary['totals']['cited_excerpts']} literal cited excerpts. Every item binds its "
            'document ID, original sentence-selection indices, excerpt bytes and frozen document hash. '
            'The unchanged snapshot rule and collapsed context must admit every retrieved passage. '
            'Upstream abstentions remain abstentions; generated rationale text is excluded.', '',
            'These are structured evidence packets, not certified semantic explanations. Literal source '
            'occurrence cannot establish verdict entailment, relevance, completeness or publisher truth. '
            'No original answer, benchmark result, policy or confidence fit was changed, and no model '
            'inference or human label was added. The earlier NLI explanation diagnostic remains separate. '
            'Semantic explanation truth and the full charter remain open. See '
            'artifacts/evidence_display_audit_v1/RESULTS.md. No manuscript or GitHub push was performed.']
    if tradeoff_summary is not None:
        lines += ['', '## Cited-evidence guard correctness/coverage trade-offs', '',
            'All 3840 saved English policy records are paired with existing benchmark labels. Original '
            'candidates are preserved; altered contexts lose both correct and incorrect answers. In the '
            '60-case no-gate OCR control, the guard withholds 28 correct and 32 incorrect candidates. '
            'One swapped no-gate context still passes with an incorrect answer. The strict rule therefore '
            'cannot establish broad robustness, semantic explanation truth or accuracy benefit. '
            'This is descriptive replay on already observed, overlapping contexts; no threshold selection '
            'or model inference is performed. See artifacts/guard_tradeoffs_v1/RESULTS.md. '
            'Remaining trained/source/date/family experiments and the full charter remain open.']
    lines+=['', '## Latest verified steps', '',
        'Received scoped English and multilingual clean-confirmation reruns are verified. The latest receipt checks cover their saved exports; full development training and corpus/index reproduction remain outside that result.', '',
        'The exploratory AVeriTeC capture-timing augmentation lowers macro-F1 from 0.5193 to 0.4796, so the earlier classifier is retained. The multilingual source-pool guard admits 600 originals and withholds 600 same-claim foreign-pool substitutions. These are bounded checks and do not establish provenance efficacy, semantic language detection or completion of the charter.']
    (out/'PROJECT_STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json_atomic(out/'output_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.name!='output_manifest.json'})
    print(f'Consolidated {len(entries)} verified runs and {len(requirements)} requirements')


if __name__=='__main__':main()
