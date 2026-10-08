"""Summarize verified explanation diagnostics without selecting a threshold."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=('ocr_noise','fabricated_citation','authoritative_wording','swapped_context')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(summary):
    details=summary['details']
    paired={}
    for cohort in ('scifact','climate_retrieved'):
        keys=sorted(k for k in details if k.startswith(cohort+':') and k.endswith(':original')
                    and k.rsplit(':',1)[0]+':ocr_noise' in details)
        if len(keys)!=30:raise ValueError('Changed-context sample differs')
        base=[details[k]['max_entailment'] for k in keys]
        if any(v is None for v in base):raise ValueError('Missing baseline scores')
        paired[cohort]={'claims':len(keys),'baseline_mean_max_entailment':statistics.mean(base),
                        'conditions':{}}
        for condition in CONDITIONS:
            changed=[details[k.rsplit(':',1)[0]+':'+condition]['max_entailment'] for k in keys]
            if any(v is None for v in changed):raise ValueError('Missing changed-context scores')
            diffs=[b-a for a,b in zip(base,changed,strict=True)]
            paired[cohort]['conditions'][condition]={
                'mean_changed_max_entailment':statistics.mean(changed),
                'mean_paired_difference':statistics.mean(diffs),
                'increased':sum(x>0 for x in diffs),'decreased':sum(x<0 for x in diffs),
                'unchanged':sum(x==0 for x in diffs)}
    quotes={k:sum(v[field] for v in summary['groups'].values()) for k,field in (
        ('quoted','quotes'),('in_evidence','quotes_found_in_evidence'),
        ('in_claim_only','quotes_found_in_claim_only'),('unmatched','quotes_unmatched'))}
    return {'tasks':summary['tasks'],'pairs':summary['pairs'],
            'pairs_scored':sum(g['pairs_scored'] for g in summary['groups'].values()),
            'pairs_skipped':sum(g['pairs_skipped'] for g in summary['groups'].values()),
            'quotes':quotes,'paired':paired}


def main():
    root=ROOT/'artifacts';source=root/'explanation_diagnostic_received'
    received=json.loads((source/'output_manifest.json').read_text())
    for name,value in received.items():
        if Path(name).name!=name or digest(source/name)!=value:
            raise ValueError('Received output digest mismatch')
    windows=root/'explanation_diagnostic_windows_verification_received'
    local=root/'explanation_diagnostic_verification_v1'
    for folder in (windows,local):
        receipt=json.loads((folder/'audit_manifest.json').read_text())
        if receipt['source_output_manifest_sha256']!=digest(source/'output_manifest.json'):
            raise ValueError('Verifier input binding differs')
        if receipt['verifier_sha256']!=digest(ROOT/'scripts/verify_explanation_diagnostic.py'):
            raise ValueError('Verifier source hash differs')
        for name,hash_value in receipt['files'].items():
            if digest(folder/name)!=hash_value:raise ValueError('Verifier file hash mismatch')
    for name in ('verification.json','RESULTS.md'):
        if (windows/name).read_text()!= (local/name).read_text():
            raise ValueError('Windows/local verification content differs')
    summary=json.loads((source/'summary.json').read_text())
    result=analyze(summary)
    if result['tasks']!=840 or result['pairs']!=2520 or result['pairs_scored']+result['pairs_skipped']!=2520:
        raise ValueError('Unexpected diagnostic coverage')
    out=root/'explanation_diagnostic_analysis_v1';out.mkdir(exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# Received explanation diagnostic: bounded analysis','',
           f"The received Windows export passed local verification for {result['tasks']} saved English explanations and {result['pairs']} explanation–passage pairs. {result['pairs_scored']} pairs have saved NLI scores; {result['pairs_skipped']} exceeded the token budget. Each platform's receipt matches its own bytes, and the Windows/local verification text agrees after line-ending normalization. Model inference was not repeated here.",'',
           'The score compares an entire generated explanation with one passage at a time. It does not measure explanation truth, support from multiple passages, source authentication or calibrated confidence. Scores from Not Enough Evidence explanations need particular care because a statement that evidence is missing may not be entailed by any passage.','',
           '| Cohort | Original mean of maximum passage entailment (300) |',
           '|---|---:|']
    for cohort in ('scifact','climate_retrieved'):
        score=summary['groups'][cohort+':original']['mean_max_entailment']
        lines.append(f'| {cohort} | {score:.4f} |')
    lines+=['','The next table pairs the same thirty claims within each cohort. Differences are descriptive and the four stress prompts can change the generated answer as well as the premise text. No threshold or significance decision was made.','',
            '| Cohort | Perturbation | Baseline mean (30) | Changed mean (30) | Mean paired difference | Increases / decreases |',
            '|---|---|---:|---:|---:|---:|']
    for cohort,values in result['paired'].items():
        for condition,rec in values['conditions'].items():
            lines.append(f"| {cohort} | {condition} | {values['baseline_mean_max_entailment']:.4f} | {rec['mean_changed_max_entailment']:.4f} | {rec['mean_paired_difference']:+.4f} | {rec['increased']}/{rec['decreased']} |")
    q=result['quotes']
    lines+=['',f"Literal quoted spans: {q['quoted']}; found in evidence {q['in_evidence']}, found only in the claim {q['in_claim_only']}, and unmatched by case/whitespace normalized text {q['unmatched']}. These are string checks. A mismatch can result from paraphrase or punctuation and is not a factual error label.",'',
            'The fabricated-citation and authority controls sometimes raise the NLI proxy. Prefix length, tokenization and changed generator wording can contribute. The proxy cannot certify that an archive citation is genuine or an explanation accurate. Keep the mixed results and leave explanation truth unresolved.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'audit_manifest.json').write_text(json.dumps({
        'source_output_manifest_sha256':digest(source/'output_manifest.json'),
        'windows_verification_sha256':digest(windows/'audit_manifest.json'),
        'local_verification_sha256':digest(local/'audit_manifest.json'),
        'analysis_script_sha256':digest(Path(__file__)),
        'files':{p.name:digest(p) for p in out.iterdir() if p.name!='audit_manifest.json'}},indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('Analyzed 840 verified explanations and paired stress controls.')


if __name__=='__main__':main()
