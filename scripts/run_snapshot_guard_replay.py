"""Replay an optional source-snapshot guard before frozen cached generation."""
import json
from collections import Counter

from apv_rag.source_snapshot_guard import execute_snapshot_guarded
from apv_rag.splits import write_json_atomic
from audit_source_trace import ROOT,DATASETS,checked_receipt,digest,load
from run_fresh_pipeline import response_key
from run_gated_generation import qwen_prompt

ENGINE_FIELDS=('evidence','gate_probability','threshold','generation_requests','stages','reasons',
               'verdict_prompt','explanation_prompt','verdict_prompt_tokens','explanation_prompt_tokens',
               'generated_verdict','generated_explanation','raw_candidate_label','candidate_label',
               'numeric_provenance')


def backend(responses,used):
    def generate(kind,question):
        prompt=qwen_prompt(question)
        key=response_key(kind,prompt)
        if key not in responses:raise ValueError('Missing exact response for permitted context')
        record=responses[key]
        if record['kind']!=kind or record['prompt']!=prompt:raise ValueError('Response binding mismatch')
        used.add(key)
        return {n:record[n] for n in ('answer','prompt','prompt_tokens')}
    return generate


def replay(rows,corpora,responses):
    used=set();generate=backend(responses,used)
    decisions=[]
    seen=set()
    for row in rows:
        key=(row['cohort'],str(row['claim_id']),row['condition'])
        if key in seen:raise ValueError('Duplicate context')
        seen.add(key)
        result=execute_snapshot_guarded(row['claim'],row['shown_claim'],row['retrieved_evidence'],
            row['evidence'],corpora[row['cohort']],generate)
        if result['snapshot_source_bound']:
            if any(result[f]!=row[f] for f in ENGINE_FIELDS):
                raise ValueError(f'Permitted context changed existing generation: {key}')
        elif result['generation_requests'] or result['candidate_label'] is not None:
            raise ValueError('Unbound evidence reached generation')
        decisions.append({'cohort':key[0],'claim_id':key[1],'condition':key[2],
                          'snapshot_source_bound':result['snapshot_source_bound'],
                          'source_trace':result['source_trace'],'reasons':result['reasons'],
                          'candidate_label':result['candidate_label'],
                          'original_candidate_label':row['candidate_label'],
                          'generation_requests':result['generation_requests'],
                          'original_generation_requests':row['generation_requests'],
                          'synthetic_reference_echo_suppressed':bool(not result['snapshot_source_bound']
                              and row.get('synthetic_reference_echo'))})
    counts={}
    for d in decisions:
        name=f"{d['cohort']}:{d['condition']}";c=counts.setdefault(name,Counter())
        c['contexts']+=1
        c['permitted']+=d['snapshot_source_bound']
        c['refused']+=not d['snapshot_source_bound']
        c['original_candidate_suppressed']+=not d['snapshot_source_bound'] and d['original_candidate_label'] is not None
        c['recorded_requests_not_issued']+=len(d['original_generation_requests']) if not d['snapshot_source_bound'] else 0
        c['reference_echo_suppressed']+=d['synthetic_reference_echo_suppressed']
    return decisions,{k:dict(v) for k,v in counts.items()},used


def main():
    fresh=ROOT/'artifacts/fresh_pipeline_received';stress=ROOT/'artifacts/text_stress_received'
    receipts={p.name:checked_receipt(p) for p in (fresh,stress)}
    audit=load(ROOT/'artifacts/source_trace_audit_v1/audit_manifest.json')
    if audit['received_receipts_sha256']!=receipts:raise ValueError('Prior trace receipt changed')
    corpora={}
    for cohort,relative in DATASETS.items():
        folder=ROOT/relative;manifest=load(folder/'manifest.json')
        if digest(folder/'corpus.jsonl')!=manifest['files']['corpus.jsonl']:
            raise ValueError('Corpus checksum changed')
        docs=[json.loads(line) for line in (folder/'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
        corpora[cohort]={str(d['doc_id']):d for d in docs}
        if len(corpora[cohort])!=len(docs):raise ValueError('Duplicate source ID')
    baseline=[{**row,'condition':'original'} for row in load(fresh/'predictions.json') if row['policy']=='no_gate']
    contexts=load(stress/'contexts.json');stressed=[]
    for row in load(stress/'predictions.json'):
        if row['policy']!='no_gate':continue
        key=f"{row['cohort']}:{row['claim_id']}:{row['condition']}"
        context=contexts[key]
        if context['evidence']!=row['evidence'] or context['true_label']!=row['true_label']:
            raise ValueError('Stress context binding changed')
        stressed.append({**context,**row})
    if len(baseline)!=600 or len(stressed)!=360:raise ValueError('Context coverage changed')
    old={p.name:load(p/'responses.json') for p in (fresh,stress)}
    base_decisions,base_counts,base_used=replay(baseline,corpora,old[fresh.name])
    stress_decisions,stress_counts,stress_used=replay(stressed,corpora,old[stress.name])
    if sum(not d['snapshot_source_bound'] for d in base_decisions)!=0:
        raise ValueError('Pinned original context refused')
    out=ROOT/'artifacts/snapshot_guard_replay_v1';out.mkdir(exist_ok=True)
    summary={'scope':'Optional pre-generation guard on a pinned corpus snapshot; cached-response offline controller replay, not a live neural test or publisher authentication.',
             'source_receipts_sha256':receipts,'original_contexts':len(baseline),
             'stress_contexts':len(stressed),'distinct_exact_responses_reused':{
                 'original':len(base_used),'stress':len(stress_used)},
             'counts':{**base_counts,**stress_counts}}
    write_json_atomic(out/'decisions.json',base_decisions+stress_decisions)
    write_json_atomic(out/'summary.json',summary)
    lines=['# Optional source-snapshot guard replay','',
           'This separate controller checks every passage against the pinned corpus before requesting a verdict or explanation. It preserves existing frozen controller results when the source check passes. Invalid contexts abstain before cached generation is requested. No model weights were loaded, no new answers were generated and the established policies were not modified.','',
           '| Cohort | Condition | Permitted / contexts | Existing candidates suppressed | Recorded requests not issued | Fake-reference echoes suppressed |',
           '|---|---|---:|---:|---:|---:|']
    for name,c in summary['counts'].items():
        lines.append(f"| {name.split(':')[0]} | {name.split(':')[1]} | {c['permitted']}/{c['contexts']} | {c['original_candidate_suppressed']} | {c['recorded_requests_not_issued']} | {c['reference_echo_suppressed']} |")
    lines+=['','Request counts are replay counts from saved answers, not measured speed or cost savings. The rule is intentionally strict: it rejects legitimate OCR edits or source revisions and may pass a swapped context whose selected excerpts still match the target claim. A corpus snapshot cannot establish the original publisher, claim relevance or explanation truth. It is an optional bounded integrity control, not an authenticated full RAG evaluation.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json_atomic(out/'audit_manifest.json',{'source_receipts_sha256':receipts,
        'script_sha256':digest(__import__('pathlib').Path(__file__)),
        'guard_sha256':digest(ROOT/'src/apv_rag/source_snapshot_guard.py'),
        'files':{p.name:digest(p) for p in out.iterdir() if p.name!='audit_manifest.json'}})
    print('Replayed 600 original and 360 stress contexts with the optional source guard.')


if __name__=='__main__':main()
