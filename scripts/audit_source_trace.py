"""Audit passage-to-frozen-corpus binding and literal explanation references.

This cannot establish publisher authenticity or semantic explanation truth.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from apv_rag.source_snapshot_guard import passage_trace as _passage_trace, snapshot_gate

ROOT=Path(__file__).resolve().parents[1]
DATASETS={'scifact':'data/external/scifact/sealed_v1',
          'climate_retrieved':'data/external/climate_fever/frozen_v1'}
BRACKET=re.compile(r'\[([^\[\]]+)\]')
URL=re.compile(r'https?://[^\s\])>,;]+',re.I)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def checked_receipt(folder):
    manifest=load(folder/'output_manifest.json')
    for name,hash_value in manifest.items():
        if Path(name).name!=name or digest(folder/name)!=hash_value:
            raise ValueError(f'Output receipt mismatch: {folder.name}/{name}')
    return digest(folder/'output_manifest.json')


def passage_trace(row,corpus):
    return _passage_trace(row['claim'],row['retrieved_evidence'],corpus)


def explanation_trace(row):
    answer=row.get('generated_explanation') or ''
    allowed={str(e['id']) for e in row['evidence']}
    found=BRACKET.findall(answer)
    known=[id for id in found if id in allowed]
    input_text=' '.join([row.get('shown_claim',row['claim'])]+[e['text'] for e in row['evidence']])
    urls=[m.group().rstrip('.!?') for m in URL.finditer(answer)]
    unseen=[url for url in urls if url not in input_text]
    decisive=row.get('raw_candidate_label') in ('Supported','Refuted')
    return {'has_explanation':int(bool(answer)),'decisive_explanations':int(decisive and bool(answer)),
            'known_ids':len(known),
            'unknown_ids':len(found)-len(known),
            'unseen_urls':len(unseen),'uncited_decisive_verdict':int(decisive and bool(answer) and not known),
            'synthetic_reference_echo':int(bool(row.get('synthetic_reference')) and
                                              row['synthetic_reference'].lower() in answer.lower())}


def accumulate(rows,corpus,require_exact=False):
    totals=Counter(rows=0)
    for row in rows:
        bound=passage_trace(row,corpus)
        if require_exact and bound['bound_passages']!=bound['passages']:
            raise ValueError('Original saved evidence differs from frozen source corpus')
        totals.update(bound)
        totals.update(explanation_trace(row))
        totals['snapshot_gate_passed']+=int(snapshot_gate(bound))
        totals['snapshot_gate_refused']+=int(not snapshot_gate(bound))
        totals['snapshot_gate_suppressed_outputs']+=int(not snapshot_gate(bound) and row.get('candidate_label') is not None)
        totals['snapshot_gate_suppressed_echoes']+=int(not snapshot_gate(bound) and
            bool(row.get('synthetic_reference')) and
            row['synthetic_reference'].lower() in (row.get('generated_explanation') or '').lower())
        totals['rows']+=1
    return dict(totals)


def main():
    corpus={};raw_digest={}
    for cohort,relative in DATASETS.items():
        folder=ROOT/relative;manifest=load(folder/'manifest.json')
        for name in ('claims_dev.jsonl','corpus.jsonl'):
            raw_digest[f'{cohort}/{name}']=digest(folder/name)
            if raw_digest[f'{cohort}/{name}']!=manifest['files'][name]:
                raise ValueError('Frozen raw dataset mismatch')
        docs=[json.loads(line) for line in (folder/'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
        corpus[cohort]={str(doc['doc_id']):doc for doc in docs}
        if len(corpus[cohort])!=len(docs):raise ValueError('Duplicate corpus IDs')
    fresh=ROOT/'artifacts/fresh_pipeline_received';stress=ROOT/'artifacts/text_stress_received'
    source_receipts={folder.name:checked_receipt(folder) for folder in (fresh,stress)}
    for name,origin in [('fresh_pipeline_verification_v1',fresh),('text_stress_verification_v1',stress)]:
        verification=load(ROOT/'artifacts'/name/'audit_manifest.json')
        if verification['source_output_manifest_sha256']!=source_receipts[origin.name]:
            raise ValueError('Source verifier binding differs')
    baseline={}
    for row in load(fresh/'predictions.json'):
        if row['policy']!='no_gate':continue
        key=(row['cohort'],str(row['claim_id']))
        if key in baseline:raise ValueError('Duplicate baseline')
        baseline[key]=row
    if len(baseline)!=600:raise ValueError('Missing baseline claims')
    baseline_summary={cohort:accumulate([row for key,row in baseline.items() if key[0]==cohort],corpus[cohort],True)
                      for cohort in DATASETS}
    contexts=load(stress/'contexts.json')
    stress_rows={}
    for row in load(stress/'predictions.json'):
        if row['policy']!='no_gate':continue
        key=(row['cohort'],str(row['claim_id']),row['condition'])
        if key in stress_rows:raise ValueError('Duplicate stress context')
        context=contexts[f'{key[0]}:{key[1]}:{key[2]}']
        if row['evidence']!=context['evidence']:
            raise ValueError('Stress context differs from saved prediction')
        row={**context,**row}
        parent=baseline.get(key[:2])
        if parent is None or row['claim']!=parent['claim'] or row['true_label']!=parent['true_label']:
            raise ValueError('Stress claim and label binding mismatch')
        stress_rows[key]=row
    if len(stress_rows)!=360:raise ValueError('Stress coverage differs')
    stress_summary={cohort:{} for cohort in DATASETS}
    for cohort in DATASETS:
        for condition in ('baseline','ocr_noise','fabricated_citation','authoritative_wording',
                          'swapped_context','evidence_absent'):
            subset=[r for k,r in stress_rows.items() if k[0]==cohort and k[2]==condition]
            if len(subset)!=30:raise ValueError('Stress cohort/condition coverage differs')
            for row in subset:
                parent=baseline[cohort,str(row['claim_id'])]
                if condition=='baseline' and (row['retrieved_evidence']!=parent['retrieved_evidence'] or
                                              row['evidence']!=parent['evidence']):
                    raise ValueError('Stress baseline context differs from source')
            stress_summary[cohort][condition]=accumulate(subset,corpus[cohort],condition=='baseline')
    folder=ROOT/'artifacts/source_trace_audit_v1';folder.mkdir(exist_ok=True)
    payload={'scope':'Exact frozen-corpus passage/selection trace and literal explanation references; no publisher authentication or semantic truth assessment.',
             'raw_dataset_sha256':raw_digest,'received_receipts_sha256':source_receipts,
             'baseline':baseline_summary,'stress':stress_summary}
    (folder/'audit.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# Source snapshot and explanation trace audit','',
           'This checks saved excerpts against the frozen benchmark corpus and the recorded claim-specific sentence selection. A match means the excerpt begins with the selected corpus text; tokenizer clipping length is not recounted. Corpus checksums match the frozen source receipts. Dataset membership does not authenticate a publisher, the original webpage or a historical claim.','',
           '| Cohort | Original passages bound / saved | Explanations | Unknown bracket IDs | Unseen URLs | Uncited Supported/Refuted explanations |',
           '|---|---:|---:|---:|---:|---:|']
    for cohort,metrics in baseline_summary.items():
        lines.append(f"| {cohort} | {metrics['bound_passages']}/{metrics['passages']} | {metrics['has_explanation']} | {metrics['unknown_ids']} | {metrics['unseen_urls']} | {metrics['uncited_decisive_verdict']}/{metrics['decisive_explanations']} |")
    lines+=['','Stress controls use thirty claims per cohort. The counts below are passages still bound to the *original claim* and frozen corpus; altered text or a donor passage is expected to fail this exact test. An absent context contains no passages.','',
            '| Cohort | Condition | Bound passages / saved | Explanations | Unknown IDs | Unseen URLs | Synthetic reference echoes |',
            '|---|---|---:|---:|---:|---:|']
    for cohort,conditions in stress_summary.items():
        for condition,m in conditions.items():
            lines.append(f"| {cohort} | {condition} | {m['bound_passages']}/{m['passages']} | {m['has_explanation']} | {m['unknown_ids']} | {m['unseen_urls']} | {m['synthetic_reference_echo']} |")
    lines+=['','Only the ungated row is counted per claim or stress context, avoiding repeated answers shared across policies. Bracketed IDs and URLs are checked as literal strings, not as factual support. A missing ID may reflect the current prompt, which does not require citations. Unseen URLs are flags for review, not automatically false claims. The generated explanations were not semantically judged and no human review or new model inference was performed. Source authentication and factual explanation validation remain open.']
    lines+=['', '## Pinned snapshot rule replay','',
            'A possible pre-generation rule accepts a context only if it has at least one passage and every passage matches the selected frozen corpus text for the current claim. This is an offline replay on existing answers, not a measured live controller or proof of attack detection outside this synthetic batch.','',
            '| Cohort | Condition | Would pass / 30 | Existing outputs suppressed | Reference echoes suppressed |',
            '|---|---|---:|---:|---:|']
    for cohort,conditions in stress_summary.items():
        for condition,m in conditions.items():
            lines.append(f"| {cohort} | {condition} | {m['snapshot_gate_passed']} | {m['snapshot_gate_suppressed_outputs']} | {m['snapshot_gate_suppressed_echoes']} |")
    lines+=['','All 600 original contexts also pass the exact snapshot rule. Legitimate OCR changes, corpus revisions and updated source text would fail until a newly pinned corpus is checked. Exact snapshot matching does not determine whether a passage is truthful, independent or relevant.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (folder/'audit_manifest.json').write_text(json.dumps({
        'raw_dataset_sha256':raw_digest,'received_receipts_sha256':source_receipts,
        'script_sha256':digest(Path(__file__)),
        'files':{n:digest(folder/n) for n in ('audit.json','RESULTS.md')}},indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('Traced 600 original and 360 stress contexts to frozen corpus snapshots.')


if __name__=='__main__':main()
