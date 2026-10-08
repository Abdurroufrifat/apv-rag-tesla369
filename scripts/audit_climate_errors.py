"""Post-hoc evidence-loss diagnostics; does not change frozen predictions."""
import json
from pathlib import Path
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'data/external/climate_fever/frozen_v1'
    folder = root / 'artifacts/climate_rag_received'
    manifest = json.loads((folder/'output_manifest.json').read_text(encoding='utf-8'))
    if sha256(folder/'predictions.json') != manifest['predictions.json']:
        raise ValueError('Received prediction checksum mismatch')
    source_manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if sha256(source/'climate-fever.jsonl') != source_manifest['raw_sha256']:
        raise ValueError('Raw dataset checksum mismatch')
    raw = {r['claim_id']: r for r in map(json.loads,(source/'climate-fever.jsonl').read_text(encoding='utf-8').splitlines())}
    corpus = {r['doc_id']:r for r in map(json.loads,(source/'corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    rows = json.loads((folder/'predictions.json').read_text(encoding='utf-8'))
    details = []
    normalize = lambda text: ' '.join(text.split())
    for row in rows:
        annotated = raw[row['claim_id']]
        relevant = [e for e in annotated['evidences'] if e['evidence_label'] == annotated['claim_label'] and e['evidence_label'] in ('SUPPORTS','REFUTES')]
        candidates = []
        for gold in relevant:
            retrieved = [e for e in row['evidence'] if corpus[e['id']]['title'] == gold['article']]
            selected = any(normalize(gold['evidence']) in normalize(' '.join(corpus[e['id']]['abstract'][i] for i in e['selected_sentence_indices'])) for e in retrieved)
            shown = any(normalize(gold['evidence']) in normalize(e['text']) for e in retrieved)
            candidates.append({'evidence_id':gold['evidence_id'], 'article_retrieved':bool(retrieved), 'selected_exact_sentence':selected, 'shown_exact_sentence':shown})
        if not relevant:
            stage = 'no_matching_stance_annotation'
        elif not any(e['article_retrieved'] for e in candidates):
            stage = 'matching_stance_article_not_retrieved'
        elif not any(e['selected_exact_sentence'] for e in candidates):
            stage = 'matching_stance_sentence_not_selected'
        elif not any(e['shown_exact_sentence'] for e in candidates):
            stage = 'selected_matching_stance_sentence_not_fully_visible'
        else:
            stage = 'matching_stance_sentence_visible'
        details.append({'claim_id':row['claim_id'], 'stage':stage, 'raw_correct':row['raw_candidate_label']==row['true_label'], 'guarded_correct':row['structural_candidate_label']==row['true_label'], 'true_label':row['true_label'], 'candidates':candidates})
    stages = sorted({d['stage'] for d in details})
    summary = {stage:{'claims':sum(d['stage']==stage for d in details), 'raw_correct':sum(d['stage']==stage and d['raw_correct'] for d in details), 'guarded_correct':sum(d['stage']==stage and d['guarded_correct'] for d in details)} for stage in stages}
    out = root/'artifacts/climate_error_audit'
    out.mkdir(exist_ok=True)
    write_json_atomic(out/'claim_diagnostics.json',details)
    write_json_atomic(out/'summary.json',summary)
    lines=['# CLIMATE-FEVER evidence-loss audit', '', '| Diagnostic stage | Claims | Raw correct | Guarded correct |','|---|---:|---:|---:|']
    for stage,m in summary.items():
        lines.append(f"| {stage} | {m['claims']} | {m['raw_correct']} | {m['guarded_correct']} |")
    lines += ['', 'Gold annotations used only for this post-hoc diagnostic, never generation or retrieval. Matching stance means an evidence sentence annotated with the same SUPPORTS/REFUTES label as the claim. NEI claims and stance claims without such annotations are not evidence-recall eligible.', '', 'These are evidence-visibility stages, not exclusive causal explanations of prediction errors. Exact normalized-text visibility can miss tokenizer differences. A visible annotated sentence does not establish that it alone suffices to decide the entire claim. Incorrect predictions with visible evidence identify cases for reasoning diagnostics, not proven reasoning failure. No model inference, tuning or prediction changes.', '', 'Frozen CLIMATE-FEVER accuracy remains raw38%,guarded36%. No claim of independent confirmation, verified explanations, source authentication or project completion.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert sum(m['claims'] for m in summary.values())==300
    assert sum(m['guarded_correct'] for m in summary.values())==108
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
