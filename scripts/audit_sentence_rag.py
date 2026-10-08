"""Post-hoc evidence and explanation diagnostics; never alters predictions."""
import json
from pathlib import Path
from collections import Counter
from apv_rag.splits import sha256, write_json_atomic


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'artifacts/sentence_rag_received'
    hashes = json.loads((folder / 'output_manifest.json').read_text())
    for name in ('predictions.json', 'input_manifest.json', 'generation_summary.json'):
        if sha256(folder / name) != hashes[name]:
            raise ValueError('Received output checksum mismatch')
    source = root / 'data/external/scifact/sealed_v1'
    identity = json.loads((folder / 'input_manifest.json').read_text())
    for name, digest in identity['source_sha256'].items():
        if sha256(source / name) != digest:
            raise ValueError('Source checksum mismatch')
    claims = {c['id']: c for c in map(json.loads, (source / 'claims_dev.jsonl').read_text().splitlines())}
    corpus = {c['doc_id']: c for c in map(json.loads, (source / 'corpus.jsonl').read_text().splitlines())}
    rows = json.loads((folder / 'predictions.json').read_text())
    diagnostics = []
    for row in rows:
        claim = claims[row['claim_id']]
        gold_ids = {int(k) for k in claim['evidence']}
        retrieved = {e['id']: e['text'] for e in row['evidence']}
        hit = bool(gold_ids & set(retrieved)) if gold_ids else None
        visible_sets = []
        for doc, annotations in claim['evidence'].items():
            doc_id = int(doc)
            for annotation in annotations:
                spans = annotation['sentences']
                if spans:
                    visible_sets.append(doc_id in retrieved and all(
                        ' '.join(corpus[doc_id]['abstract'][i].split()) in ' '.join(retrieved[doc_id].split())
                        for i in spans
                    ))
        diagnostics.append({
            'claim_id': row['claim_id'],
            'gold_evidence_document_hit': hit,
            'complete_gold_sentence_set_visible_exact_text': any(visible_sets) if gold_ids else None,
            'guarded_correct': row['structural_candidate_label'] == row['true_label'],
            'guarded_answered': row['structural_candidate_label'] is not None,
            'sentence_max_entailments': [s['max_entailment'] for s in row['explanation_nli_audit']],
            'numeric_rejection': 'novel_numeric_value' in row['reasons'],
        })
    annotated = [d for d in diagnostics if d['gold_evidence_document_hit'] is not None]
    breakdown = {}
    for key in ('gold_evidence_document_hit', 'complete_gold_sentence_set_visible_exact_text'):
        breakdown[key] = {str(flag): {'claims': sum(d[key] == flag for d in annotated),
            'guarded_correct': sum(d[key] == flag and d['guarded_correct'] for d in annotated)}
            for flag in (True, False)}
    sentences = [x for d in diagnostics for x in d['sentence_max_entailments']]
    summary = {
        'claims': len(rows), 'gold_evidence_annotated_claims': len(annotated),
        'document_recall_at_3_claim_level': sum(d['gold_evidence_document_hit'] for d in annotated) / len(annotated),
        'complete_gold_sentence_set_visible_exact_text_fraction': sum(d['complete_gold_sentence_set_visible_exact_text'] for d in annotated) / len(annotated),
        'breakdown': breakdown, 'sentences': len(sentences),
        'nli_entailment_threshold_diagnostics': {str(t): {
            'sentences_at_or_above': sum(x >= t for x in sentences),
            'claims_all_sentences_at_or_above': sum(bool(d['sentence_max_entailments']) and all(x >= t for x in d['sentence_max_entailments']) for d in diagnostics)
        } for t in (.5, .7, .9)},
        'guarded_wrong_label_counts': dict(Counter(r['true_label'] for r in rows if r['structural_candidate_label'] and r['structural_candidate_label'] != r['true_label'])),
        'scope': 'Post-hoc diagnostics only. Gold evidence used for analysis, never retrieval or generation. No threshold selected or applied.',
        'limitations': 'Full gold-sentence visibility is an exact normalized-text diagnostic, not semantic sufficiency. Sentence truncation or tokenizer decoding can cause nonmatches. NLI entailment is a machine score, not verified grounding; NEI explanations can correctly describe absence without being entailed by a passage. Gold document recall excludes claims without gold evidence annotations. Neural inference not rerun.',
    }
    output = root / 'artifacts/sentence_rag_audit'
    output.mkdir(exist_ok=True)
    write_json_atomic(output / 'claim_diagnostics.json', diagnostics)
    write_json_atomic(output / 'audit_summary.json', summary)
    lines = ['# Evidence and explanation audit', '',
        f"Gold-evidence annotated claims: {len(annotated)} / {len(rows)}.",
        f"At least one gold document retrieved in top three: {summary['document_recall_at_3_claim_level']:.2%}.",
        f"At least one complete gold sentence set visible by exact normalized text: {summary['complete_gold_sentence_set_visible_exact_text_fraction']:.2%}.", '',
        '## Saved NLI score diagnostics', '', '| Threshold | Sentences at or above | Claims with every sentence at or above |', '|---|---:|---:|']
    for t, counts in summary['nli_entailment_threshold_diagnostics'].items():
        lines.append(f"| {t} | {counts['sentences_at_or_above']} / {len(sentences)} | {counts['claims_all_sentences_at_or_above']} / {len(rows)} |")
    lines += ['', summary['scope'], '', summary['limitations'], '', 'These diagnostics do not establish source authentication, learned evidence sufficiency or factual explanation accuracy. No manuscript or GitHub push.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(output / 'output_manifest.json', {p.name: sha256(p) for p in output.iterdir() if p.name != 'output_manifest.json'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
