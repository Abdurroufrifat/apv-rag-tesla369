"""Exploratory guard replay on existing generation; no neural rerun."""
import json
from pathlib import Path
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2
from apv_rag.splits import sha256, write_json_atomic
from run_sentence_rag_scifact import summarize


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'artifacts/sentence_rag_received'
    manifest = json.loads((folder / 'output_manifest.json').read_text())
    if sha256(folder / 'predictions.json') != manifest['predictions.json']:
        raise ValueError('Received prediction checksum mismatch')
    rows = json.loads((folder / 'predictions.json').read_text())
    diagnostics = []
    for row in rows:
        provenance = numeric_provenance_v2(row['generated_explanation'] or '', row['shown_claim'], row['evidence'])
        reasons = [r for r in row['reasons'] if r != 'novel_numeric_value']
        if provenance['absent_from_inputs']:
            reasons.append('novel_numeric_value')
        row['replayed_numeric_v2_label'] = row['generated_verdict'] if not reasons else None
        diagnostics.append({'claim_id': row['claim_id'], 'provenance_v2': provenance, 'reasons_v2': reasons, 'label_v2': row['replayed_numeric_v2_label']})
    summary = {'rows': len(rows), 'metrics': summarize(rows, 'replayed_numeric_v2_label'),
        'rejected_claim_ids': [d['claim_id'] for d in diagnostics if d['reasons_v2']],
        'source_predictions_sha256': manifest['predictions.json'],
        'extractor_sha256': sha256(root / 'src/apv_rag/numeric_integrity_v2.py'),
        'scope': 'Post-hoc diagnostic replay. Original benchmark scores and generation unchanged. Numeric presence does not establish factual grounding. No unit conversion or source authentication.'}
    out = root / 'artifacts/numeric_guard_v2_replay'
    out.mkdir(exist_ok=True)
    write_json_atomic(out / 'diagnostics.json', diagnostics)
    write_json_atomic(out / 'summary.json', summary)
    (out / 'RESULTS.md').write_text('# Numeric extractor v2 replay\n\n'+json.dumps(summary,indent=2)+'\n\nList markers are excluded; numbers attached to explicitly listed units are recognized. Decimal forms normalize without rounding. Digits inside gene identifiers are excluded. Units are not compared, so identical values with different units can still pass. Derived numbers and scientific meaning are not verified. This is exploratory on already observed outputs, not independent confirmation.\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
