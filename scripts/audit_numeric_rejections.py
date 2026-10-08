"""Post-hoc list-marker sensitivity; original numeric guard remains frozen."""
import json
import re
from pathlib import Path
from apv_rag.input_numeric_integrity import numeric_provenance
from apv_rag.splits import sha256, write_json_atomic
from run_sentence_rag_scifact import summarize


def remove_list_markers(text):
    return re.sub(r'^([ \t]*)\d+[.)][ \t]+', r'\1', text, flags=re.MULTILINE)


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'artifacts/sentence_rag_received'
    hashes = json.loads((folder / 'output_manifest.json').read_text())
    if sha256(folder / 'predictions.json') != hashes['predictions.json']:
        raise ValueError('Prediction checksum mismatch')
    rows = json.loads((folder / 'predictions.json').read_text())
    details = []
    for row in rows:
        stripped = remove_list_markers(row['generated_explanation'] or '')
        numeric = numeric_provenance(stripped, row['shown_claim'], row['evidence'])
        reasons = [r for r in row['reasons'] if r != 'novel_numeric_value']
        if numeric['absent_from_inputs']:
            reasons.append('novel_numeric_value')
        row['list_marker_sensitivity_label'] = row['generated_verdict'] if not reasons else None
        if 'novel_numeric_value' in row['reasons']:
            details.append({'claim_id': row['claim_id'], 'original_absent': row['numeric_provenance']['absent_from_inputs'], 'after_list_marker_removal_absent': numeric['absent_from_inputs'], 'verdict_correct': row['raw_candidate_label'] == row['true_label'], 'explanation': row['generated_explanation'], 'shown_claim': row['shown_claim'], 'evidence': row['evidence']})
    summary = {'original_rejections': len(details), 'list_marker_only_rejections': sum(not d['after_list_marker_removal_absent'] for d in details), 'remaining_rejections': sum(bool(d['after_list_marker_removal_absent']) for d in details), 'posthoc_list_marker_sensitivity_metrics': summarize(rows, 'list_marker_sensitivity_label'), 'scope': 'Exploratory post-hoc sensitivity, not a replacement for frozen results. Correct verdict does not establish correct explanation. No numeric canonicalization, unit conversion or factual validation performed.'}
    out = root / 'artifacts/numeric_rejection_audit'
    out.mkdir(exist_ok=True)
    write_json_atomic(out / 'cases.json', details)
    write_json_atomic(out / 'summary.json', summary)
    lines = ['# Numeric rejection audit', '', f"Original rejections: {len(details)}. Rejections explained entirely by line-start list markers: {summary['list_marker_only_rejections']}. Remaining: {summary['remaining_rejections']}.", '', 'Correction to earlier interpretation: the 12 rejected correct verdicts were not automatically false guard rejections. Verdict correctness says nothing about explanation correctness.', '', 'List markers such as 1. and 2. are formatting, not numerical evidence claims. This sensitivity removes only those markers. The frozen numeric guard and original benchmark scores remain unchanged.', '', 'Remaining cases:', '']
    for d in details:
        if d['after_list_marker_removal_absent']:
            lines.append(f"- Claim {d['claim_id']}: {d['after_list_marker_removal_absent']} remains unmatched. Inspect cases.json for supplied inputs and explanation.")
    lines += ['', summary['scope'], '', 'Numbers attached to units can be missed by the original word-boundary regex. A value absent from inputs can also be an implicit reference or derived quantity; this audit does not resolve semantic correctness. Do not treat recovered answers as verified explanations.']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
