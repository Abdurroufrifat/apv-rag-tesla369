"""Freeze disjoint FEVER calibration and confirmation claims, excluding prior outcomes."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.fever_calibration import SEED, normalized, partition_claims
from apv_rag.fever_nli import load_claims, read_jsonl
from apv_rag.splits import sha256, write_json_atomic

SOURCE_SHA256 = 'e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7'


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':')) + '\n' for r in rows), encoding='utf-8')


def main():
    out = ROOT / 'data/external/fever/calibration_v1'
    if out.exists():
        raise FileExistsError('Frozen calibration cohorts exist; refusing overwrite')
    prior = ROOT / 'data/external/fever/heldout_v1'
    previous_claims = load_claims(prior)
    source = prior / 'shared_task_dev.jsonl'
    if sha256(source) != SOURCE_SHA256:
        raise ValueError('Pinned FEVER source checksum mismatch')
    xdir = ROOT / 'artifacts/xfever_generation_received'
    xhash = json.loads((xdir / 'output_manifest.json').read_text(encoding='utf-8'))
    if sha256(xdir / 'predictions.json') != xhash['predictions.json']:
        raise ValueError('Previously observed XFEVER file changed')
    xrows = [r for r in json.loads((xdir / 'predictions.json').read_text(encoding='utf-8'))
             if r['file'] == 'en/test.6h.jsonl']
    if len(xrows) != 600:
        raise ValueError('Unexpected prior XFEVER cohort')
    excluded_ids = {r['claim_id'] for r in xrows} | {r['id'] for r in previous_claims}
    excluded_texts = {normalized(r['shown_claim']) for r in xrows} | {
        normalized(r['claim']) for r in previous_claims}
    rows = read_jsonl(source)
    if len(rows) != 19998:
        raise ValueError('Pinned FEVER source row count mismatch')
    dev, confirm = partition_claims(rows, exclude_ids=excluded_ids, exclude_texts=excluded_texts)
    out.mkdir(parents=True)
    cohorts = {}
    for name, selected in (('development', dev), ('confirmation', confirm)):
        folder = out / name
        folder.mkdir()
        write_lines(folder / 'model_inputs.jsonl', [{'id': r['id'], 'claim': r['claim']}
                                                   for r in selected])
        write_lines(folder / 'gold.jsonl', [{'id': r['id'], 'label': r['label'],
                                           'evidence': r['evidence']} for r in selected])
        cohorts[name] = {'claims': len(selected), 'selected_ids': [r['id'] for r in selected],
                         'model_inputs_sha256': sha256(folder / 'model_inputs.jsonl'),
                         'gold_sha256': sha256(folder / 'gold.jsonl')}
    write_json_atomic(out / 'selection_manifest.json', {
        'source_sha256': SOURCE_SHA256, 'selection_seed': SEED,
        'selection_rule': 'SHA-256(seed:claim_id), unique normalized text, first 600 development; next 300 confirmation',
        'prior_fever_inputs_sha256': sha256(prior / 'model_inputs.jsonl'),
        'prior_xfever_predictions_sha256': sha256(xdir / 'predictions.json'),
        'excluded_prior_ids': sorted(excluded_ids), 'excluded_prior_texts': sorted(excluded_texts),
        'cohorts': cohorts, 'selection_uses_labels_or_evidence': False,
        'limitation': 'Fresh claims from the same public FEVER development set; not an official blind test or independent dataset. Exact IDs and normalized text are excluded; near duplicates and shared Wikipedia pages may remain.'})
    write_json_atomic(out / 'output_manifest.json', {
        str(p.relative_to(out)).replace('\\', '/'): sha256(p)
        for p in sorted(out.rglob('*')) if p.is_file()})
    print('Frozen 600 calibration claims and 300 confirmation claims; zero ID/text overlap.')
    print('Selection manifest SHA-256:', sha256(out / 'selection_manifest.json'))


if __name__ == '__main__':
    main()
