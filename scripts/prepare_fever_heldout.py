"""Freeze external FEVER claims before any APV-RAG scoring or model selection."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from apv_rag.splits import sha256, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://fever.ai/download/fever/shared_task_dev.jsonl'
SOURCE_SHA256 = 'e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7'
CORPUS_URL = 'https://fever.ai/download/fever/wiki-pages.zip'
LABELS = {'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'}
SEED = 'apv-rag-fever-heldout-v1'


def normalized(text):
    return ' '.join(text.casefold().split())


def select_claims(rows, count=300, exclude_ids=(), exclude_claims=()):
    seen = set()
    for row in rows:
        if (type(row.get('id')) is not int or not isinstance(row.get('claim'), str)
                or not row['claim'].strip() or row.get('label') not in LABELS
                or not isinstance(row.get('evidence'), list)):
            raise ValueError('Invalid FEVER claim row')
        if row['id'] in seen:
            raise ValueError('Duplicate FEVER claim ID')
        seen.add(row['id'])
    if type(count) is not int or count < 1 or count > len(seen):
        raise ValueError('Invalid selection count')
    excluded_ids = set(exclude_ids)
    excluded_texts = {normalized(text) for text in exclude_claims}
    selected = []
    selected_texts = set()
    # Selection sees IDs and claim text only. Labels and evidence do not enter ranking.
    for row in sorted(rows, key=lambda row: (
        hashlib.sha256(f"{SEED}:{row['id']}".encode('ascii')).hexdigest(),
        row['id'])):
        text = normalized(row['claim'])
        if row['id'] in excluded_ids or text in excluded_texts or text in selected_texts:
            continue
        selected.append(row)
        selected_texts.add(text)
        if len(selected) == count:
            return selected
    raise ValueError('Not enough distinct, unobserved FEVER claims')


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                       separators=(',', ':')) + '\n' for row in rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if sha256(source) != SOURCE_SHA256:
        raise ValueError('Official FEVER development file SHA-256 mismatch')
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 19998:
        raise ValueError('Official FEVER development record count changed')
    xfolder = ROOT / 'artifacts/xfever_generation_received'
    xreceipt = json.loads((xfolder / 'output_manifest.json').read_text(encoding='utf-8'))
    if sha256(xfolder / 'predictions.json') != xreceipt['predictions.json']:
        raise ValueError('Previously observed XFEVER predictions changed')
    observed = [row for row in json.loads((xfolder / 'predictions.json').read_text(encoding='utf-8'))
                if row['file'] == 'en/test.6h.jsonl']
    if len(observed) != 600 or not all(type(r['claim_id']) is int for r in observed):
        raise ValueError('Unexpected XFEVER English claim coverage')
    observed_ids = {r['claim_id'] for r in observed}
    observed_claims = {normalized(r['shown_claim']) for r in observed}
    selected = select_claims(rows, exclude_ids=observed_ids,
                             exclude_claims=observed_claims)
    if any(r['id'] in observed_ids or normalized(r['claim']) in observed_claims
           for r in selected):
        raise ValueError('Selected FEVER claim overlaps observed XFEVER evaluation')
    out = ROOT / 'data/external/fever/heldout_v1'
    out.mkdir(parents=True, exist_ok=True)
    prior = out/'selection_manifest.json'
    if prior.exists():
        frozen = json.loads(prior.read_text(encoding='utf-8'))
        if frozen.get('selected_ids') != [r['id'] for r in selected]:
            raise ValueError('Frozen FEVER selection differs; do not replace an evaluated cohort')
    raw = out / 'shared_task_dev.jsonl'
    if raw.exists():
        if sha256(raw) != SOURCE_SHA256:
            raise ValueError('Frozen local FEVER source changed')
    else:
        shutil.copyfile(source, raw)
    inputs = [{'id': r['id'], 'claim': r['claim']} for r in selected]
    gold = [{'id': r['id'], 'label': r['label'], 'evidence': r['evidence']}
            for r in selected]
    write_lines(out / 'model_inputs.jsonl', inputs)
    write_lines(out / 'gold.jsonl', gold)
    write_json_atomic(out / 'selection_manifest.json', {
        'source_url': SOURCE_URL, 'source_sha256': SOURCE_SHA256,
        'source_claims': len(rows), 'selection_rule': 'lowest SHA-256(seed:integer_claim_id)',
        'selection_seed': SEED, 'selected_claims': len(selected),
        'prior_xfever_output_manifest_sha256': sha256(xfolder/'output_manifest.json'),
        'excluded_observed_xfever_claim_ids': len(observed_ids),
        'excluded_observed_xfever_claim_texts': len(observed_claims),
        'selected_ids': [r['id'] for r in selected],
        'post_selection_label_counts': dict(sorted(Counter(r['label'] for r in selected).items())),
        'corpus_url': CORPUS_URL,
        'corpus_archive_size_from_2026_10_04_http_head': 1713485474,
        'corpus_sha256': None,
        'limitations': 'The Wikipedia archive is required for full retrieval and has not been downloaded or hashed here. Labels were not used to select claims. This is a frozen input set, not an evaluation result.',
    })
    write_json_atomic(out / 'output_manifest.json', {
        p.name: sha256(p) for p in out.iterdir() if p.name != 'output_manifest.json'})
    print(f'Frozen {len(selected)} FEVER claims from {len(rows)} official development rows; no model inference.')


if __name__ == '__main__':
    main()
