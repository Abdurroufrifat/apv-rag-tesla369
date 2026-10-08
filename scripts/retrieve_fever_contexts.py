"""Retrieve the frozen FEVER claims from the pinned index, without labels."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from apv_rag.fever_corpus import (EXPECTED_ARCHIVE_SHA256, EXPECTED_PAGES,
                                  file_sha256, retrieve_claims)


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'data/external/fever/heldout_v1'
    db_path = root / 'data/processed/fever/heldout_v1/wiki.sqlite'
    output = root / 'artifacts/fever_retrieval_v1'
    output_file = output / 'contexts.jsonl'
    receipt = output / 'input_manifest.json'
    if output_file.exists() or receipt.exists():
        raise FileExistsError('FEVER retrieval output exists; refusing to replace a frozen run')
    hashes = json.loads((source / 'output_manifest.json').read_text(encoding='utf-8'))
    for name in ('model_inputs.jsonl', 'selection_manifest.json'):
        if file_sha256(source / name) != hashes[name]:
            raise ValueError(f'Frozen FEVER input changed: {name}')
    selection = json.loads((source / 'selection_manifest.json').read_text(encoding='utf-8'))
    claims = [json.loads(line) for line in (source / 'model_inputs.jsonl').read_text(
        encoding='utf-8').splitlines()]
    if len(claims) != 300 or [r['id'] for r in claims] != selection['selected_ids']:
        raise ValueError('Frozen FEVER cohort differs from manifest')
    with closing(sqlite3.connect(f'file:{db_path.resolve().as_posix()}?mode=ro', uri=True)) as con:
        index_meta = dict(con.execute('SELECT key, value FROM metadata'))
        page_count = con.execute('SELECT COUNT(*) FROM pages').fetchone()[0]
        empty_count = con.execute('SELECT COUNT(*) FROM empty_placeholders').fetchone()[0]
    if (index_meta.get('archive_sha256') != EXPECTED_ARCHIVE_SHA256 or
            index_meta.get('source_rows') != str(EXPECTED_PAGES) or
            index_meta.get('pages') != str(page_count) or
            index_meta.get('empty_placeholders') != str(empty_count) or
            page_count < 1 or page_count + empty_count != EXPECTED_PAGES):
        raise ValueError('FEVER index does not match pinned archive and audited record counts')
    result = retrieve_claims(db_path, source / 'model_inputs.jsonl', output_file)
    if result['claims'] != 300:
        output_file.unlink()
        raise ValueError('Unexpected FEVER retrieval count')
    metadata = {'scope': 'claim-only FTS5 retrieval; no verdict model or gold labels opened',
                'archive_sha256': index_meta['archive_sha256'], 'page_count': page_count,
                'source_records': EXPECTED_PAGES, 'empty_placeholders': empty_count,
                'model_inputs_sha256': hashes['model_inputs.jsonl'],
                'selection_manifest_sha256': hashes['selection_manifest.json'],
                'contexts_sha256': result['contexts_sha256'],
                'retrieval': 'fts5_bm25_title3_body1, top3 documents and top3 sentences',
                'code_sha256': {name: file_sha256(root / name) for name in (
                    'src/apv_rag/fever_corpus.py', 'src/apv_rag/sentence_context.py',
                    'scripts/retrieve_fever_contexts.py')}}
    receipt.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f"FEVER retrieval complete: {result['claims']} claims; contexts SHA-256 {result['contexts_sha256']}")
    print(f'Output: {output_file}')


if __name__ == '__main__':
    main()
