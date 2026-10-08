import hashlib
import json
import sqlite3
import zipfile

import pytest

from apv_rag.fever_corpus import build_index, retrieve_claims, verify_archive


def fixture_archive(tmp_path):
    archive = tmp_path / 'wiki-pages.zip'
    rows = [
        {'id': 'Arthur_Schopenhauer', 'text': 'Arthur Schopenhauer was born in 1788.',
         'lines': '0\tArthur Schopenhauer was born in 1788.\n1\tHe was a philosopher.'},
        {'id': 'Corsica', 'text': 'Corsica is an island in the Mediterranean Sea.',
         'lines': '0\tCorsica is an island in the Mediterranean Sea.'},
    ]
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('wiki-pages/wiki-000.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def test_stream_build_and_claim_only_retrieval(tmp_path):
    archive, digest = fixture_archive(tmp_path)
    db = tmp_path / 'wiki.sqlite'
    assert verify_archive(archive, len(archive.read_bytes()), digest) == digest
    metadata = build_index(archive, db, expected_size=archive.stat().st_size,
                           expected_sha256=digest, expected_pages=2)
    assert metadata['pages'] == 2
    claims = tmp_path / 'model_inputs.jsonl'
    claims.write_text(json.dumps({'id': 100, 'claim': 'Arthur Schopenhauer was born in 1788.'}) + '\n')
    output = tmp_path / 'contexts.jsonl'
    retrieve_claims(db, claims, output, top_k=2)
    row = json.loads(output.read_text().strip())
    assert row['id'] == 100
    assert row['evidence'][0]['id'] == 'Arthur_Schopenhauer'
    assert row['evidence'][0]['selected_sentence_indices'][0] == 0
    assert 'label' not in row and 'gold' not in row
    with sqlite3.connect(db) as con:
        assert con.execute('select count(*) from pages').fetchone()[0] == 2
    with pytest.raises(FileExistsError):
        retrieve_claims(db, claims, output)
    with pytest.raises(FileExistsError):
        build_index(archive, db, expected_size=archive.stat().st_size,
                    expected_sha256=digest, expected_pages=2)


def test_rejects_wrong_digest_and_duplicate_pages(tmp_path):
    archive, digest = fixture_archive(tmp_path)
    with pytest.raises(ValueError, match='SHA-256'):
        verify_archive(archive, archive.stat().st_size, '0' * 64)
    with zipfile.ZipFile(archive, 'a') as z:
        z.writestr('wiki-pages/wiki-001.jsonl', json.dumps(
            {'id': 'Corsica', 'text': 'Other', 'lines': '0\tOther'}) + '\n')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    db = tmp_path / 'wiki.sqlite'
    with pytest.raises(ValueError, match='Duplicate'):
        build_index(archive, db, expected_size=archive.stat().st_size,
                    expected_sha256=digest, expected_pages=3)
    assert not db.exists()


def test_ignores_non_data_jsonl_entries(tmp_path):
    archive, _ = fixture_archive(tmp_path)
    with zipfile.ZipFile(archive, 'a') as z:
        z.writestr('__MACOSX/wiki-pages/._wiki-000.jsonl', b'\x00\x05\x16\x07metadata')
        z.writestr('notes.jsonl', 'This is not a Wikipedia data shard')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = build_index(archive, tmp_path / 'wiki.sqlite', expected_size=archive.stat().st_size,
                         expected_sha256=digest, expected_pages=2)
    assert result['pages'] == 2


def test_malformed_real_shard_reports_member_and_line(tmp_path):
    archive = tmp_path / 'wiki-pages.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('wiki-pages/wiki-000.jsonl', 'not JSON\n')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=r'wiki-pages/wiki-000.jsonl:1'):
        build_index(archive, tmp_path / 'wiki.sqlite', expected_size=archive.stat().st_size,
                    expected_sha256=digest, expected_pages=1)


def test_empty_source_placeholders_are_counted_and_audited(tmp_path):
    archive, _ = fixture_archive(tmp_path)
    with zipfile.ZipFile(archive, 'a') as z:
        z.writestr('wiki-pages/wiki-001.jsonl', json.dumps({'id': '', 'text': '', 'lines': ''}) + '\n')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    db = tmp_path / 'wiki.sqlite'
    result = build_index(archive, db, expected_size=archive.stat().st_size,
                         expected_sha256=digest, expected_pages=3)
    assert result['pages'] == 2
    assert result['source_rows'] == 3
    assert result['empty_placeholders'] == 1
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT shard, line_number FROM empty_placeholders').fetchall() == [
            ('wiki-pages/wiki-001.jsonl', 1)]


def test_empty_id_with_nonempty_content_still_fails(tmp_path):
    archive = tmp_path / 'wiki-pages.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('wiki-pages/wiki-001.jsonl', json.dumps({'id': '', 'text': 'content', 'lines': ''}) + '\n')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Malformed FEVER page'):
        build_index(archive, tmp_path / 'wiki.sqlite', expected_size=archive.stat().st_size,
                    expected_sha256=digest, expected_pages=1)
