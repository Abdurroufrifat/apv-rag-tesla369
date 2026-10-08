"""Stream FEVER Wikipedia into SQLite FTS5 and retrieve without gold labels."""

import hashlib
import json
import re
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path

from apv_rag.sentence_context import select_sentences

EXPECTED_ARCHIVE_BYTES = 1713485474
EXPECTED_ARCHIVE_SHA256 = '4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2'
EXPECTED_PAGES = 5416537  # Official raw JSON-record count, including empty placeholders.
STOPWORDS = frozenset('a an and are as at be by for from has in is it of on or the to was were with'.split())


def data_members(archive):
    """Match the data directory used by the official FEVER loader."""
    return sorted((member for member in archive.infolist()
                   if not member.is_dir() and
                   re.fullmatch(r'wiki-pages/wiki-[0-9]+\.jsonl', member.filename)),
                  key=lambda member: member.filename)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path, expected_size=EXPECTED_ARCHIVE_BYTES,
                   expected_sha256=EXPECTED_ARCHIVE_SHA256):
    path = Path(path)
    if path.stat().st_size != expected_size:
        raise ValueError('FEVER archive size mismatch')
    digest = file_sha256(path)
    if digest.lower() != expected_sha256.lower():
        raise ValueError('FEVER archive SHA-256 mismatch')
    with zipfile.ZipFile(path) as archive:
        if not data_members(archive):
            raise ValueError('FEVER archive contains no JSONL pages')
    return digest


def build_index(archive_path, db_path, *, expected_size=EXPECTED_ARCHIVE_BYTES,
                expected_sha256=EXPECTED_ARCHIVE_SHA256, expected_pages=EXPECTED_PAGES,
                progress=None):
    """Refuse an existing index; leave no final DB if build/validation fails."""
    db_path = Path(db_path)
    if db_path.exists():
        raise FileExistsError(db_path)
    partial = db_path.with_name(db_path.name + '.partial')
    if partial.exists():
        raise FileExistsError(f'Incomplete index exists: {partial}; remove it before restarting')
    digest = verify_archive(archive_path, expected_size, expected_sha256)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    count = source_rows = empty_placeholders = 0
    try:
        with closing(sqlite3.connect(partial)) as con:
            con.execute('CREATE TABLE pages (page_id TEXT NOT NULL UNIQUE, lines TEXT NOT NULL)')
            con.execute("CREATE VIRTUAL TABLE wiki_fts USING fts5(title, body, content='')")
            con.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            con.execute('CREATE TABLE empty_placeholders (shard TEXT NOT NULL, line_number INTEGER NOT NULL, '
                        'PRIMARY KEY (shard, line_number))')
            with zipfile.ZipFile(archive_path) as archive:
                members = data_members(archive)
                if progress:
                    progress(f'Archive hash verified; reading {len(members)} Wikipedia shards')
                for member in members:
                    if progress:
                        progress(f'Reading {member.filename}; indexed {count:,} pages')
                    with archive.open(member) as stream:
                        for line_number, raw in enumerate(stream, 1):
                            if not raw.strip():
                                continue
                            try:
                                row = json.loads(raw)
                            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                                raise ValueError(
                                    f'Invalid FEVER JSON at {member.filename}:{line_number}') from exc
                            if not isinstance(row, dict) or not {'id', 'text', 'lines'} <= set(row):
                                raise ValueError(f'Malformed FEVER page at {member.filename}:{line_number}')
                            page_id, body, lines = row['id'], row['text'], row['lines']
                            if not all(isinstance(x, str) for x in (page_id, body, lines)):
                                raise ValueError(f'Malformed FEVER page at {member.filename}:{line_number}')
                            source_rows += 1
                            if page_id == body == lines == '':
                                con.execute('INSERT INTO empty_placeholders VALUES (?, ?)',
                                            (member.filename, line_number))
                                empty_placeholders += 1
                                continue
                            if not page_id:
                                raise ValueError(f'Malformed FEVER page at {member.filename}:{line_number}')
                            try:
                                cursor = con.execute(
                                    'INSERT INTO pages (page_id, lines) VALUES (?, ?)',
                                    (page_id, lines))
                            except sqlite3.IntegrityError as exc:
                                raise ValueError(f'Duplicate FEVER page: {page_id}') from exc
                            con.execute('INSERT INTO wiki_fts (rowid, title, body) VALUES (?, ?, ?)',
                                        (cursor.lastrowid, page_id.replace('_', ' '), body))
                            count += 1
                            if count % 5000 == 0:
                                con.commit()
                            if progress and count % 50000 == 0:
                                progress(f'Indexed {count:,} Wikipedia pages')
            if source_rows != expected_pages or count < 1:
                raise ValueError(f'Unexpected FEVER source record count: {source_rows}; indexed pages: {count}')
            con.execute('INSERT INTO metadata VALUES (?, ?)', ('archive_sha256', digest))
            con.execute('INSERT INTO metadata VALUES (?, ?)', ('pages', str(count)))
            con.execute('INSERT INTO metadata VALUES (?, ?)', ('source_rows', str(source_rows)))
            con.execute('INSERT INTO metadata VALUES (?, ?)', ('empty_placeholders', str(empty_placeholders)))
            con.execute('INSERT INTO metadata VALUES (?, ?)', ('retrieval', 'fts5_bm25_title3_body1'))
            con.commit()
        partial.replace(db_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {'archive_sha256': digest, 'pages': count, 'source_rows': source_rows,
            'empty_placeholders': empty_placeholders, 'retrieval': 'fts5_bm25_title3_body1'}


def claim_query(claim):
    tokens = [t for t in re.findall(r'\w+', claim.casefold()) if t not in STOPWORDS]
    tokens = list(dict.fromkeys(tokens))[:16]
    return ' OR '.join('"' + t + '"' for t in tokens)


def page_sentences(lines):
    result = []
    for line in lines.splitlines():
        parts = line.split('\t', 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
            result.append((int(parts[0]), parts[1]))
    return result


def retrieve_one(con, claim, top_k=3):
    query = claim_query(claim)
    if not query:
        return []
    ranked = con.execute(
        'SELECT pages.page_id, pages.lines, bm25(wiki_fts, 3.0, 1.0) AS score '
        'FROM wiki_fts JOIN pages ON pages.rowid = wiki_fts.rowid '
        'WHERE wiki_fts MATCH ? ORDER BY score ASC, wiki_fts.rowid ASC LIMIT ?',
        (query, top_k)).fetchall()
    evidence = []
    for page_id, lines, score in ranked:
        sentences = page_sentences(lines)
        selected = select_sentences(claim, [text for _, text in sentences])
        indices = [sentences[i][0] for i in selected['sentence_indices']]
        evidence.append({'id': page_id, 'text': selected['text'],
                         'selected_sentence_indices': indices, 'fts5_bm25': float(score)})
    return evidence


def retrieve_claims(db_path, claims_path, output_path, *, top_k=3):
    """Read only model inputs; never open gold.jsonl or select parameters from labels."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if top_k < 1:
        raise ValueError('top_k must be positive')
    temporary = output_path.with_name(output_path.name + '.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids = set()
    try:
        with closing(sqlite3.connect(f'file:{Path(db_path).resolve().as_posix()}?mode=ro', uri=True)) as con:
            if con.execute("SELECT value FROM metadata WHERE key='retrieval'").fetchone() != (
                'fts5_bm25_title3_body1',
            ):
                raise ValueError('FEVER index metadata mismatch')
            with Path(claims_path).open(encoding='utf-8') as inputs, temporary.open('w', encoding='utf-8') as output:
                for line in inputs:
                    row = json.loads(line)
                    if set(row) != {'id', 'claim'} or row['id'] in ids or not isinstance(row['claim'], str):
                        raise ValueError('Expected unique, claim-only FEVER input')
                    ids.add(row['id'])
                    context = {'id': row['id'], 'claim': row['claim'],
                               'evidence': retrieve_one(con, row['claim'], top_k)}
                    output.write(json.dumps(context, ensure_ascii=False, sort_keys=True) + '\n')
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {'claims': len(ids), 'contexts_sha256': file_sha256(output_path)}
