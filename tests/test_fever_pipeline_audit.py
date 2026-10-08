"""Audit must catch edited model outputs even if export hashes are regenerated."""
import copy
import json
import sqlite3

import pytest

from apv_rag.fresh_pipeline import execute_policies, feature_entry
from apv_rag.fever_pipeline import answer_score
from apv_rag.splits import write_json_atomic
from run_fresh_pipeline import response_key
from run_gated_generation import qwen_prompt


def fixture(folder):
    claim = {'id': 1, 'claim': 'A is an island.'}
    doc = {'id': 'A', 'text': 'A is an island.', 'selected_sentence_indices': [0], 'fts5_bm25': -1.0}
    prepared = {'claim': claim['claim'], 'shown_claim': claim['claim'],
                'retrieved_evidence': [doc], 'evidence': [doc]}
    feature = feature_entry(claim['claim'], [doc], [[0.1, 0.8, 0.1]], [0.9])
    head = {'columns': [0], 'mean': [0], 'scale': [1], 'coefficients': [[0]], 'intercept': [2]}
    heads = {n: head for n in ('nli', 'embedding', 'combined')}
    used = {}
    def generate(kind, question):
        response = {'answer': 'Supported' if kind == 'verdict' else 'The evidence calls A an island.',
                    'prompt': qwen_prompt(question), 'prompt_tokens': 40}
        used[response_key(kind, response['prompt'])] = dict(response, kind=kind, origin='current_run_live_or_resume')
        return response
    rows = execute_policies(prepared, feature, heads, generate)
    for row in rows:
        row.update(claim_id=1, claim=claim['claim'], shown_claim=claim['claim'], retrieved_evidence=[doc],
                   context_sha256=feature['context_sha256'], score=answer_score(row, feature))
    for name, value in [('prepared_contexts.json', {'1': prepared}), ('feature_cache.json', {'1': feature}),
                        ('predictions.json', rows), ('responses_used.json', used)]:
        write_json_atomic(folder / name, value)
    (folder / 'retrieved_contexts.jsonl').write_text(json.dumps(dict(claim,evidence=[doc]))+'\n')
    with sqlite3.connect(folder / 'generation_cache.sqlite') as con:
        con.execute('CREATE TABLE answers(key TEXT PRIMARY KEY,response TEXT)')
        for key, record in used.items():
            con.execute('INSERT INTO answers VALUES (?,?)', (key,json.dumps({n:record[n] for n in ('answer','prompt','prompt_tokens')})))
    return [claim], heads


def test_audit_replays_controller_and_rejects_edited_scores(tmp_path):
    from verify_fever_pipeline import verify_stage_data
    claims, heads = fixture(tmp_path)
    result = verify_stage_data(tmp_path, claims, heads)
    assert result['policy_records'] == 4
    assert result['responses'] == 2
    rows = json.loads((tmp_path/'predictions.json').read_text())
    rows[0]['score'] = 0.99
    write_json_atomic(tmp_path/'predictions.json', rows)
    with pytest.raises(ValueError, match='Controller'):
        verify_stage_data(tmp_path, claims, heads)


def test_audit_rejects_generation_cache_answer_conflict(tmp_path):
    from verify_fever_pipeline import verify_stage_data
    claims, heads = fixture(tmp_path)
    with sqlite3.connect(tmp_path/'generation_cache.sqlite') as con:
        con.execute("UPDATE answers SET response='{}'")
    with pytest.raises(ValueError, match='cache'):
        verify_stage_data(tmp_path, claims, heads)
