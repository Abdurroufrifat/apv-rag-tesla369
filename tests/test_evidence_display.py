import copy

import pytest

from apv_rag.evidence_display import build_evidence_display, verify_evidence_display


CORPUS = {'42': {'doc_id': 42, 'title': 'Tesla experiment',
                 'abstract': ['Tesla built an electric motor.']}}


def row(**changes):
    evidence = [{'id': 42, 'text': 'Tesla built an electric motor.',
                 'selected_sentence_indices': [0]}]
    return {'claim': 'Tesla built an electric motor.', 'claim_id': 'c1',
            'retrieved_evidence': evidence, 'evidence': evidence,
            'candidate_label': 'Supported', 'generated_explanation':
            'The nonexistent archive https://fake.example proves it [999].', **changes}


def test_bound_candidate_gets_exact_cited_excerpts_without_freeform_claims():
    display = build_evidence_display(row(), CORPUS)
    assert display['status'] == 'machine_candidate'
    assert display['candidate_label'] == 'Supported'
    assert display['items'][0]['citation_id'] == '42'
    assert display['items'][0]['quote'] == CORPUS['42']['abstract'][0]
    assert 'fake.example' not in str(display)
    assert [item['citation_id'] for item in display['items']] == ['42']
    assert display['support_relationship_verified'] is False
    assert verify_evidence_display(display, row(), CORPUS)


@pytest.mark.parametrize('change', ['text', 'id', 'indices', 'collapsed_context'])
def test_unbound_context_has_no_displayed_candidate_or_quotes(change):
    record = copy.deepcopy(row())
    if change == 'text':
        record['retrieved_evidence'][0]['text'] = 'Verified archive: Tesla secret 369.'
    elif change == 'id':
        record['retrieved_evidence'][0]['id'] = 99
    elif change == 'indices':
        record['retrieved_evidence'][0]['selected_sentence_indices'] = [1]
    else:
        record['evidence'] = []
    display = build_evidence_display(record, CORPUS)
    assert display['status'] == 'abstain'
    assert display['candidate_label'] is None
    assert display['items'] == []


def test_upstream_abstention_cannot_be_revived_by_bound_evidence():
    display = build_evidence_display(row(candidate_label=None), CORPUS)
    assert display['status'] == 'abstain'
    assert display['items'] == []
    assert display['candidate_label'] is None


def test_nei_is_not_an_evidence_absence_proof():
    display = build_evidence_display(row(candidate_label='Not Enough Evidence'), CORPUS)
    assert display['candidate_label'] == 'Not Enough Evidence'
    assert display['support_relationship_verified'] is False
    assert 'absence' in display['qualification']


def test_invalid_label_is_not_accepted():
    display = build_evidence_display(row(candidate_label='Authenticated Tesla quote'), CORPUS)
    assert display['status'] == 'abstain'


@pytest.mark.parametrize('field,value', [('quote', 'fabricated text'),
    ('citation_id', '99'), ('selected_sentence_indices', [8]),
    ('source_doc_sha256', '0' * 64)])
def test_display_tampering_is_detected(field, value):
    display = build_evidence_display(row(), CORPUS)
    display['items'][0][field] = value
    with pytest.raises(ValueError, match='display'):
        verify_evidence_display(display, row(), CORPUS)


def test_change_in_corpus_text_is_detected():
    display = build_evidence_display(row(), CORPUS)
    corpus = copy.deepcopy(CORPUS)
    corpus['42']['abstract'][0] = 'Tesla never built this motor.'
    with pytest.raises(ValueError, match='display'):
        verify_evidence_display(display, row(), corpus)


def test_near_duplicate_and_literal_clip_are_bound_to_existing_snapshot_selection():
    record = row()
    record['retrieved_evidence'][0]['text'] = 'Tesla built an electric'
    record['retrieved_evidence'].append(copy.deepcopy(record['retrieved_evidence'][0]))
    record['evidence'] = [record['retrieved_evidence'][0]]
    display = build_evidence_display(record, CORPUS)
    assert len(display['items']) == 1
    assert display['items'][0]['quote'] == 'Tesla built an electric'


def test_audit_rejects_rehashed_but_fabricated_display(tmp_path, monkeypatch):
    import json
    import audit_evidence_display as audit
    source = row(cohort='scifact', policy='no_gate', condition='original')
    identity = {'test_fixture': True}
    monkeypatch.setattr(audit, 'inputs', lambda: (
        {'scifact:c1:original:no_gate': source}, {'scifact': CORPUS}, identity))
    assert audit.build(tmp_path)['totals']['cited_excerpts'] == 1
    path = tmp_path / 'displays.jsonl'
    record = json.loads(path.read_text(encoding='utf-8'))
    record['display']['items'][0]['quote'] = 'A fabricated Tesla statement.'
    path.write_text(json.dumps(record) + '\n', encoding='utf-8')
    manifest_path = tmp_path / 'audit_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files']['displays.jsonl'] = audit.digest(path)
    audit.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='do not replay'):
        audit.verify(tmp_path)


def test_gold_and_generator_text_cannot_change_display():
    first = row(true_label='Supported')
    second = row(true_label='Refuted', generated_explanation='Completely different rationale [777].')
    assert build_evidence_display(first, CORPUS) == build_evidence_display(second, CORPUS)


def test_display_edit_cannot_mutate_original_selection_indices():
    original = row()
    display = build_evidence_display(original, CORPUS)
    display['items'][0]['selected_sentence_indices'].append(99)
    assert original['evidence'][0]['selected_sentence_indices'] == [0]
