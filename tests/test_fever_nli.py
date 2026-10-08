import copy

import pytest

from apv_rag.fever_nli import validate_contexts, predict_context, score_predictions


def context():
    return {'id': 1, 'claim': 'A is an island.', 'evidence': [
        {'id': 'A', 'text': 'A is an island.', 'selected_sentence_indices': [0],
         'fts5_bm25': -1.0}]}


def test_contexts_bind_claims_and_reject_label_leakage():
    row = context()
    claims = [{'id': 1, 'claim': row['claim']}]
    assert validate_contexts(claims, [row]) == [row]
    changed = copy.deepcopy(row)
    changed['claim'] = 'Other claim'
    with pytest.raises(ValueError, match='alignment'):
        validate_contexts(claims, [changed])
    changed = copy.deepcopy(row)
    changed['label'] = 'SUPPORTS'
    with pytest.raises(ValueError, match='claim-only'):
        validate_contexts(claims, [changed])
    changed = copy.deepcopy(row)
    changed['evidence'][0]['selected_sentence_indices'] = [0, 0]
    with pytest.raises(ValueError, match='sentence'):
        validate_contexts(claims, [changed])


def test_predictions_use_fixed_mean_and_keep_evidence_alignment():
    row = context()
    row['evidence'].append({'id': 'B', 'text': 'B is a city.',
                            'selected_sentence_indices': [2], 'fts5_bm25': -0.5})
    pred = predict_context(row, [[0.1, 0.8, 0.1], [0.3, 0.5, 0.2]])
    assert pred['probabilities'] == pytest.approx([0.65, 0.2, 0.15])
    assert pred['predicted_label'] == 'Supported'
    assert pred['evidence'] == row['evidence']
    assert 'true_label' not in pred
    with pytest.raises(ValueError, match='alignment'):
        predict_context(row, [[0.1, 0.8, 0.1]])
    empty = {'id': 2, 'claim': 'Unknown', 'evidence': []}
    assert predict_context(empty, [])['probabilities'] == [0, 0, 1]


@pytest.mark.filterwarnings("ignore:F-score is ill-defined")
@pytest.mark.filterwarnings("ignore:A single label was found")
def test_scoring_checks_argmax_and_complete_gold_sentence_group():
    pred = predict_context(context(), [[0.1, 0.8, 0.1]])
    gold = [{'id': 1, 'label': 'SUPPORTS', 'evidence': [[[1, 2, 'A', 0]]]}]
    result = score_predictions([pred], gold)
    assert result['accuracy'] == 1
    assert result['retrieval']['complete_sentence_groups_found'] == 1
    # A page match without the required sentence must not pass sentence recall.
    gold[0]['evidence'][0].append([1, 3, 'A', 2])
    result = score_predictions([pred], gold)
    assert result['retrieval']['complete_page_groups_found'] == 1
    assert result['retrieval']['complete_sentence_groups_found'] == 0
    changed = copy.deepcopy(pred)
    changed['predicted_label'] = 'Refuted'
    with pytest.raises(ValueError, match='argmax'):
        score_predictions([changed], gold)
    with pytest.raises(ValueError, match='alignment'):
        score_predictions([pred], [{'id': 2, 'label': 'SUPPORTS', 'evidence': []}])


def test_scoring_runs_on_frozen_outputs_and_rejects_tampering(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path

    import score_fever_nli
    from apv_rag.fever_nli import load_claims, verify_prediction_files
    from apv_rag.splits import sha256, write_json_atomic

    project = Path(__file__).resolve().parents[1]
    source = tmp_path / 'data/external/fever/heldout_v1'
    source.mkdir(parents=True)
    for name in ('model_inputs.jsonl', 'selection_manifest.json', 'gold.jsonl'):
        shutil.copyfile(project / 'data/external/fever/heldout_v1' / name, source / name)
    claims = load_claims(source)
    rows = [dict(c, evidence=[]) for c in claims]
    retrieval = tmp_path / 'artifacts/fever_retrieval_v1'
    retrieval.mkdir(parents=True)
    contexts = retrieval / 'contexts.jsonl'
    contexts.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    write_json_atomic(retrieval / 'input_manifest.json', {'scope': 'test fixture'})
    run = tmp_path / 'artifacts/fever_nli_v1'
    run.mkdir()
    predictions = [predict_context(r, []) for r in rows]
    write_json_atomic(run / 'predictions.json', predictions)
    write_json_atomic(run / 'input_manifest.json', {
        'claims': 300, 'contexts_sha256': sha256(contexts),
        'retrieval_receipt_sha256': sha256(retrieval / 'input_manifest.json'),
        'model_inputs_sha256': sha256(source / 'model_inputs.jsonl')})
    write_json_atomic(run / 'output_manifest.json', {
        n: sha256(run / n) for n in ('input_manifest.json', 'predictions.json')})
    for name in ('scripts/score_fever_nli.py', 'src/apv_rag/fever_nli.py', 'src/apv_rag/metrics.py'):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, tmp_path / name)
    monkeypatch.setattr(score_fever_nli, 'ROOT', tmp_path)
    score_fever_nli.main()
    summary_path = tmp_path / 'artifacts/fever_nli_scoring_v1/summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['claims'] == 300
    assert summary['accuracy'] == pytest.approx(104 / 300)
    assert summary['retrieval']['verifiable_claims'] == 196
    assert summary['retrieval']['complete_sentence_groups_found'] == 0
    with pytest.raises(FileExistsError):
        score_fever_nli.main()
    predictions[0]['claim'] = 'Changed after inference'
    write_json_atomic(run / 'predictions.json', predictions)
    with pytest.raises(ValueError, match='checksum'):
        verify_prediction_files(run)


def test_softmax_rounding_does_not_break_prediction_replay():
    scores = [[0.10001, 0.80001, 0.10001]]
    pred = predict_context(context(), scores)
    assert pred == predict_context(context(), pred['nli_probabilities_cen'])
    assert pred['nli_probabilities_cen'] == scores
