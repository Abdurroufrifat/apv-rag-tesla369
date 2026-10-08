import copy

import numpy as np
import pytest

from apv_rag.fever_calibration import apply_temperature, fit_temperature, partition_claims


def test_temperature_adjusts_confidence_and_preserves_verdicts():
    p = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0, 0, 1]])
    result = apply_temperature(p, 3)
    np.testing.assert_array_equal(result.argmax(1), p.argmax(1))
    np.testing.assert_allclose(result.sum(1), 1)
    assert result[0, 0] < p[0, 0]
    np.testing.assert_allclose(apply_temperature(p, 1), p, atol=1e-11)
    with pytest.raises(ValueError):
        apply_temperature(p, 0)
    with pytest.raises(ValueError):
        apply_temperature([[0.8, 0.8, 0]], 2)


def test_fit_uses_only_development_scores_and_reduces_overconfident_nll():
    p = [[0.98, 0.01, 0.01]] * 6
    result = fit_temperature(p, [0, 0, 1, 1, 2, 2])
    assert result['temperature'] > 1
    assert result['development_nll_after'] < result['development_nll_before']
    assert result['development_claims'] == 6
    with pytest.raises(ValueError):
        fit_temperature(p, [0])


def test_partition_excludes_ids_and_text_and_ignores_labels():
    rows = [{'id': i, 'claim': f'Claim {i}', 'label': 'SUPPORTS', 'evidence': []}
            for i in range(15)]
    rows[14]['claim'] = '  CLAIM   1 '
    dev, confirm = partition_claims(rows, exclude_ids={0}, exclude_texts={'claim 1'},
                                    development_count=4, confirmation_count=3)
    ids = {r['id'] for r in dev + confirm}
    assert len(ids) == 7
    assert ids.isdisjoint({0, 1, 14})
    assert {r['id'] for r in dev}.isdisjoint({r['id'] for r in confirm})
    changed = copy.deepcopy(rows)
    for row in changed:
        row['label'] = 'REFUTES'
        row['evidence'] = [[[1, 1, 'Different_page', 0]]]
    a, b = partition_claims(changed, exclude_ids={0}, exclude_texts={'claim 1'},
                            development_count=4, confirmation_count=3)
    assert [r['id'] for r in a + b] == [r['id'] for r in dev + confirm]


def test_confirmation_reports_confidence_changes_without_refitting():
    from apv_rag.fever_calibration import compare_confirmation
    from apv_rag.fever_nli import predict_context

    predictions = []
    for i, scores in enumerate(([0.05, 0.9, 0.05], [0.9, 0.05, 0.05], [0.05, 0.05, 0.9])):
        row = {'id': i, 'claim': f'Claim {i}', 'evidence': [
            {'id': f'Page_{i}', 'text': 'Evidence.', 'selected_sentence_indices': [0],
             'fts5_bm25': -1.0}]}
        predictions.append(predict_context(row, [scores]))
    gold = [{'id': i, 'label': label, 'evidence': [[[1, 1, f'Page_{i}', 0]]]}
            for i, label in enumerate(('REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO'))]
    report, transformed = compare_confirmation(predictions, gold, {'temperature': 3})
    assert report['raw_accuracy'] == report['calibrated_accuracy'] == pytest.approx(1 / 3)
    assert report['argmax_changed'] == 0
    assert report['calibrated_nll'] < report['raw_nll']
    assert [r['id'] for r in transformed] == [r['id'] for r in predictions]
    with pytest.raises(ValueError, match='alignment'):
        compare_confirmation(predictions, list(reversed(gold)), {'temperature': 3})


def test_completed_inference_stage_replays_without_model_and_rejects_changed_context(tmp_path):
    import json
    from pathlib import Path

    from apv_rag.fever_nli import predict_context
    from apv_rag.splits import sha256, write_json_atomic
    from run_fever_calibration import infer_cohort

    source = tmp_path / 'source'
    (source / 'development').mkdir(parents=True)
    claims = [{'id': 10, 'claim': 'A claim'}]
    input_file = source / 'development/model_inputs.jsonl'
    input_file.write_text(json.dumps(claims[0]) + '\n')
    output = tmp_path / 'output'
    stage = output / 'development_nli'
    stage.mkdir(parents=True)
    identity = {'test_fixture': 'completed frozen outputs'}
    write_json_atomic(stage / 'input_manifest.json', dict(identity, cohort='development',
                       model_inputs_sha256=sha256(input_file)))
    row = dict(claims[0], evidence=[])
    contexts = stage / 'contexts.jsonl'
    contexts.write_text(json.dumps(row) + '\n')
    write_json_atomic(stage / 'retrieval_manifest.json', {'contexts.jsonl': sha256(contexts)})
    pred = predict_context(row, [])
    write_json_atomic(stage / 'predictions.json', [pred])
    write_json_atomic(stage / 'output_manifest.json', {
        n: sha256(stage / n) for n in ('input_manifest.json', 'predictions.json')})
    result = infer_cohort('development', claims, source, output, Path('/absent-index'),
                          identity, Path('/absent-model'))
    assert result == [pred]
    contexts.write_text(json.dumps(dict(row, claim='Changed claim')) + '\n')
    with pytest.raises(ValueError, match='checksum'):
        infer_cohort('development', claims, source, output, Path('/absent-index'),
                     identity, Path('/absent-model'))
