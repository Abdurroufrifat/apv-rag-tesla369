import copy

import numpy as np
import pytest

from apv_rag.fever_pipeline import fit_correctness, correctness_probability, summarize_policy


def test_correctness_fit_targets_final_answer_success():
    fit = fit_correctness([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
    assert fit['accepted_development_answers'] == 4
    assert correctness_probability(0.9, fit) > correctness_probability(0.1, fit)
    # A one-class fit is explicit and uses predeclared Beta(1,1) smoothing.
    constant = fit_correctness([0.8, 0.9], [True, True])
    assert correctness_probability(0.2, constant) == 0.75
    absent = fit_correctness([], [])
    assert correctness_probability(0.8, absent) is None
    with pytest.raises(ValueError):
        fit_correctness([1.2], [True])


def test_abstention_is_not_an_nei_prediction_and_confidence_is_correctness():
    rows = [{'claim_id': 1, 'candidate_label': 'Supported', 'score': 0.9},
            {'claim_id': 2, 'candidate_label': None, 'score': None},
            {'claim_id': 3, 'candidate_label': 'Not Enough Evidence', 'score': 0.2}]
    gold = [{'id': 1, 'label': 'SUPPORTS'}, {'id': 2, 'label': 'NOT ENOUGH INFO'},
            {'id': 3, 'label': 'REFUTES'}]
    fit = fit_correctness([0.9, 0.1], [True, False])
    result = summarize_policy(rows, gold, fit)
    assert result['coverage'] == pytest.approx(2/3)
    assert result['covered_accuracy'] == 0.5
    assert result['accuracy_all_claims_abstentions_as_errors'] == pytest.approx(1/3)
    assert result['correctness_calibration']['answers'] == 2
    assert result['correctness_calibration']['raw_score_brier'] == pytest.approx((0.01+0.04)/2)
    broken = copy.deepcopy(rows)
    broken[0]['claim_id'] = 9
    with pytest.raises(ValueError, match='alignment'):
        summarize_policy(broken, gold, fit)


def test_correctness_feature_scores_the_generated_label_not_nli_argmax():
    from apv_rag.fever_pipeline import answer_score
    feature = {'scores_cen': [[0.7, 0.2, 0.1]]}
    assert answer_score({'candidate_label': 'Supported'}, feature) == pytest.approx(0.2)
    assert answer_score({'candidate_label': 'Refuted'}, feature) == pytest.approx(0.7)
    assert answer_score({'candidate_label': None}, None) is None
    with pytest.raises(ValueError):
        fit_correctness([0.4], [0.5])


def test_all_gate_policies_share_context_responses_and_gate_before_generation():
    from apv_rag.fresh_pipeline import feature_entry, execute_policies
    from apv_rag.fever_pipeline import answer_score

    doc = {'id': 'Page_A', 'text': 'A is an island.', 'selected_sentence_indices': [0]}
    prepared = {'claim': 'A is an island.', 'shown_claim': 'A is an island.',
                'retrieved_evidence': [doc], 'evidence': [doc]}
    entry = feature_entry(prepared['shown_claim'], [doc], [[0.1, 0.8, 0.1]], [0.9])
    def head(intercept):
        return {'columns': [0], 'mean': [0], 'scale': [1], 'coefficients': [[0]], 'intercept': [intercept]}
    heads = {'nli': head(-2), 'embedding': head(2), 'combined': head(2)}
    requests = []
    def generate(kind, question):
        requests.append((kind, question))
        return {'answer': 'Supported' if kind == 'verdict' else 'The evidence states it is an island.',
                'prompt': question, 'prompt_tokens': 40}
    rows = execute_policies(prepared, entry, heads, generate)
    rejected = next(r for r in rows if r['policy'] == 'nli')
    assert rejected['generation_requests'] == []
    assert rejected['candidate_label'] is None
    accepted = [r for r in rows if r['candidate_label'] is not None]
    assert len(accepted) == 3
    assert len({q for kind, q in requests if kind == 'verdict'}) == 1
    assert all(answer_score(r, entry) == pytest.approx(0.8) for r in accepted)
    with pytest.raises(ValueError, match='binding'):
        execute_policies(dict(prepared, shown_claim='Changed claim'), entry, heads, generate)


def test_cached_pipeline_stage_replays_without_loading_neural_models(tmp_path):
    import json
    from apv_rag.splits import sha256, write_json_atomic
    from run_fever_pipeline import run_stage

    source = tmp_path / 'source'
    (source / 'development').mkdir(parents=True)
    inputs = source / 'development/model_inputs.jsonl'
    inputs.write_text('{"id":1,"claim":"Claim"}\n')
    out = tmp_path / 'output'
    folder = out / 'development'
    folder.mkdir(parents=True)
    identity = {'test_fixture': True}
    write_json_atomic(folder / 'input_manifest.json', dict(identity, cohort='development', claims_sha256=sha256(inputs)))
    write_json_atomic(folder / 'predictions.json', [{'claim_id': 1, 'policy': 'no_gate', 'candidate_label': None}])
    write_json_atomic(folder / 'output_manifest.json', {n: sha256(folder/n)
        for n in ('input_manifest.json','predictions.json')})
    rows = run_stage('development', [{'id': 1, 'claim': 'Claim'}], source, out, None, {}, identity)
    assert rows[0]['candidate_label'] is None
    (folder / 'predictions.json').write_text('[]\n')
    with pytest.raises(ValueError, match='checksum'):
        run_stage('development', [{'id': 1, 'claim': 'Claim'}], source, out, None, {}, identity)
