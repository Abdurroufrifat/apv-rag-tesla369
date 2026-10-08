import copy

import pytest

from apv_rag.frozen_component_ablation import masked_probability, summarize


def model():
    coefficients = [0.0] * 13
    coefficients[0], coefficients[9] = 2.0, -1.0
    return {'columns': list(range(13)), 'mean': [0.0] * 13, 'scale': [1.0] * 13,
            'coefficients': [coefficients], 'intercept': [0.0]}


def test_mean_mask_removes_only_selected_logit_contribution_without_mutation():
    features = [1.0] * 13
    head = model()
    before = copy.deepcopy((features, head))
    assert masked_probability(features, head, 'full') > 0.5
    assert masked_probability(features, head, 'without_nli') < 0.5
    assert masked_probability(features, head, 'without_embedding') > 0.5
    assert masked_probability(features, head, 'without_both') == 0.5
    assert (features, head) == before


def test_invented_metadata_ablation_and_invalid_features_are_refused():
    with pytest.raises(ValueError):
        masked_probability([0.0] * 13, model(), 'without_date')
    with pytest.raises(ValueError):
        masked_probability([float('nan')] * 13, model(), 'full')


def test_abstention_metrics_include_missed_answers_and_null_covered_accuracy():
    result = summarize([None, None], ['Supported', 'Refuted'])
    assert result['correct'] == 0 and result['all_query_accuracy'] == 0.0
    assert result['coverage'] == 0.0 and result['accepted_accuracy'] is None
    result = summarize(['Supported', None], ['Supported', 'Refuted'])
    assert result['all_query_accuracy'] == 0.5 and result['accepted_accuracy'] == 1.0
    with pytest.raises(ValueError):
        summarize(['Supported'], ['Refuted', 'Supported'])
