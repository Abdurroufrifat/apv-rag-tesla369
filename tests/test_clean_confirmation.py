import math

import pytest

from run_clean_confirmation import compare_values, pinned_requirements, require_empty


def test_comparison_preserves_discrete_answers_and_reports_numeric_drift():
    old = {'label': 'Support', 'score': 0.7, 'ids': [1, 2]}
    assert compare_values({'label': 'Support', 'score': 0.700001, 'ids': [1, 2]}, old) == []
    assert compare_values(dict(old, label='Refute'), old)
    assert compare_values(dict(old, score=0.71), old)
    assert compare_values(dict(old, ids=[2, 1]), old)


@pytest.mark.parametrize('value', [math.nan, math.inf, True])
def test_invalid_numeric_values_never_match(value):
    assert compare_values({'score': value}, {'score': 1.0})


def test_lock_excludes_project_and_preserves_cpu_build_pin():
    records = [{'name': 'apv-rag', 'version': '0.1.0'},
               {'name': 'torch', 'version': '2.14.1+cpu'}]
    assert pinned_requirements(records) == 'torch==2.14.1+cpu\n'
    with pytest.raises(ValueError):
        pinned_requirements([{'name': 'torch', 'version': '2; invalid'}])


def test_existing_results_are_refused(tmp_path):
    require_empty(tmp_path)
    (tmp_path / 'feature_cache.json').write_text('{}')
    with pytest.raises(ValueError):
        require_empty(tmp_path)
