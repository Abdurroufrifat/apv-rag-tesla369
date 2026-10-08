import pytest
from analyze_guard_tradeoffs import compare


def test_correct_and_wrong_answers_withheld_are_counted_separately():
    result = compare([('Supported', None, 'Supported'),
                      ('Refuted', None, 'Supported'),
                      ('Supported', 'Supported', 'Supported'),
                      (None, None, 'Refuted')])
    assert result['withheld_correct'] == 1
    assert result['withheld_incorrect'] == 1
    assert result['before_accuracy_all'] == .5
    assert result['after_accuracy_all'] == .25
    assert result['after_coverage'] == .25


def test_no_answer_has_no_covered_accuracy():
    result = compare([(None, None, 'Supported')])
    assert result['after_covered_accuracy'] is None


@pytest.mark.parametrize('before,after', [(None, 'Supported'), ('Supported', 'Refuted')])
def test_guard_cannot_revive_or_relabel(before, after):
    with pytest.raises(ValueError):
        compare([(before, after, 'Supported')])
