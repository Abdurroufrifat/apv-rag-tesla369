import pytest

from apv_rag.sufficiency_proxy import proxy_labels, select_threshold


def test_proxy_target_includes_conflicting_evidence_without_inventing_annotations():
    assert proxy_labels(
        ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    ).tolist() == [1, 1, 0, 1]
    with pytest.raises(ValueError):
        proxy_labels(["unknown"])


def test_threshold_ties_choose_half_and_invalid_scores_fail():
    threshold, _ = select_threshold([0, 1], [0, 1])
    assert threshold == 0.5
    with pytest.raises(ValueError):
        select_threshold([0, 1], [float("nan"), 1])
