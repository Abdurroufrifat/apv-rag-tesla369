import pytest

from apv_rag.scifact import connected_groups, normalized_claim, target_label


def test_target_mapping_and_conflicts():
    assert target_label({"evidence": {}}) == "Not Enough Evidence"
    assert target_label({"evidence": {"1": [{"label": "SUPPORT"}]}}) == "Supported"
    assert target_label({"evidence": {"1": [{"label": "CONTRADICT"}]}}) == "Refuted"
    assert (
        target_label({"evidence": {"1": [{"label": "SUPPORT"}, {"label": "CONTRADICT"}]}}) is None
    )
    with pytest.raises(ValueError):
        target_label({"evidence": {"1": [{"label": "unexpected"}]}})


def test_transitive_groups_and_claim_normalization():
    groups = connected_groups([{"cited_doc_ids": x} for x in [[1], [1, 2], [2], [], [3]]])
    assert groups[0] == groups[1] == groups[2]
    assert len(set(groups)) == 3
    assert normalized_claim("Ａ Claim, TEST!") == normalized_claim("a claim test")
