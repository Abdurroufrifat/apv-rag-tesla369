from apv_rag.grounded_generation import evidence_decision, explanation_passes


def test_empty_unclear_and_conflicting_evidence_abstain():
    assert evidence_decision([])["selected"] is None
    assert evidence_decision([[0.1, 0.1, 0.8]])["selected"] is None
    assert evidence_decision([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])["selected"] is None
    assert evidence_decision([[0.9, 0.05, 0.05]])["label"] == "Refuted"


def test_generated_explanations_require_entailment_and_nonempty_text():
    assert explanation_passes("Summary", [0.05, 0.9, 0.05])
    assert not explanation_passes("Summary", [0.9, 0.05, 0.05])
    assert not explanation_passes("", [0.05, 0.9, 0.05])
