import pytest

from apv_rag.generative_rag import make_prompt, parse_answer


def test_citations_and_format_are_checked_without_claiming_entailment():
    assert parse_answer("Supported | Found in [12].", [12])["status"] == "machine_candidate"
    assert parse_answer("Supported | Found in [13].", [12])["status"] == "abstain"
    assert parse_answer("Refuted | No citation.", [12])["status"] == "abstain"
    assert parse_answer("Supported", [12])["status"] == "abstain"
    assert (
        parse_answer("Not Enough Evidence | No evidence.", [])["candidate_label"]
        == "Not Enough Evidence"
    )


def test_prompt_has_no_gold_fields_and_rejects_duplicate_ids():
    prompt = make_prompt("Claim", [{"id": 12, "text": "Evidence"}])
    assert "[12] Evidence" in prompt
    with pytest.raises(ValueError):
        make_prompt("Claim", [{"id": 12, "text": "A"}, {"id": 12, "text": "B"}])
