from apv_rag.input_numeric_integrity import numeric_provenance


def test_claim_references_are_distinct_from_invented_numbers():
    claim = "The claimed temperature was 20 degrees."
    evidence = [{"text": "Temperature was not measured."}]
    result = numeric_provenance("The claimed 20 degrees is unestablished [3].", claim, evidence)
    assert result["claim_only_values"] == ["20"]
    assert result["absent_from_inputs"] == []
    assert numeric_provenance("Temperature was 0.005.", claim, evidence)["absent_from_inputs"] == [
        "0.005"
    ]
