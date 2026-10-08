from apv_rag.generation_integrity import new_numeric_values


def test_changed_numbers_rejected_but_source_ids_ignored():
    evidence = [{"text": "The temperature was 20 degrees Celsius."}]
    assert new_numeric_values("Temperature was 0.005 degrees [123].", evidence) == ["0.005"]
    assert new_numeric_values("Temperature was 20 degrees [123].", evidence) == []
