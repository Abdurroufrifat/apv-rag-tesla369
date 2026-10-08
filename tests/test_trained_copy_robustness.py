import numpy as np

from apv_rag.trained_copy_robustness import training_variants, model_records


def test_derivatives_preserve_parent_label_and_total_weight_without_gold_inputs():
    original = {"claim": "A", "label": "Refuted", "justification": "SECRET",
                "questions": [{"answers": [{"answer": "No", "source_url": "https://a.org/one"}]}]}
    records, labels, weights, parents = training_variants([original])
    assert labels == ["Refuted"] * 3 and parents == [0] * 3
    assert [len(row["questions"][0]["answers"]) for row in records] == [1, 6, 11]
    assert all("label" not in row and "justification" not in row for row in records)
    assert original["label"] == "Refuted"
    np.testing.assert_allclose(weights.sum(), 1)
    assert model_records([original]) == [records[0]]
