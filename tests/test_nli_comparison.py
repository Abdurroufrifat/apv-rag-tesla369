import numpy as np
import pytest

from apv_rag.nli_comparison import nli_features, pool_corpus, semantic_ranking


def test_comparison_validator_rejects_a_modified_output(tmp_path):
    import json

    from apv_rag.nli_comparison import validate_comparison
    from apv_rag.splits import sha256

    summary = tmp_path / "comparison_summary.json"
    summary.write_text('{"official_dev_records_used": 0}', encoding="utf-8")
    (tmp_path / "output_manifest.json").write_text(
        json.dumps({summary.name: sha256(summary)}), encoding="utf-8"
    )
    summary.write_text('{"official_dev_records_used": 1}', encoding="utf-8")
    assert validate_comparison(tmp_path)


def test_pooling_preserves_contradiction_and_entailment_as_separate_features():
    features = nli_features([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    assert features.tolist() == pytest.approx([0.8, 0.8, 0.1, 0.45, 0.45, 0.1, 2])
    assert nli_features([]).tolist() == [0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        nli_features([[0.8, 0.8, 0.8]])


def test_dense_ranking_is_stable_and_excludes_nonpositive_matches():
    assert semantic_ranking(np.asarray([0.9, 0.9, -0.1]), 5) == [0, 1]


def test_corpus_excludes_labels_and_justifications_and_deduplicates_excerpts():
    row = {
        "label": "SECRET",
        "justification": "LEAK",
        "questions": [{"answers": [{"answer": "Evidence.", "source_url": "https://a.org"}]}],
    }
    documents = pool_corpus([row, row])
    assert documents == [{"text": "Evidence.", "source_url": "https://a.org"}]
