import numpy as np
import pytest

from apv_rag.source_pipeline import PipelineConfig, verify_claim


def doc(identifier, text="Tesla electric motor evidence", **kw):
    return dict(id=identifier, text=text, language="en", **kw)


def scorer(claim, texts):
    return [[0.05, 0.9, 0.05] for _ in texts]


def test_missing_and_language_mismatch_abstain_without_scoring():
    def forbidden(*args):
        raise AssertionError("Scorer should not run")

    assert verify_claim("Tesla motor", "en", [], forbidden)["status"] == "abstain"
    r = verify_claim("Tesla motor", "en", [doc("a") | {"language": "sr"}], forbidden)
    assert r["abstention_reasons"][0] == "no_compatible_retrieved_evidence"


def test_duplicate_repetition_cannot_supply_independent_evidence():
    docs = [doc("a"), doc("b", "Tesla electric generator evidence")]
    original = verify_claim("Tesla electric", "en", docs, scorer)
    copied = verify_claim("Tesla electric", "en", docs + [doc(str(i)) for i in range(20)], scorer)
    assert original["status"] == copied["status"] == "machine_candidate"
    assert np.allclose(original["stance_scores"], copied["stance_scores"])
    assert copied["provenance_components"] == 2
    assert verify_claim("Tesla electric", "en", [doc("a"), doc("b")], scorer)["status"] == "abstain"


def test_transitive_family_and_text_dependencies():
    docs = [
        doc("a", provenance_family_id="x"),
        doc("b", "Tesla generator evidence", provenance_family_id="x"),
        doc("c", "Tesla generator evidence", provenance_family_id="y"),
    ]
    r = verify_claim("Tesla evidence", "en", docs, scorer)
    assert r["provenance_components"] == 1
    assert r["status"] == "abstain"


def test_weights_primary_requirement_and_invalid_inputs():
    docs = [doc("a", source_rank=1), doc("b", "Tesla generator evidence", source_rank=5)]
    r = verify_claim("Tesla evidence", "en", docs, scorer, PipelineConfig(require_primary=True))
    assert "missing_declared_primary_source" in r["abstention_reasons"]
    with pytest.raises(ValueError):
        verify_claim("Tesla", "en", [doc("a", source_rank=0)], scorer)
    with pytest.raises(ValueError):
        verify_claim("Tesla", "en", docs, lambda c, t: [[1, 0, 0]])


def test_source_weight_ablation_and_stance_abstention():
    docs = [doc("a", source_rank=1), doc("b", "Tesla generator evidence", source_rank=5)]

    def conflicting(claim, texts):
        return [[0.05, 0.9, 0.05], [0.9, 0.05, 0.05]]

    weighted = verify_claim("Tesla evidence", "en", docs, conflicting)
    unweighted = verify_claim(
        "Tesla evidence", "en", docs, conflicting, PipelineConfig(weight_sources=False)
    )
    assert weighted["status"] == "machine_candidate"
    assert unweighted["status"] == "abstain"
    assert "ambiguous_stance" in unweighted["abstention_reasons"]
    neutral = verify_claim(
        "Tesla evidence", "en", docs, lambda claim, texts: [[0.05, 0.05, 0.9] for _ in texts]
    )
    assert "neutral_dominant" in neutral["abstention_reasons"]


def test_shared_index_matches_default_retrieval():
    from apv_rag.retrieval import BM25Index

    docs = [doc("a"), doc("b", "Tesla generator evidence")]
    shared = BM25Index([d["text"] for d in docs])
    baseline = verify_claim("Tesla evidence", "en", docs, scorer)
    reused = verify_claim("Tesla evidence", "en", docs, scorer, retrieval_index=shared)
    assert baseline == reused
    with pytest.raises(ValueError):
        verify_claim("Tesla evidence", "en", docs, scorer, retrieval_index=BM25Index([]))
