import copy

from apv_rag.multilingual_source_guard import admit_context, execute_language_bound


def fixture():
    en = {"key": "en/test.jsonl:0", "file": "en/test.jsonl", "claim": "An island.",
          "language": "en", "retrieved_evidence": [{"id": "EN", "text": "An island.", "score": 1.0}]}
    es = {"key": "es/test.jsonl:0", "file": "es/test.jsonl", "claim": "Una isla.",
          "language": "es", "retrieved_evidence": [{"id": "ES", "text": "Una isla.", "score": 1.0}]}
    prepared = {"claim": "An island.", "shown_claim": "An island.",
                "retrieved_evidence": [{"id": "EN", "text": "An island.", "score": 1.0}],
                "evidence": [{"id": "EN", "text": "An island.", "score": 1.0}]}
    corpora = {"en/test.jsonl": [{"id": "EN", "text": "An island."}],
               "es/test.jsonl": [{"id": "ES", "text": "Una isla."}]}
    return en, es, prepared, corpora


def test_original_and_clipped_source_bind_to_declared_pool():
    en, _, prepared, corpora = fixture()
    assert admit_context(en, prepared, corpora)
    clipped = copy.deepcopy(prepared)
    clipped["retrieved_evidence"][0]["text"] = "An is"
    clipped["evidence"][0]["text"] = "An is"
    assert admit_context(en, clipped, corpora)


def test_same_claim_foreign_pool_is_refused_before_generation():
    en, es, prepared, corpora = fixture()
    swapped = copy.deepcopy(prepared)
    swapped["retrieved_evidence"] = copy.deepcopy(es["retrieved_evidence"])
    swapped["evidence"] = copy.deepcopy(es["retrieved_evidence"])
    assert not admit_context(en, swapped, corpora)
    calls = []
    def generate(kind, question):
        calls.append((kind, question))
        raise AssertionError("model must not be called")
    rows = execute_language_bound(en, swapped, corpora, None, None, {}, generate)
    assert calls == []
    assert [r["policy"] for r in rows] == ["no_gate", "nli", "embedding", "combined"]
    assert all(r["candidate_label"] is None and not r["generation_requests"]
               and r["reasons"] == ["source_language_pool_mismatch"]
               and r["evidence"] == [] for r in rows)


def test_changed_claim_id_text_and_declared_language_fail():
    en, _, prepared, corpora = fixture()
    altered = copy.deepcopy(prepared)
    altered["claim"] = "Different."
    assert not admit_context(en, altered, corpora)
    altered = copy.deepcopy(prepared)
    altered["retrieved_evidence"][0]["id"] = "ES"
    altered["evidence"][0]["id"] = "ES"
    assert not admit_context(en, altered, corpora)
    altered = copy.deepcopy(en)
    altered["language"] = "es"
    assert not admit_context(altered, prepared, corpora)
    altered = copy.deepcopy(prepared)
    altered["retrieved_evidence"][0]["text"] = "A changed line."
    altered["evidence"][0]["text"] = "A changed line."
    assert not admit_context(en, altered, corpora)


def test_admitted_context_delegates_to_unchanged_controller(monkeypatch):
    en, _, prepared, corpora = fixture()
    called = []
    def controller(*args):
        called.append(args)
        return [{"policy": "no_gate", "candidate_label": "Supported"}]
    monkeypatch.setattr("apv_rag.multilingual_source_guard.execute", controller)
    rows = execute_language_bound(en, prepared, corpora, "english", "multi", {}, lambda *_: None)
    assert len(called) == 1
    assert rows == [{"policy": "no_gate", "candidate_label": "Supported",
                     "source_language_pool_bound": True}]


def test_exact_shared_excerpt_in_two_pools_is_admitted_in_each():
    en, es, prepared, corpora = fixture()
    shared = {"id": "SHARED", "text": "2020"}
    corpora["en/test.jsonl"].append(shared)
    corpora["es/test.jsonl"].append(shared)
    en["retrieved_evidence"] = [{**shared, "score": 1.0}]
    es["retrieved_evidence"] = [{**shared, "score": 1.0}]
    prepared["retrieved_evidence"] = [{**shared, "score": 1.0}]
    prepared["evidence"] = [{**shared, "score": 1.0}]
    assert admit_context(en, prepared, corpora)
