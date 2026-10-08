from apv_rag.own_excerpt import own_premises


def test_cached_runner_reproduces_original_and_preserves_inputs(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import pytest
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from apv_rag import own_excerpt
    from apv_rag.evidence_baseline import LABELS
    from apv_rag.heldout import cache_receipt
    from apv_rag.splits import sha256, write_json_atomic

    original = tmp_path / "artifacts/retrieval_nli_comparison"
    original.mkdir(parents=True)
    split = tmp_path / "data/processed/averitec/phase2b_split_v0_1"
    split.mkdir(parents=True)
    data = tmp_path / "data/external/averitec/official_7c62d1e"
    data.mkdir(parents=True)
    rows = [{"claim": f"claim {i}", "label": LABELS[i % 4]} for i in range(88)]
    source = data / "train.json"
    write_json_atomic(source, rows)
    ids = {"train": list(range(80)), "validation": list(range(80, 88))}
    for name, positions in ids.items():
        write_json_atomic(split / f"{name}_indices.json", positions)
    metadata = {
        "source_sha256": sha256(source),
        "models": {"nli": {}},
        "packages": {name: "test" for name in ("torch", "transformers", "scikit-learn", "numpy")},
        **{f"{name}_indices_sha256": sha256(split / f"{name}_indices.json") for name in ids},
    }
    write_json_atomic(original / "input_manifest.json", metadata)
    rng = np.random.default_rng(369)
    x_train, x_val, x_own = (
        rng.normal(size=(80, 7)),
        rng.normal(size=(8, 7)),
        rng.normal(size=(8, 7)),
    )
    y = [r["label"] for r in rows[:80]]
    model = CalibratedClassifierCV(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=369),
        ),
        method="sigmoid",
        cv=StratifiedKFold(5, shuffle=True, random_state=369),
    )
    model.fit(x_train, y)
    p = model.predict_proba(x_val)[:, [list(model.classes_).index(label) for label in LABELS]]
    predictions = []
    methods = {m: {"decision_exponent": 1} for m in ("bm25_nli", "dense_nli")}
    for method in methods:
        np.save(original / f"train_{method}_features.npy", x_train)
        np.save(original / f"validation_{method}_features.npy", x_val)
        predictions.extend(
            {
                "method": method,
                "seed": 369,
                "upstream_index": i,
                "true_label": rows[i]["label"],
                "probabilities": p[j].tolist(),
            }
            for j, i in enumerate(ids["validation"])
        )
    write_json_atomic(original / "predictions.json", predictions)
    write_json_atomic(
        original / "output_manifest.json", {path.name: sha256(path) for path in original.iterdir()}
    )
    before = {path.name: sha256(path) for path in original.iterdir()}
    frozen = tmp_path / "artifacts/frozen_heldout_protocol"
    frozen.mkdir()
    write_json_atomic(
        frozen / "protocol.json",
        {"methods": methods, "seeds": [369], "training_priors": [0.25] * 4},
    )
    output = tmp_path / "artifacts/own_excerpt_comparison"
    output.mkdir()
    np.save(output / "own_validation_features.npy", x_own)
    cache_receipt(output / "own_validation_features.npy", save=True)
    code = tmp_path / "src/apv_rag"
    code.mkdir(parents=True)
    (code / "nli_comparison.py").write_bytes(
        Path(own_excerpt.__file__).with_name("nli_comparison.py").read_bytes()
    )
    monkeypatch.setattr(own_excerpt, "version", lambda name: "test")
    monkeypatch.setattr(own_excerpt, "_model_files", lambda path: {})
    monkeypatch.setitem(own_excerpt.SPLITS, "train", SimpleNamespace(sha256=sha256(source)))
    own_excerpt.run_own_excerpt(tmp_path)
    result = json.loads((output / "own_excerpt_summary.json").read_text())
    assert result["official_dev_records_read"] == 0
    assert len(result["results"]) == 4
    assert {path.name: sha256(path) for path in original.iterdir()} == before
    assert len(json.loads((output / "predictions.json").read_text())) == 32
    with pytest.raises(FileExistsError):
        own_excerpt.run_own_excerpt(tmp_path)
    from apv_rag import own_excerpt_refit

    refit = tmp_path / "artifacts/own_excerpt_refit"
    refit.mkdir()
    np.save(refit / "own_train_features.npy", x_train)
    cache_receipt(refit / "own_train_features.npy", save=True)
    own_excerpt_refit.run_refit(tmp_path)
    report = json.loads((refit / "own_excerpt_refit_summary.json").read_text())
    assert report["official_dev_records_read"] == 0
    assert report["training_claims"] == 80
    assert report["validation_claims"] == 8
    assert len(report["results"]) == 2
    assert len(json.loads((refit / "predictions.json").read_text())) == 16
    assert {path.name: sha256(path) for path in original.iterdir()} == before


def test_own_excerpts_preserve_order_deduplicate_and_limit_to_five():
    record = {
        "questions": [
            {
                "answers": [
                    {"answer": "  a ", "source_url": "u"},
                    {"answer": "a", "source_url": "u"},
                    {"answer": "", "source_url": "v"},
                    *[{"answer": str(i), "source_url": "v"} for i in range(6)],
                ]
            }
        ]
    }
    assert own_premises(record) == ["a", "0", "1", "2", "3"]
    assert own_premises({"questions": []}) == []
