# Evidence Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible TF-IDF and calibrated logistic-regression evidence baseline for AVeriTeC using only the frozen Phase 2B train/validation split.

**Architecture:** A focused dataset module renders claim-only and claim-plus-evidence text without prohibited fields. A metrics module computes classification, calibration, and abstention metrics from aligned probability arrays. A runner validates inputs, fits models with training-only cross-validation calibration, selects an evidence model using internal validation macro-F1, and writes immutable run artifacts. A validator checks every run file against its manifest.

**Tech Stack:** Python 3.10+, scikit-learn, NumPy, PyYAML, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-14-evidence-baseline-design.md`

## Global Constraints

- Use only `data/external/averitec/official_7c62d1e/train.json` for fitting and the frozen Phase 2B indices for model development.
- Keep `data/external/averitec/official_7c62d1e/dev.json` unread by the baseline runner.
- Exclude label, justification, URLs, speaker, reporting source, and fact-check article fields from model text.
- Use the fixed seed `369`; compare `C = 0.25, 1.0, 4.0`; use five-fold training-only calibration.
- Save outputs only under `artifacts/phase2c_evidence_baseline/`; refuse to overwrite a completed run.
- Do not commit or push to GitHub until the user says the complete project is ready.

---

## File structure

| File | Responsibility |
|---|---|
| `requirements-local.txt`, `pyproject.toml` | Add the local ML dependencies with compatible version bounds. |
| `config/evidence_baseline.yaml` | Freeze seed, vectorizer, model, calibration, coverage, and output settings. |
| `src/apv_rag/evidence_baseline.py` | Render permitted evidence text, load frozen indices, fit models, select settings, and write results. |
| `src/apv_rag/metrics.py` | Compute classification, multiclass Brier, ECE, and risk-coverage metrics. |
| `scripts/run_evidence_baseline.py` | Run the full Phase 2C baseline and create one output directory. |
| `scripts/validate_evidence_baseline.py` | Verify input/output hashes, index alignment, metrics, and no official-dev access. |
| `docs/PHASE_2C_EVIDENCE_BASELINE.md` | Windows commands, expected files, and interpretation rules. |
| `tests/test_evidence_baseline.py` | Unit tests for text rendering, forbidden fields, selection, and output alignment. |
| `tests/test_metrics.py` | Unit tests for metric functions and abstention ordering. |

## Task 1: Freeze dependencies, configuration, and evidence text rendering

**Files:**
- Create: `config/evidence_baseline.yaml`
- Create: `src/apv_rag/evidence_baseline.py`
- Modify: `requirements-local.txt`
- Modify: `pyproject.toml`
- Test: `tests/test_evidence_baseline.py`

**Interfaces:**
- Consumes: `list[dict[str, Any]]` AVeriTeC records and upstream indices.
- Produces: `render_claim_only(record: Mapping[str, Any]) -> str`, `render_claim_plus_evidence(record: Mapping[str, Any]) -> str`, and `load_split_records(records: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write failing rendering tests**

```python
def test_claim_plus_evidence_keeps_questions_and_answers_only():
    row = {
        "claim": "Claim text",
        "label": "Supported",
        "justification": "Do not include me",
        "speaker": "Do not include me",
        "questions": [{"question": "Q?", "answers": [{"answer_type": "extractive", "answer": "A", "source_medium": "web text", "source_url": "https://x"}]}],
    }
    text = render_claim_plus_evidence(row)
    assert "Claim text" in text and "Q?" in text and "A" in text
    assert "Do not include me" not in text and "https://x" not in text
```

- [ ] **Step 2: Run the rendering test and verify it fails**

Run: `python -m pytest tests/test_evidence_baseline.py::test_claim_plus_evidence_keeps_questions_and_answers_only -v`

Expected: FAIL because `apv_rag.evidence_baseline` does not exist.

- [ ] **Step 3: Add configuration and dependencies**

Create `config/evidence_baseline.yaml` with:

```yaml
protocol_version: "0.1"
random_seed: 369
variants: [claim_only, claim_plus_evidence]
regularization_c: [0.25, 1.0, 4.0]
tfidf:
  ngram_range: [1, 2]
  min_df: 2
  max_features: 50000
calibration_cv: 5
ece_bins: 15
target_coverages: [0.50, 0.80, 1.00]
output_directory: artifacts/phase2c_evidence_baseline
```

Add `numpy>=1.26,<3` and `scikit-learn>=1.5,<2` to both local dependencies and project dependencies.

- [ ] **Step 4: Implement the minimal renderer and index loader**

```python
def render_claim_only(record: Mapping[str, Any]) -> str:
    return _require_text(record, "claim")

def render_claim_plus_evidence(record: Mapping[str, Any]) -> str:
    blocks = [f"CLAIM: {_require_text(record, 'claim')}"]
    for question in _require_list(record, "questions"):
        blocks.append(f"QUESTION: {_require_text(question, 'question')}")
        for answer in _require_list(question, "answers"):
            blocks.append(f"ANSWER_TYPE: {_optional_text(answer, 'answer_type')}")
            blocks.append(f"ANSWER: {_optional_text(answer, 'answer')}")
            blocks.append(f"SOURCE_MEDIUM: {_optional_text(answer, 'source_medium')}")
    return "\n".join(blocks)
```

Validate that every index is an integer in range, unique, and references a non-empty claim.

- [ ] **Step 5: Run targeted and full tests**

Run: `python -m pytest tests/test_evidence_baseline.py -v` and `python -m pytest -q`

Expected: all tests pass.

## Task 2: Implement machine-only evaluation metrics

**Files:**
- Create: `src/apv_rag/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `y_true: Sequence[str]`, `y_pred: Sequence[str]`, `probabilities: np.ndarray`, `labels: Sequence[str]`, and `coverages: Sequence[float]`.
- Produces: `classification_metrics(...) -> dict[str, Any]`, `multiclass_brier(...) -> float`, `expected_calibration_error(...) -> float`, and `risk_coverage(...) -> dict[str, float]`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_multiclass_brier_is_zero_for_perfect_predictions():
    labels = ["A", "B"]
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert multiclass_brier(["A", "B"], probabilities, labels) == 0.0

def test_risk_coverage_keeps_highest_confidence_first():
    output = risk_coverage(["A", "B"], ["A", "A"], np.array([0.9, 0.4]), [0.5, 1.0])
    assert output["accuracy_at_50_coverage"] == 1.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_metrics.py -v`

Expected: FAIL because `apv_rag.metrics` does not exist.

- [ ] **Step 3: Implement metrics without fitting any model**

Use `sklearn.metrics.classification_report`, `balanced_accuracy_score`, and `f1_score` for classification. Compute Brier as the mean sum of squared error across one-hot labels. Compute ECE from maximum probability bins. Sort confidence descending before computing risk-coverage values.

- [ ] **Step 4: Run metric tests and full test suite**

Run: `python -m pytest tests/test_metrics.py -v` and `python -m pytest -q`

Expected: all tests pass.

## Task 3: Fit, select, and save the baseline run

**Files:**
- Modify: `src/apv_rag/evidence_baseline.py`
- Create: `scripts/run_evidence_baseline.py`
- Test: `tests/test_evidence_baseline.py`

**Interfaces:**
- Consumes: validated upstream train data, Phase 2B train/validation indices, and `config/evidence_baseline.yaml`.
- Produces: `run_baseline(project_root: Path, run_id: str | None = None) -> Path` and a run directory containing `run_manifest.json`, `validation_predictions.jsonl`, `metrics.json`, and `model_selection.json`.

- [ ] **Step 1: Write failing selection and alignment tests**

```python
def test_select_setting_uses_macro_f1_then_lower_c():
    rows = [{"c": 4.0, "macro_f1": 0.61}, {"c": 0.25, "macro_f1": 0.61}]
    assert select_setting(rows)["c"] == 0.25

def test_prediction_indices_must_match_validation_indices():
    with pytest.raises(ValueError, match="prediction indices"):
        assert_prediction_alignment([2, 4], [{"upstream_index": 2}])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_evidence_baseline.py -v`

Expected: FAIL because `select_setting` and `assert_prediction_alignment` are absent.

- [ ] **Step 3: Implement model fitting**

For each variant and C value, construct:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
    ("classifier", CalibratedClassifierCV(
        estimator=LogisticRegression(C=c_value, max_iter=2000, random_state=369),
        method="sigmoid",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=369),
    )),
])
```

Fit only on Phase 2B training records. Evaluate every candidate on Phase 2B validation records. Select only from `claim_plus_evidence` candidates by highest macro-F1, breaking ties by lower C. Save both variants’ metrics so the claim-only diagnostic remains visible.

- [ ] **Step 4: Write atomic artifacts and manifest**

Each prediction JSONL line must include `upstream_index`, `true_label`, `predicted_label`, `probabilities`, and `confidence`. Use `write_json_atomic` for all JSON files. The manifest must include hashes for `train.json`, split manifest, config, every output file, seed, selected C, labels, and package versions. Refuse to reuse a nonempty output directory.

- [ ] **Step 5: Run the synthetic smoke test**

Use a synthetic four-class fixture with 20 rows. Run the baseline function against its temporary project root and assert all expected output files exist and prediction indices align.

Run: `python -m pytest tests/test_evidence_baseline.py -v`

Expected: PASS.

## Task 4: Validate and document Phase 2C runs

**Files:**
- Create: `scripts/validate_evidence_baseline.py`
- Create: `docs/PHASE_2C_EVIDENCE_BASELINE.md`
- Modify: `README.md`
- Test: `tests/test_evidence_baseline.py`

**Interfaces:**
- Consumes: a completed `artifacts/phase2c_evidence_baseline/<run_id>` directory.
- Produces: exit code 0 only when the manifest hashes, prediction alignment, and metric inputs are valid.

- [ ] **Step 1: Write the failing validator test**

```python
def test_validator_rejects_prediction_index_mismatch(tmp_path: Path):
    run_dir = make_complete_synthetic_run(tmp_path)
    (run_dir / "validation_predictions.jsonl").write_text('{"upstream_index": 999}\n')
    assert validate_run(run_dir) == ["prediction indices do not match validation indices"]
```

- [ ] **Step 2: Run the validator test and verify failure**

Run: `python -m pytest tests/test_evidence_baseline.py::test_validator_rejects_prediction_index_mismatch -v`

Expected: FAIL because `validate_run` does not exist.

- [ ] **Step 3: Implement the run validator**

Load the manifest, verify output SHA-256 values, compare prediction upstream indices with frozen validation indices, check each probability vector has the four fixed labels and sums to one within `1e-6`, and recompute headline metrics from predictions. Reject any manifest that lists `dev.json` as a model input.

- [ ] **Step 4: Document exact Windows execution**

Document these commands:

```powershell
.\.venv\Scripts\python.exe scripts\run_evidence_baseline.py
.\.venv\Scripts\python.exe scripts\validate_evidence_baseline.py
.\.venv\Scripts\python.exe -m pytest -q
```

State that the official development split remains unused and that model results are not historical authentication of Tesla claims.

- [ ] **Step 5: Run final verification**

Run: `python -m ruff check src scripts tests`, `python -m pytest -q`, `python scripts/validate_phase2.py`, `python scripts/validate_averitec.py`, `python scripts/validate_averitec_splits.py`, and the new baseline validator after a real run.

Expected: all checks pass. Do not run Git commands.

## Plan self-review

- Spec coverage: Tasks 1–4 cover text inputs, field exclusion, training-only calibration, fixed C selection, all requested metrics, artifact safety, validation, tests, and Windows documentation.
- Placeholder scan: no unfinished tasks or unspecified interfaces remain.
- Type consistency: the renderer, metric, runner, and validator function names used by later tasks are defined in the interfaces above.
