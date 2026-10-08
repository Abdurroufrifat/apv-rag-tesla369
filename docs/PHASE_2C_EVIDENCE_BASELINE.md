# Phase 2C: machine-only evidence baseline

## What this phase measures

Phase 2C compares two calibrated TF-IDF logistic-regression models on the frozen
AVeriTeC internal split:

- `claim_only` is a diagnostic for label cues in claim wording.
- `claim_plus_evidence` is the reported evidence baseline. Its input contains the
  claim, evidence questions, answer text, answer type, and source medium.

The renderer excludes labels, justifications, URLs, speaker names, reporting
sources, and fact-check article metadata. Both models fit only the 2,458 Phase 2B
training records and are compared on the 609 Phase 2B validation records.

The official 500-record AVeriTeC development split remains unused. Tesla records
remain an unlabeled archival stress test and do not affect model fitting or model
selection.

## Install the update on Windows

The update ZIP must be extracted directly into this existing folder:

```text
D:\apv-rag-tesla369
```

When Windows asks whether to replace files, choose **Replace the files in the
destination**. Do not extract it into `data`, `scripts`, or another nested
`apv-rag-tesla369` folder.

Open Visual Studio Code and choose **File → Open Folder**, then select:

```text
D:\apv-rag-tesla369
```

Open **Terminal → New Terminal**. Confirm the prompt starts with:

```text
PS D:\apv-rag-tesla369>
```

## Run Phase 2C

Run these commands one at a time:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& D:\apv-rag-tesla369\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
python scripts\validate_averitec.py
python scripts\validate_averitec_splits.py
python scripts\run_evidence_baseline.py
python scripts\validate_evidence_baseline.py
python -m pytest -q
```

The model run is CPU-friendly. A GPU or Colab is not required for this baseline.
Do not open or edit files inside `data\external\averitec` or
`data\processed\averitec` before running it.

## Expected output

The runner creates a timestamped folder under:

```text
D:\apv-rag-tesla369\artifacts\phase2c_evidence_baseline\run_YYYYMMDDTHHMMSSZ
```

It contains:

| File | Contents |
|---|---|
| `metrics.json` | Selected evidence-model metrics |
| `model_selection.json` | Results for both variants and all three C values |
| `validation_predictions.jsonl` | One aligned prediction per validation record |
| `run_manifest.json` | Input/output hashes, seed, package versions, and selected setting |

The validator checks file hashes, frozen validation-index alignment, probability
vectors, confidence values, recomputed metrics, and the rule that `dev.json` was
not a model input.

## Fixed analysis rule

Select the `claim_plus_evidence` setting with the highest internal-validation
Macro-F1. If settings tie, select the lower C value. Report Macro-F1, per-class
precision/recall/F1, balanced accuracy, multiclass Brier score, 15-bin expected
calibration error, and risk at 50%, 80%, and 100% coverage.

Do not interpret these results as proof that a Tesla quotation is authentic. The
experiment measures fact-verification behavior on published AVeriTeC labels.
