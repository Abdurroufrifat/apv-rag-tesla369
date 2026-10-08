# Evidence baseline design

## Goal

Create the first reproducible machine-only AVeriTeC evidence baseline. It must
test whether published question-answer evidence helps verify a claim, without
using the untouched official development split during training or model choice.

## Scope

The baseline has two model variants:

1. `claim_only`: TF-IDF features from the claim text.
2. `claim_plus_evidence`: TF-IDF features from the claim plus the published
   question, answer, and answer type fields.

Both use multinomial logistic regression. The claim-only variant is a diagnostic
lower bound. The claim-plus-evidence variant is the reported local evidence
baseline. Neither is called APV-RAG or presented as the final model.

## Inputs

- immutable `data/external/averitec/official_7c62d1e/train.json`;
- immutable `data/external/averitec/official_7c62d1e/dev.json`;
- frozen Phase 2B index and group files;
- a configuration file containing seed, TF-IDF settings, regularization values,
  and abstention coverages.

The known empty training claim remains excluded by the Phase 2B index file.

## Data flow

1. Validate the official AVeriTeC files and the Phase 2B split manifest.
2. Construct one deterministic text record per upstream index.
3. Fit each model only on the 2,458 Phase 2B training indices.
4. Fit calibration only on training data through stratified cross-validation;
   do not fit it on the 609 validation records.
5. Evaluate both variants on the 609 internal validation records.
6. Save predictions, probabilities, and metrics in an ignored results folder.
7. Keep the official 500-record development set unused until the baseline and
   its settings are frozen.

## Text construction

For `claim_only`, use the claim text.

For `claim_plus_evidence`, append each published question and its answers in the
following fixed order: question, answer type, answer text, and source medium.
URLs, verdict labels, justifications, speaker names, reporting source, and
fact-check article URLs are excluded from model features because they could leak
source-specific patterns or labels.

## Model selection

Compare three fixed regularization values: `C = 0.25, 1.0, 4.0`. Select the
claim-plus-evidence setting with the highest internal validation macro-F1. If
scores tie, select the lower C value. The selected setting is recorded in a
manifest. No official-development result is examined before this decision.

## Metrics

Report macro-F1, per-class precision/recall/F1, balanced accuracy, multiclass
Brier score, 15-bin expected calibration error, and risk-coverage values at 50%,
80%, and 100% coverage. The abstention score is the maximum predicted class
probability; lower-confidence predictions are abstained first.

## Outputs

All generated output is ignored by Git under:

```text
artifacts/phase2c_evidence_baseline/
```

Each run writes a run manifest, model settings, validation predictions, and a
metrics JSON file. The manifest records input hashes, seed, package versions,
selected setting, and output hashes.

## Error handling

The runner stops if the AVeriTeC or Phase 2B validator fails, if prediction
indices do not match validation indices, if a required evidence field has an
unexpected type, or if an output directory already contains a completed run.
It never replaces upstream data or split files.

## Tests

Unit tests will cover text construction, exclusion of forbidden fields,
probability metric calculations, deterministic model-setting selection, and
prediction-index alignment. An end-to-end smoke test will use synthetic records
only. The real dataset run will be validated by its output manifest.
