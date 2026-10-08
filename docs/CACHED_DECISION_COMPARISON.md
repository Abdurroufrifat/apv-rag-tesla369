# Cached NLI decision comparison

The cached arrays supplied by the Windows run were checked against its output
manifest. Both methods have 2,458 training and 609 validation rows with seven
finite features. No model inference or downloads were repeated.

Decision score for class c is p(c) / prior(c)^alpha. This score is used only
for argmax verdict selection, not as a probability or confidence estimate.
The original calibrated probabilities are preserved exactly. Brier score and
probability calibration therefore remain unchanged; selective-risk results
must be recomputed for the changed verdict rule before reuse.

Alpha candidates were 0, 0.25, 0.5, 0.75, and 1. For each retrieval method,
selection maximized mean training out-of-fold Macro-F1 across five seeds;
ties favored smaller alpha. Outer folds used three-fold StratifiedGroupKFold
and frozen provenance/claim group assignments. Each outer model used five-fold
training-only sigmoid calibration. Priors for outer predictions were computed
from the outer fitting partition, not its held-out labels. The final verdict
rule uses full training priors and the original five saved validation runs.

The cached retrieval corpus was not rebuilt per outer fold. This is a grouped
cross-validation analysis of the aggregation stage, not a fully nested
end-to-end retrieval evaluation. Inner calibration folds remain stratified by
class rather than grouped. Validation results were inspected in the earlier
diagnostic, so these measurements are development results, not a fresh final
evaluation.

| Method | Selected alpha | Mean training OOF Macro-F1 | Original validation Macro-F1 | Selected validation Macro-F1 |
|---|---:|---:|---:|---:|
| BM25+NLI | 1 | 0.3250 | 0.2447 | 0.3220 |
| Dense+NLI | 1 | 0.3393 | 0.2532 | 0.3358 |

The validation values average five seeds. At seed 369, BM25 predicts 196
Supported, 274 Refuted, 11 Not Enough Evidence, and 128 Conflicting Evidence
records. Dense retrieval predicts 106, 301, 179, and 23 respectively.
Recovering predictions for all four classes does not by itself establish good
minority-class accuracy. No superiority or final-test claim follows from this
development comparison.

The official development set remains unused. Tesla records remain excluded.

## Windows verification

Extract the consolidated ZIP directly into `D:\apv-rag-tesla369`, preserving
the existing model folders, virtual environment, and original inference cache.
The completed decision outputs are included; do not rerun model inference.

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\validate_cached_decision_comparison.py
.\.venv\Scripts\python.exe -m pytest -q
```

The runner `run_cached_decision_comparison.py` is included for reproducibility.
It refuses to overwrite completed decision artifacts and needs the original
four feature arrays and inference manifests to perform a new run.
