# Expanded feature decision-rule selection

Run from `D:\apv-rag-tesla369`:

```powershell
.\.venv\Scripts\python.exe scripts\select_expanded_decision_rule.py
```

Requires the completed `artifacts/retrieved_feature_comparison` outputs and original retrieval comparison inputs. No neural inference or model download is required. Existing completed selection outputs are never overwritten.

For each retrieval method, select the prior-adjustment exponent from 0, 0.25, 0.5, 0.75, and 1 using mean macro-F1 across five seeds of three grouped training folds. Priors come only from each fitting fold; ties favor the smaller exponent. Refit on all training records and evaluate the selected rule and unadjusted argmax on internal validation. The original sigmoid calibration uses stratified inner folds.

This is a classifier-level development experiment. Retrieval features use the fixed training corpus across outer folds, so this is not fully nested retrieval evaluation. Internal validation has already been observed repeatedly. The script does not read official development records and does not establish independent confirmation.

After completion, provide these files from `artifacts/expanded_training_rule_selection`:

- `selection_summary.json`
- `predictions.json`
- `output_manifest.json`
