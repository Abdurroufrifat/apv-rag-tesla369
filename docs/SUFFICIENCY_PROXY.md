# Learned benchmark-label proxy

This exploratory experiment predicts AVeriTeC NEI versus non-NEI from existing retrieved-passage features. Non-NEI includes Supported, Refuted and Conflicting Evidence/Cherrypicking. These labels concern the benchmark's evidence collection, not whether our retrieved passages are sufficient. Do not call predictions authenticated evidence or use them as historical gold verdicts.

Use the frozen 2,458 training and 609 internal validation claims and existing checksummed BM25/dense seven/eighteen-feature matrices. Compare balanced logistic regression with standardized inputs, C=1 and maximum 2,000 iterations across the established five seeds. No neural inference or new model download is needed. Three group-separated training folds supply OOF probabilities. Select a threshold from 0.1 through 0.9 by binary macro-F1; ties choose the value closest to 0.5, then the smaller value. Fit the final scaler and classifier on training only. Evaluate the selected threshold and fixed 0.5 on validation, with a training-prior baseline.

Report binary macro-F1, balanced accuracy, AUROC, positive average precision, Brier score and positive prediction fraction. Probabilities from balanced logistic regression are not assumed calibrated. Save every validation probability, selected threshold, scaler and coefficient, source and feature hashes, package versions and code hashes. No official development set is read. Validation has already informed earlier development, so these results are exploratory.

Existing full-training retrieval features are reused within OOF folds. Retrieval construction is not nested inside folds; this limits what the training OOF estimate establishes. Actual retrieved-evidence sufficiency, selective downstream accuracy and external generalization remain untested by this experiment. The source pipeline is not automatically changed by this proxy.

Run in the activated Windows environment from D:\apv-rag-tesla369:

```powershell
python scripts\run_sufficiency_proxy.py
python -m pytest -q
Compress-Archive -Path .\artifacts\sufficiency_proxy\*.json -DestinationPath .\sufficiency_proxy_outputs.zip
```

Send sufficiency_proxy_outputs.zip for verification. Existing artifacts/retrieved_feature_comparison feature files are required. Completed runs refuse overwrite. Manuscript writing remains paused.
