# Own-excerpt training comparison

Extract the consolidated update directly into `D:\apv-rag-tesla369`.
Preserve models, `.venv`, previous artifacts and pair caches. Run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_own_excerpt_refit.py --threads 4 --batch-size 8
```

The runner computes NLI features for up to five own excerpts for each of the
2458 original training claims. It reuses the earlier pair cache where possible
and resumes missing pairs after interruption. The existing 609 own-validation
feature rows are reused without further inference. No models are downloaded.

Each of five classifiers fits only on own-excerpt training features and training
labels. Scaling and sigmoid calibration follow the original settings. Inner
calibration uses five stratified folds, not grouped folds; the outer internal
train/validation boundary is checked for connected claim/article leakage.

Both argmax and exponent-one prior adjustment are reported with no new rule
search. Results compare against the previous retrieved-trained classifiers
evaluated on the same own-excerpt validation inputs. This tests the effect of
changing classifier training inputs while holding the validation input fixed.

This is development analysis using oracle benchmark excerpts. It does not
measure open-web retrieval performance, guarantee sufficient evidence or
confirm the revised method on unseen data. Official dev is not opened.
Existing held-out results stay frozen. All new neural inference is pending the
Windows run; the cached comparison path has synthetic integration coverage.

Upload from `D:\apv-rag-tesla369\artifacts\own_excerpt_refit`:

- `own_excerpt_refit_summary.json`
- `predictions.json`
- `output_manifest.json`
