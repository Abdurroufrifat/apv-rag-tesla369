# Feature-group ablation

Extract the consolidated update directly into `D:\apv-rag-tesla369`, keeping
previous artifacts, caches, models and `.venv`. Run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_feature_ablation.py
```

No neural inference occurs. Eight fixed representations use the saved eighteen
features: base seven, all eighteen, each added group alone with the base, and
all features with each group removed. The added groups are NLI variation and
uncertainty (six), lexical overlap (two), and source/length proxies (three).

All five seeds, calibrated logistic-regression settings, training priors and
the two decision rules are unchanged. Baseline and full probabilities must
reproduce the previous comparison. Scaling/fitting use training rows only.
Official dev is not opened. No validation winner is automatically selected.

This describes conditional group contributions in the oracle excerpt setting.
Effects are not additive or causal, and source/length proxies may reflect
benchmark annotation practices. Repeated internal-validation analysis is
exploratory and cannot replace a new confirmation dataset.

Upload from `D:\apv-rag-tesla369\artifacts\feature_group_ablation`:

- `ablation_summary.json`
- `predictions.json`
- `output_manifest.json`

The real ablation remains pending your run.
