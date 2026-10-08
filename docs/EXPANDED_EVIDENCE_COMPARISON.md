# Expanded evidence representation

Extract the consolidated update into `D:\apv-rag-tesla369`, keeping all previous
artifacts, models and `.venv`. Run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_expanded_evidence_comparison.py
```

No NLI or embedding inference occurs. The runner reads the checksummed own-excerpt
training and validation audits already produced by the earlier steps.

It compares the original seven pooled NLI features against eighteen features:
the same seven plus three NLI standard deviations, mean normalized NLI entropy,
the product of maximum contradiction and entailment, neutral-argmax fraction,
maximum and mean claim-token overlap, unique source-host count, and mean and
maximum log-transformed excerpt word counts. Feature names and their order are
saved in the summary. No labels, justifications, gold verdicts or fact-check
article identifiers enter the feature vector.

The original calibrated logistic regression and five seeds are kept. Scaling
and model fitting use training rows only. Ordinary argmax and exponent-one
prior adjustment are reported without searching for a new exponent. The
seven-feature baseline must reproduce the previous own-excerpt refit probabilities.
Connected training/validation groups and audit order are checked before fitting.

This is exploratory development in an oracle excerpt setting. Additional
features may fail to improve classification. Host counts and overlap are proxies,
not evidence sufficiency judgments, and may reflect benchmark annotation practices.
Official dev is not read and earlier held-out results remain unchanged.
Separate new confirmation data would be needed for the revised method.

Upload from `D:\apv-rag-tesla369\artifacts\expanded_evidence_comparison`:

- `expanded_evidence_summary.json`
- `predictions.json`
- `output_manifest.json`

The real comparison remains pending your run; feature extraction has synthetic
tests and the authoring environment checks software regressions.
