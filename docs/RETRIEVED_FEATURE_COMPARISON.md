# Expanded features on retrieved excerpts

Extract the update directly into `D:\apv-rag-tesla369`, preserving all previous
artifacts, caches, models and `.venv`. Run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_retrieved_feature_comparison.py
```

No neural inference occurs. Saved BM25/dense retrieval audits, split-local
corpora and NLI scores are reused. The runner compares the same seven and
eighteen features on both retrieval methods, with five seeds and the original
calibrated logistic regression. Both argmax and exponent-one prior adjustment
are reported; no new exponent is searched. The ablation's best observed variant
is not automatically substituted for the full eighteen-feature representation.

Document positions, excerpt/NLI alignment, corpus identity, file checksums,
feature reproduction and connected split boundaries are checked. The baseline
must reproduce the original validation probabilities. Scaling, calibration and
classifier fitting use training rows only. Official dev is never opened.

This tests a representation change with retrieval outputs fixed. The candidate
corpora still contain benchmark-annotated answer excerpts, so it is neither
open-web RAG nor an official benchmark score. New confirmation data remains
necessary for any revised method. The real comparison is pending your run.

Upload from `D:\apv-rag-tesla369\artifacts\retrieved_feature_comparison`:

- `retrieved_feature_summary.json`
- `predictions.json`
- `output_manifest.json`
