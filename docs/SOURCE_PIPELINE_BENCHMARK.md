# Source-pipeline benchmark and ablations

Run on the existing SciFact local cache:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_source_pipeline_benchmark.py
```

The script reuses saved target NLI scores and full-corpus BM25 indexing. It does not retrain or download models. Five configurations are evaluated: the full heuristic pipeline, no family collapse, no source weights, no confidence/margin thresholds, and required-primary-source with all primary flags absent. The last configuration is a controlled gate test, not a natural missing-archive benchmark.

SciFact supplies no source rank or provenance-family annotation. All source ranks are 5 and exact normalized duplicate text is the only dependency signal. Therefore source-weight ablation must have equal scores; it cannot establish the benefit of weighting sources. Different texts are not proven independent simply because exact duplicates are absent. No gold rationale sentences or cited-document identifiers enter retrieval.

Report coverage, accuracy and three-class macro-F1 on covered claims, correct candidates divided by all claims, and overlapping abstention-reason counts. Covered-only metrics must always be accompanied by coverage. Different coverage configurations cannot establish a superiority claim from covered accuracy alone. Thresholds remain engineering defaults and this observed SciFact dataset is now exploratory.

If a family collapse selects an abstract outside the existing top-five inference cache, the evaluator stops rather than inventing its NLI score. Completed outputs are never overwritten.

Provide `benchmark_summary.json`, `predictions.json`, and `output_manifest.json` from `artifacts\source_pipeline_benchmark`. Keep `input_manifest.json` locally for the input audit. This stage does not complete XFEVER, multilingual inference, open-web generation, or a learned evidence-sufficiency model. Manuscript writing remains paused.
