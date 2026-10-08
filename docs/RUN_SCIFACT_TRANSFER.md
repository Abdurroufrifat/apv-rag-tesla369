# Run the fixed SciFact transfer evaluation

Extract the updated project into the existing `D:\apv-rag-tesla369` folder, preserving models and artifacts. Run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts\run_scifact_transfer.py
```

The evaluator requires the original NumPy, scikit-learn, PyTorch, transformers, and NLI model versions. It reuses the expanded BM25 training features. Only SciFact inference is new. It does not download models or train on SciFact labels.

Inference runs on CPU and prints progress every 25 claims. Interrupted runs reuse the SQLite pair cache only when the input identity matches. Completed results are not overwritten. Changing code, packages, dataset bytes, models, or thread/batch settings requires a separately documented run, not editing cached results.

Provide these files from `artifacts\scifact_transfer`:

- `transfer_summary.json`
- `predictions.json`
- `output_manifest.json`
- `input_manifest.json`

The summary reports three-label target macro-F1, accuracy, per-class metrics, and out-of-target predictions. Four-class probabilities are retained. Source-host features use the single dataset repository host for all abstracts; this is not article provenance. Model premises are truncated by the existing NLI tokenizer at 256 pair tokens. Paired grouped uncertainty analysis follows after prediction verification. No SciFact result has been produced in the authoring environment.
