# Frozen held-out excerpt evaluation

Extract the consolidated ZIP directly into `D:\apv-rag-tesla369`.
Keep the existing `.venv`, `models`, dataset and original comparison caches.

Run in the VS Code PowerShell terminal:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\run_heldout_evaluation.py --threads 4 --batch-size 8
.\.venv\Scripts\python.exe scripts\validate_heldout_evaluation.py
.\.venv\Scripts\python.exe -m pytest -q
```

No model download is needed. The first command opens the official dev dataset
only after frozen source, code, model, training-feature and package checks pass.
The original model comparison must remain in `artifacts/retrieval_nli_comparison`.
Do not overwrite the frozen protocol to resolve a mismatch. Report the named
mismatch so its cause can be investigated.

The evaluator fits only the original training indices, keeps the selected
decision exponents, and applies them to the 500 official development claims.
The overlap audit uses normalized exact claims and fact-check article URLs,
including transitive links across the full train/dev collections. It excludes
claims connected to fitted training records from the primary analysis and
reports all 500 separately. Empty claims are excluded from the primary analysis.
This audit cannot detect every semantic duplicate or source dependency.

The corpus contains deduplicated official-dev answer excerpts, including
excerpts attached to excluded claims. This is the frozen oracle-corpus design:
it does not simulate finding evidence on the open web. Labels are used only
for scoring, never for model fitting or rule selection.

The report includes five-seed Macro-F1 differences, grouped bootstrap intervals,
paired group randomization tests and Holm adjustment within each population.
Only the independent population is primary. All-dev statistics are secondary.
Raw Brier scores and argmax ECE remain unchanged by the decision adjustment.
Inner calibration folds preserve the earlier stratified, non-grouped design.

Interrupted inference can resume using a fixed input identity, pair cache and
checksummed completed arrays. A completed evaluation refuses overwriting.
Do not change methods after viewing its results and describe another run as
confirmation. The local freeze is not an external preregistration.

After completion, send these four files from
`D:\apv-rag-tesla369\artifacts\heldout_excerpt_evaluation`:

- `heldout_summary.json`
- `overlap_audit.json`
- `predictions.json`
- `output_manifest.json`

The evaluator has synthetic tests, but neural inference has not been run in
the authoring environment. Actual held-out metrics remain pending your run.
This evaluation alone does not complete the full APV-RAG system.
