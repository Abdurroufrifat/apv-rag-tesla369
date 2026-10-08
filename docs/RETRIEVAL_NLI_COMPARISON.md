# Local retrieval/NLI comparison

This runner compares BM25+NLI and normalized-embedding retrieval+NLI using
local models. It is a benchmark evidence-excerpt experiment, not an open-web
RAG system or official AVeriTeC evaluation. Model inference has not been run
in the authoring environment; no performance results are supplied.

Training and validation each have their own corpus, built from that split's
benchmark-provided answers, deduplicated by text and source URL. Both methods
see the same corpus within each split. Validation evidence is not used in the
training corpus. Labels and justifications are excluded from retrieval and
NLI input. Only the claim is used as a query; NLI receives retrieved answer
text as premise and the claim as hypothesis.

This is an oracle excerpt corpus: fact-checker-selected evidence is available
in advance. It does not reproduce retrieval from original source pages. Its
results cannot be compared directly with the earlier classifiers supplied
with each claim's gold evidence or with official leaderboard scores.

Fixed settings: top 5 positive-score documents, BM25 k1=1.2 and b=0.75,
normalized embedding dot product, NLI input length 256 tokens, CPU inference.
NLI classes are checked against contradiction, entailment, neutral. Features
are each class's maximum and mean probability plus retrieved document count.
No-document queries yield zeros. A standardized class-balanced logistic
regression learns the four-way verdict mapping from training labels with C=1.
Five-fold sigmoid calibration and feature scaling are fitted inside training
folds. Seeds are 369, 1369, 2369, 3369, 4369. They vary calibration folds,
not deterministic retrieval or pretrained NLI predictions.

Local model files, dependency versions, source data, and split IDs are hashed
before inference. Changes to those inputs stop cache reuse. The SQLite pair
cache commits every batch; interrupted inference resumes without recalculating
committed pairs. Document embeddings and completed feature matrices are reused.
Avoid editing cached files. The output validator checks final file checksums
and recomputes classification metrics, but does not rerun the neural models.

## Windows commands

Extract the updated consolidated package into `D:\apv-rag-tesla369`. Preserve
the existing `models` folder and virtual environment.

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\validate_averitec_splits.py
.\.venv\Scripts\python.exe scripts\run_retrieval_nli_comparison.py --batch-size 8 --threads 4
.\.venv\Scripts\python.exe scripts\validate_retrieval_nli_comparison.py
```

Run the comparison only if split validation passes. If batch memory is too
large, use `--batch-size 2`; completed pairs remain reusable. CPU inference
may take substantial time. Progress is printed every 50 records. Do not start
two copies of the runner against the same output directory.

The experiment reads only upstream `train.json`. Official development data
and Tesla observations are excluded. After completion, send
`artifacts/retrieval_nli_comparison/comparison_summary.json`,
`predictions.json`, `input_manifest.json`, and `output_manifest.json`.
Do not send model weights or the large SQLite cache.

The comparison still requires measured results and paired statistical testing
before any superiority claim. It does not complete the full APV-RAG evaluation.
