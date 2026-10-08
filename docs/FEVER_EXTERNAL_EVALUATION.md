# New external FEVER evaluation input

The official FEVER labeled development file is frozen under `data/external/fever/heldout_v1`. This is a new external cohort for this project, not an official blind test. Source: https://fever.ai/dataset/fever.html . Its downloaded 19,998-row file has SHA-256 `e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7`.

`scripts/prepare_fever_heldout.py` ranks integer claim IDs by SHA-256 with the fixed seed `apv-rag-fever-heldout-v1` and keeps 300 distinct claim texts. It excludes all IDs and matching English text in the project's prior XFEVER generation export. No labels or gold evidence affect the ranking. The resulting `model_inputs.jsonl` has IDs and claim text only; `gold.jsonl` is reserved for scoring after outputs are frozen. The selected counts are 110 SUPPORTS, 86 REFUTES and 104 NOT ENOUGH INFO. The source, selection, overlap exclusion and files are bound in `selection_manifest.json` and `output_manifest.json`. Rerunning the script refuses to replace a different selected ID list.

The official June 2017 Wikipedia archive is **not** in this package. The authors' Zenodo record provides `wiki-pages.zip` (DOI `10.5281/zenodo.4925954`); FEVER's dataset metadata records 1,713,485,474 bytes and SHA-256 `4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2`. The user's downloaded copy was checked against both values on 2026-10-04. The archive remains on the Windows machine and is not redistributed. Source pages, when indexed, are benchmark text; they do not authenticate a historical publisher.

After placing the verified archive in `D:\apv-rag-tesla369\data\external\fever\heldout_v1\wiki-pages.zip`, run from `D:\apv-rag-tesla369` in the activated project environment:

```powershell
python scripts\build_fever_index.py
python scripts\retrieve_fever_contexts.py
```

The first command checks the complete archive SHA-256 and streams all 5,416,537 source records directly from the ZIP into `data\processed\fever\heldout_v1\wiki.sqlite` with SQLite FTS5. All-empty `id`, `text`, and `lines` records are excluded from the index and recorded by shard and line number in `empty_placeholders`; raw source count and indexed page count remain separate. No extraction is needed. It can take considerable time and disk space (allow at least 20 GB free). It refuses an existing index and leaves a `.partial` file if interrupted; after checking that no build is running, remove that incomplete file before restarting. The second command reads only `model_inputs.jsonl`, searches the index with a fixed title-weighted FTS5 BM25 query, selects up to three sentences from each of three retrieved pages, and writes `artifacts\fever_retrieval_v1\contexts.jsonl` plus an input receipt. It does not read gold labels or run the verdict model. Existing output is never silently overwritten.

This FTS5 retrieval policy is a new predeclared benchmark policy, not numerically identical to the project's in-memory BM25 retriever. Any later verdict and gate evaluation must use the saved contexts without selecting settings from gold labels, pin model versions, freeze prediction hashes, and score only afterwards. The existing SciFact and climate runner cannot silently be reused as the FEVER evaluator. No FEVER accuracy result is established by indexing or retrieval.

The archive and model weights belong in the existing `D:\apv-rag-tesla369` project on the Windows machine, outside the consolidated ZIP. Do not infer end-to-end accuracy from this input-preparation step. FEVER's Wikipedia claims also do not authenticate Tesla-era publishers or prove generated explanations.

## Frozen NLI baseline and separate scoring

The Windows screenshot on 2026-10-04 reports an index containing 5,416,536 pages from 5,416,537 raw records, one audited all-empty placeholder, and completed retrieval for all 300 frozen claims. The archive and index need no rebuild for the next step. The full index and neural weights are not present in this workspace; these counts are observed Windows output, not a local independent replay.

The next baseline is fixed before its target scores are opened: use the original `models/nli-deberta-v3-small` file identities and inference packages recorded in the prior XFEVER export, as copied into `config/fever_nli_reference_v1.json`. Use each nonempty saved page excerpt as one premise, with the claim as hypothesis, CPU float32, seed 369, four threads, batch size eight and pair truncation at 256 tokens. Average normalized contradiction/entailment/neutral scores across page premises, reorder to Supported/Refuted/Not Enough Evidence and take argmax with the fixed first-index tie rule. No premises means NEI probability one. There is no retriever selection, training, threshold fitting, generator or target calibrator in this baseline.

Run in the existing project root:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_fever_nli.py
.\.venv\Scripts\python.exe scripts\score_fever_nli.py
```

The runner reads model-only inputs and contexts, checks their alignment, checks local weights and package identities, and does not open gold. It records its complete input identity before any cached inference and refuses cache reuse if inputs change. Pair inference can resume after interruption with the same command. Completion writes `artifacts/fever_nli_v1/predictions.json` and a checksum manifest; rerunning a completed evaluation refuses overwrite. The separate scorer verifies the frozen outputs, reconstructs each probability aggregation and checks every claim and passage against the saved contexts before opening the pinned `gold.jsonl`. It writes `artifacts/fever_nli_scoring_v1/summary.json` and bound inputs/outputs. The scorer also refuses overwrite.

Report accuracy, macro F1, per-class performance, Brier, 15-bin top-label ECE and descriptive risk/coverage at 50%, 80% and 100%. Report retrieval recall for finding every page or sentence in at least one annotated gold evidence group among the saved contexts, excluding NEI from recall denominators. These are retrieval diagnostics on up to nine sentences; they are not the official FEVER score. Token truncation can omit some of those sentences from NLI input. The NLI neutral class is not a validated evidence-sufficiency signal. This baseline does not establish integrated APV-RAG performance, multilingual retrieval, historical authentication or explanation truth. No FEVER neural accuracy result exists here until the Windows runner and scorer complete.
