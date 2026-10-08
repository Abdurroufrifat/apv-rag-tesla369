# Multilingual retrieved-context controller comparison

Status: runner prepared; actual model results pending execution on the Windows machine.

Run from `D:\apv-rag-tesla369` using the original virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\run_multilingual_retrieved_pipeline.py
```

The script checks frozen input hashes, the original six package versions and four local model folders. It downloads nothing. CPU float32, four threads and seed 369 are fixed. Feature records and exact-prompt Qwen responses are saved incrementally; rerunning the same command resumes the same identity. Changed input/code/model identities are rejected rather than silently rebinding caches. A completed run is replayed and exported without neural inference.

Upload `D:\apv-rag-tesla369\multilingual_retrieved_pipeline_outputs.zip`. It contains results and receipts, never weights. The full experiment runs 660 queries, 2 direct NLI baselines per query, and 4 generator/gate policies per query (2,640 policy rows). Identical prompts share deterministic responses within this run; no earlier generator responses are imported.

## Frozen sample and contexts

Reuse the original sixty label-blind explanation claim IDs from the previously observed XFEVER experiment, aligned across English and machine/human variants in Spanish, French, Indonesian, Japanese and Chinese. There are eleven files and sixty rows per file. No new outcome-based sampling is performed. This is exploratory reuse of observed data, not fresh independent confirmation.

Contexts are the previously frozen, claim-only top-three positive BM25 results using the Unicode-word/CJK-character/bigram analyzer. The per-file corpus consists of deduplicated benchmark paired excerpts and contains each target by construction. No query-specific target evidence is supplied to the model. This optimistic closed-pool setup is not full-page or open-web retrieval.

Claims are clipped to 64 Qwen tokens and each passage to 96. The original context collapse applies. Source excerpt IDs, scores, context hashes and pre/post clipping counts are recorded. English NLI and multilingual NLI both use the same shown contexts, pair max length 256. MiniLM has max length 384. Qwen max input is 1,024, output 128, deterministic greedy constrained verdict followed by an explanation. Exceeding the input budget raises an error rather than silently trimming the full prompt.

## Comparisons and confidence

The original English Deberta and MiniLM produce the unchanged thirteen gate features. Frozen NLI, embedding and combined heads use threshold 0.5. They are compared with no-gate generation. mDeBERTa multilingual NLI is evaluated separately using averaged passage CEN probabilities; it is never substituted into English-trained gate features. Empty retrieval gives the direct baselines NEI and makes generation abstain.

All predictions, direct probabilities, bound feature caches and response receipts are frozen before the scoring routine opens the gold file. There is no target fitting. Per-file metrics include three-label accuracy/F1, direct-baseline Brier/ECE, generation coverage, all-query accuracy counting abstentions as errors, and accepted-answer accuracy. A gate probability estimates the original completeness target, not answer correctness. Generated-answer `correctness_confidence` remains null. The raw English-NLI score for the generated label is explicitly uncalibrated. English FEVER temperature and correctness fits are not reused.

## Replay and limitations

`--preflight` checks inputs without weights or inference. `--verify` checks complete file coverage/checksums, sample/model/code identity, context collapse, feature binding/aggregates, multilingual probability shapes, SQLite prompt/response equality, all controller/direct outputs and scored metrics. It does not reproduce neural forward passes, independently count tokenizer budgets or validate explanation truth. Floating-point replay tolerance is 1e-12.

Translated variants share claims and are not independent samples. Gates and embeddings remain English trained. Prompts/explanation handling and numeric guards remain the original English-oriented controller; localized numeric equivalence and explanation translation quality are not established. Excerpt hashes do not authenticate historical publishers. No human annotations are requested or created; upstream benchmark labels are used only for scoring. These outputs will evaluate a scoped machine-only comparison and cannot establish the entire original charter or SCI acceptance.


## Windows checkpoint-save recovery

The runner retries PermissionError during JSON checkpoint saves for at most ten attempts (about eleven seconds of waits). It never deletes a valid target checkpoint to bypass a lock. If access remains denied, it stops with the path and resume instructions. The earlier shared writer and experiment scripts remain unchanged on disk.

An interrupted original runner can resume through the file-save fix only when its entire input identity matches the pinned original runner/protocol, all other code/model/sample/settings identities are unchanged, and it stopped in the English-feature stage. The saved prepared contexts and every existing feature record are validated before advancing the receipt. A separate `io_resume_receipt.json` preserves the original identity and checkpoint hashes. Changed experiment settings or later-stage caches are rejected. A leftover `.partial` file is not promoted to a valid checkpoint; the last valid JSON remains the resume source.
