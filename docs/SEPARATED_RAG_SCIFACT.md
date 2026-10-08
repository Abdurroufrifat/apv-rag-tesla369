# Separated multi-passage RAG experiment

Use the pinned Qwen instruction model and existing English NLI model. This exploratory runner extends the adaptively tested interface to the same 300 SciFact claims as the earlier transfer experiment. Prior SciFact outcomes have been observed. No independent confirmation or novelty claim is made.

Retrieve top three positive BM25 abstracts from the full sorted corpus, without gold citations or rationales. Preserve the first 96 Qwen tokens per abstract and 64 tokens per claim; save original and displayed claims. Use separate short verdict and explanation calls with direct chat-template tensors. CPU float32, four threads, seed 369, greedy decoding, 128 new tokens and maximum 1024 input tokens. Reject oversized prompts rather than silently truncate. Record raw generations, prompts, context IDs, retrieval scores, settings and input/code/model hashes. Cache each call separately; identity changes refuse reuse. Completed runs refuse overwrite.

Structural guards reject unknown labels, empty explanations, absent evidence and numeric strings absent from both displayed claim and evidence. Claim-only numeric references are explicitly marked and are not treated as evidence-supported quantities. Numeric strings present in inputs can still be misrepresented. Context IDs record provenance of supplied passages; they are not model-selected or verified citations.

Score both raw valid verdicts and structurally accepted answers, counting abstentions as errors across all 300 claims. Report coverage, covered accuracy and three-class macro-F1/accuracy. No probability calibration is claimed. Inspect explanation sentences using NLI against every displayed passage, preserving all C/E/N scores. These scores are auxiliary diagnostics, not gold entailment or automatic acceptance criteria: the earlier NLI smoke failure exposed relevance errors. A rationale may require multiple passages and claim context; per-passage scores cannot settle that. Summaries may repeat false claims or make incorrect logical links even when numbers match.

No target fitting, prompt selection from target scores, source authentication, archival gold labeling or human annotation is performed. This does not close learned sufficiency, source-quality efficacy, multilingual generation or the complete APV-RAG project. Manuscript writing stays paused.

Run in the activated environment at D:\apv-rag-tesla369:

```powershell
python scripts\run_separated_rag_scifact.py
Compress-Archive -Path .\artifacts\separated_rag_scifact_v1\*.json -DestinationPath .\separated_rag_scifact_outputs.zip
```

Reuse downloaded models. CPU inference may be slow. Rerun an interrupted run to resume its cached generations. Send separated_rag_scifact_outputs.zip for verification and analysis. Run the archive command only after successful completion.
