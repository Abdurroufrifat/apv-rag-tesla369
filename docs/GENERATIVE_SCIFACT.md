# Generative SciFact baseline

This exploratory component adds genuine model-generated verdicts and explanations to BM25 retrieval. It is a standalone baseline, not yet the integrated APV-RAG source-quality and sufficiency system. SciFact results have already been observed; this evaluation is not independent confirmation.

Pinned generator: google/flan-t5-base, revision 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2. Download only the listed tokenizer, config and safetensors files and record their SHA256 hashes. Training overlap with evaluation claims cannot be ruled out.

Use the same 300-claim SciFact cohort as the earlier transfer run. Retrieve the top three positive BM25 hits from all abstracts, without gold rationales or cited_doc_ids. Limit each abstract to its first 96 tokenizer tokens and the claim to 64 tokens. Preserve these actual model-visible texts and the entire prompt. Reject prompts exceeding 512 tokens rather than silently truncating. Generate greedily on CPU in float32, four threads, seed 369, maximum 96 new tokens, with no tuning on target labels.

Output format is verdict | one-sentence explanation, with numeric document IDs in brackets for Supported or Refuted. Unknown IDs, absent required citations or invalid format produce abstention. These are structural checks only: a valid citation can still fail to support a generated sentence. Not Enough Evidence is a benchmark prediction, distinct from parser abstention. Neither output is an authenticated historical verdict. Evidence is marked as data in the prompt; adversarial instruction resistance has not been established.

Report three-class macro-F1 and accuracy across all 300 claims, counting abstentions as errors, plus coverage and covered accuracy. Save raw generations, exact prompts, retrieved text, model identity, package versions, source and code hashes. Cache each completed generation for resume; reject reuse when identity changes. No probability calibration or Brier score is claimed for generation.

Run in the activated environment in D:\apv-rag-tesla369:

```powershell
python scripts\download_generative_model.py
python scripts\run_generative_scifact.py
python -m pytest -q
Compress-Archive -Path .\artifacts\generative_scifact\*.json -DestinationPath .\generative_scifact_outputs.zip
```

Generation may take substantially longer than the cached classical experiments on CPU. Rerunning resumes the same incomplete run without regenerating cached answers. Send generative_scifact_outputs.zip after completion. Manuscript writing remains paused. Source efficacy, semantic grounding, learned sufficiency and integration remain separate unfinished requirements.
