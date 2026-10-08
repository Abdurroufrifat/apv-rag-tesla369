# Constrained verdict RAG v2

Exploratory follow-up after v1 produced 119 invalid verdicts. Changes only verdict decoding: greedily permit continuations of tokenizer sequences for Supported, Refuted, Not Enough Evidence; EOS is permitted after a complete label. Explanations remain free generation. This constrains syntax, not factual accuracy or evidence sufficiency. It is not a calibrated classifier and not guaranteed to choose the highest total-probability label.

Same pinned models, cohort, top-three BM25, prompts, clipping, seeds and numeric guards as v1. Separate output directory and cache: artifacts/constrained_rag_scifact_v2. Old results are preserved. Code and protocol identity prevent reuse after changes. No new downloads. No fitting on SciFact labels. Results are exploratory because this benchmark has already been observed. Generation must run on the user's Windows environment; neural inference is unavailable here.

Run from D:\apv-rag-tesla369:

    python scripts\run_constrained_rag_scifact.py

Then upload the JSON outputs zipped with Compress-Archive. Compare coverage, accuracy including abstentions, macro F1, and numeric rejection counts with v1. Do not infer factual improvement from removal of format failures alone.
