# Frozen CLIMATE-FEVER evaluation

Freeze before neural inference. Revision03de61617b10a5c1935f8e08bb0e8ac1ee775356. Use300 deterministically hash-ranked eligible claims, excluding DISPUTED labels because the generator has three classes. No label balancing or outcome-dependent selection. Corpus groups all dataset evidence sentences by article, independent of claim labels; retrieve top3 articles from that common pool. This is an annotation-derived limited pool, not full Wikipedia. It is a different-domain test; pretrained exposure unknown, so strict independence is not established.

Pipeline fixed: pinned Qwen1.5B and English NLI; CPUfloat32 seed369 threads4; BM25 article retrieval top3; per-article sentence BM25 top3 rank order;96tokens perarticle;64claimtokens;1024promptbudget;128newtokens; greedy constrained verdict; free explanation; numeric extractor v2. Labels used only for scoring after generation. No new human annotations. NLI remains diagnostic, not a gate. Source authentication and learned sufficiency remain unresolved. Code/protocol hashes and separate resumable cache prevent silent identity changes.

Primary measures: all-claim macroF1 and accuracy with abstentions as errors. Report raw and guarded results, coverage, answered accuracy and rejected IDs. Do not tune on this cohort after observing results and still call it confirmation. Preserve any failure. No manuscript or GitHub push.

Run python scripts\run_climate_rag.py from D:\apv-rag-tesla369 using the existing activated environment. Output artifacts/climate_rag_frozen_v1. No new model download.
