# Sentence-selected RAG v3

Exploratory follow-up after the context visibility audit. Keep the same three retrieved documents, pinned models, constrained verdict decoder, prompts and 96-token per-document budget. Rank each abstract's sentences with local BM25 using the original claim, select up to three positively scoring sentences in rank order, then clip the joined selection to 96 generator tokens. If no sentence has a query match, use the first three. Retain selected sentence indices for provenance; clipping can remove part or all of later selected sentences. Indices do not mean the whole sentence was shown.

No gold sentences, verdicts or labels enter selection or generation. Gold evidence may be used only in post-run analysis. Same observed SciFact cohort: this is exploratory and cannot establish independent confirmation. Lexical sentence scores are not learned sufficiency or source authentication. Explanations remain unverified.

Output and resumable cache: artifacts/sentence_rag_scifact_v3. The original runs are preserved. Run python scripts\run_sentence_rag_scifact.py from D:\apv-rag-tesla369 in the existing activated environment. No new model download. Upload the JSON outputs as sentence_rag_outputs.zip. Neural inference is not available in the development environment.
