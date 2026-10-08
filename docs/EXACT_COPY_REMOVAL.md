# Optional exact-copy removal

Implemented a narrow optional preprocessor: within each question, remove only
identical complete answer objects with a nonempty source URL and answer text.
Keep the first occurrence and original order. Different text, URLs, metadata,
questions and missing-source answers remain intact. Input objects are not mutated.
This is not enabled automatically in earlier frozen pipelines.

Across 48 already observed supplied-evidence claims and copy counts 0, 1, 5,
10 and 25, all 240 record-identity checks recovered the original clean records
exactly. At 25 copies, all 1,200 injected answers were removed, while all 240
original answers remained. The previous domain rule retained only 48 answers.

Because the frozen deterministic classifier receives identical original inputs,
its saved clean-control probabilities were reused explicitly. No new prediction
computation, neural inference, training or gold-dependent preprocessing occurred.
The expected control result remains 9/48 correct, macro-F1 0.1253, with 16 false
Supported predictions among 31 non-support references. Exact-copy removal does
not improve the underlying clean classifier. It also cannot guarantee a higher
stressed accuracy; the earlier copied inputs had higher accuracy through changed
guesses despite more false Supported decisions.

All 358 tests passed with 66 existing sklearn warnings. Tests cover copied versus
distinct source answers, unchanged inputs, missing URLs and separate questions.
The diagnostic checked source/code/output identities and exact record restoration.
This is post-hoc mechanism analysis on observed public cases, not independent
confirmation. Paraphrases, mirror URLs, source-family authenticity, factual
explanation truth and full RAG generalization remain outside this rule.

Extract APV-RAG_Exact_Copy_Removal.zip directly into `D:\apv-rag-tesla369`.
No new Windows run or model download is needed for the included results.
Module: `src/apv_rag/exact_copy_removal.py`. Experimental evaluation and receipts:
`artifacts/exact_copy_removal_v1`. No manuscript or GitHub push.
