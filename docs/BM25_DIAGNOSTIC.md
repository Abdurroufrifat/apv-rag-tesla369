# BM25 evidence-excerpt diagnostic

The lexical retrieval component is implemented without new dependencies. It
uses BM25 with k1=1.2, b=0.75, case-folded word tokens, and document-index tie
breaking. Queries contain the claim text only. Verdict labels, justification,
speaker, and fact-checking article text are excluded from the index.

The diagnostic corpus contains 1,540 unique (answer text, source URL) pairs
provided by the benchmark for the 609 internal-validation claims. Relevance is
defined by membership in a claim's upstream answers. URLs identify documents
but are not scored. No official development record is read.

| Metric | Measured value |
|---|---:|
| Mean evidence-document recall at 5 | 0.3610 |
| Mean evidence-document recall at 10 | 0.4210 |
| Mean binary-relevance NDCG at 10 | 0.3887 |

This is an oracle corpus assembled from benchmark-selected answers. It is not
an open-web retrieval evaluation, an AVeriTeC official score, or a BM25+NLI
classification result. Answer excerpts may be fact-checker-written summaries
and are not equivalent to original source documents. The diagnostic cannot
establish end-to-end verification accuracy or historical attribution.

Artifacts are saved in `artifacts/bm25_excerpt_diagnostic`: the corpus, ranked
document IDs and scores, relevance IDs, source/split hashes, and summary.

The runner refuses to overwrite that saved diagnostic. Reuse the included
results; no rerun is necessary. The BM25 component can be reused when the real
source-document corpus and NLI backend are available.

## Remaining dependency

This execution environment has no torch, transformers, sentence-transformers,
or cached NLI model weights. A pinned pretrained NLI model and its tokenizer
must be supplied before the NLI comparison can run. Do not substitute keyword
rules or benchmark labels for entailment predictions. Dense retrieval requires
a separately pinned embedding model. Model labels must be mapped explicitly;
three-way NLI scores do not directly supply AVeriTeC's fourth conflict label.
