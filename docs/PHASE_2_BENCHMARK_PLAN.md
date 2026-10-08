# Phase 2 benchmark plan

## Objective

Test whether provenance-family reasoning and calibrated abstention improve claim
verification when many retrieved pages repeat evidence from the same origin.

## Evaluation tracks

| Track | Dataset | Role | Headline metrics |
|---|---|---|---|
| Primary | AVeriTeC | Real-world claims with web evidence | Yes |
| Transfer | SciFact | Scientific claim verification | Secondary |
| Multilingual | XFEVER | Cross-language robustness | Secondary |
| Archival | Tesla 3-6-9 | Unlabeled stress test | No |

## Required comparisons

Run BM25+NLI, dense retrieval+NLI, standard RAG, source-weighted RAG, two
component ablations, and full APV-RAG. Every method receives the same split and
corpus snapshot.

## Primary endpoints

1. Macro-F1 on the untouched primary test split.
2. Area under the risk-coverage curve for calibrated abstention.
3. RIBS-AUC under 0, 1, 5, 10, and 25 redundant source copies.

## Robustness experiments

- copied-source injection without new evidence;
- OCR corruption at four fixed rates;
- claim paraphrasing at two strengths;
- fabricated citation insertion;
- missing-primary-source conditions;
- multilingual claim/evidence mismatch.

## Statistical analysis

Use paired bootstrap 95% confidence intervals with 2,000 repetitions, paired
approximate-randomization tests, and Holm correction across planned comparisons.
Report effect sizes and seed variation, not only p-values.

## Phase 2 exit criteria

- every dataset has a frozen version and checksum manifest;
- no derivative crosses a split boundary;
- full APV-RAG beats standard RAG on at least one primary endpoint with a useful
  effect size and no material degradation on the other primary endpoints;
- provenance ablation explains the copied-source result;
- sufficiency/abstention ablation explains selective-risk behavior;
- all tables can be regenerated from saved prediction files.
