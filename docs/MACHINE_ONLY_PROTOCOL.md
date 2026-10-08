# Machine-only scientific protocol (binding)

## Decision

This project does not collect, request, or claim new human annotations. Phase 1C
review packets are retired artifacts and must not be generated or distributed.

## What supplies ground truth

All quantitative verdict evaluation uses the unchanged labels supplied by
published benchmarks. APV-RAG may transform retrieval corpora and attach
provenance metadata, but it must preserve each upstream label and record the
upstream example identifier.

## Tesla boundary

The Tesla 3-6-9 collection is an unlabeled archival stress test. It may measure:

- retrieval stability across source copies;
- provenance-family collapse;
- OCR and translation disagreement;
- citation validity;
- confidence and abstention behavior.

It may not be used to train or tune a verifier, compute accuracy, assign a gold
verdict, or claim that Tesla authored a quotation. Allowed case-study outputs are
`machine_candidate` and `abstain`. Multiple OCR, translation, or language-model
outputs are repeated measurements from machines, not independent human evidence.

## Claims permitted in the paper

The paper may claim an improvement only when it is supported on held-out benchmark
labels, accompanied by uncertainty intervals and paired tests. The Tesla case may
illustrate system behavior, but it cannot establish historical truth.

## Reproducibility rules

1. Freeze dataset versions, licenses, checksums, and upstream IDs.
2. Use official train/development/test splits where available.
3. Generate perturbations after splitting.
4. Keep every derivative and provenance family in its parent's split.
5. Tune thresholds and calibration on validation data only.
6. Run all systems with the same retrieved corpus and five fixed seeds.
7. Report failures, abstentions, compute, and missing-source rates.
8. Never silently map insufficient evidence to false.

## Publication interpretation

This is a machine-learning systems and evaluation paper, not a historical proof of
a “369 code.” Journal acceptance cannot be guaranteed; publishability depends on
measurable benchmark gains, robust ablations, reproducibility, and honest scope.
