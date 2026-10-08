# Frozen held-out excerpt evaluation

The protocol artifact locks the two retrieval methods, pretrained model file
identities, local inference dependency versions, training feature hashes,
training priors, five seeds, decision exponents, and primary analysis. It was
created using training and internal-validation artifacts only. The official
development file was not opened, hashed, or scored in this step; its expected
checksum is copied from the pre-existing pinned dataset specification.

Both methods retain exponent 1. Their baseline uses probability argmax.
The two primary comparisons are selected rule versus baseline within each
method. The endpoint is five-seed mean Macro-F1 difference. Statistical tests
use 2,000 group bootstrap and 10,000 paired group randomization samples with
Holm correction across the two comparisons. A dense-versus-BM25 claim is not
a primary hypothesis of this freeze.

Before scoring, the evaluator must audit connected provenance/claim overlap
between evaluation records and the 2,458 fitted training records using the
existing grouping policy. Independent evaluation records are the primary
population; all 500 official development records form a separately labeled
secondary report. Excluded IDs and reasons must be saved. If there are no
independent records, confirmation is unavailable. No label edits or ad hoc
replacement splits are allowed.

This remains an oracle evidence-excerpt evaluation. The candidate corpus
contains benchmark-selected development answers, with both retrieval methods
receiving the same corpus. It does not demonstrate open-web retrieval, full
APV-RAG performance, or an official AVeriTeC leaderboard score.

The model training procedure and cached training features remain fixed. The
development labels may be used for scoring only. No tuning, retraining on dev,
recalibration on dev, or revised feature/model selection is allowed after
release. Any revision would require a new independent evaluation sample.

The artifact is a checksum-backed local freeze, not an externally registered
preregistration or tamper-proof record. Keep its original bytes. A changed
dependency or code fingerprint must be resolved before evaluation rather
than silently accepted.

## What to do on Windows

Extract the updated consolidated ZIP into `D:\apv-rag-tesla369`. Preserve the
model folders, original inference caches, and virtual environment. The frozen
protocol is already included in `artifacts/frozen_heldout_protocol`; no neural
run or protocol regeneration is needed for this step.

The next implementation step is the evaluator that enforces this protocol,
including the contamination audit. No final-evaluation results exist yet.
