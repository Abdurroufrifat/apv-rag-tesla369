# Phase 2C evidence-baseline results

## Run identity

- Dataset: pinned AVeriTeC training split
- Upstream commit: `7c62d1ec8df3fb560d6efe2b85fa191135636f81`
- Internal training records: 2,458
- Internal validation records: 609
- Official development records used: 0
- Random seed: 369
- Probability calibration: sigmoid, five training-only stratified folds

## Model selection

| Input | C | Macro-F1 | Balanced accuracy | ECE |
|---|---:|---:|---:|---:|
| Claim only | 0.25 | 0.2793 | 0.3048 | 0.0663 |
| Claim only | 1.00 | 0.2829 | 0.3077 | 0.0436 |
| Claim only | 4.00 | 0.2767 | 0.3034 | 0.0356 |
| Claim + evidence | 0.25 | 0.4452 | 0.4481 | 0.0718 |
| Claim + evidence | 1.00 | 0.4611 | 0.4649 | 0.0570 |
| Claim + evidence | 4.00 | **0.4627** | **0.4649** | **0.0517** |

The frozen selection rule chose `claim_plus_evidence` with `C=4.0`. Its
Macro-F1 was 0.1798 higher than the best claim-only diagnostic (0.4627 versus
0.2829). This comparison supports using evidence text, but it does not isolate a
causal effect because the two representations contain different information.

## Selected model

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Supported | 0.6250 | 0.4971 | 0.5537 | 171 |
| Refuted | 0.6826 | 0.8363 | 0.7516 | 342 |
| Not Enough Evidence | 0.5660 | 0.5263 | 0.5455 | 57 |
| Conflicting Evidence/Cherrypicking | 0.0000 | 0.0000 | 0.0000 | 39 |

- Macro-F1: 0.4627
- Balanced accuracy: 0.4649
- Multiclass Brier score: 0.4628
- 15-bin expected calibration error: 0.0517

## Selective prediction

| Target coverage | Retained records | Accuracy | Risk |
|---:|---:|---:|---:|
| 50% | 305 | 0.7869 | 0.2131 |
| 80% | 488 | 0.6926 | 0.3074 |
| 100% | 609 | 0.6585 | 0.3415 |

Accuracy increased when the system retained only its most confident cases. This
is the baseline evidence for adding calibrated abstention in later APV-RAG
experiments.

## Limitation that must remain visible

The linear baseline did not correctly identify any validation record in the
`Conflicting Evidence/Cherrypicking` class. Macro-F1 exposes this failure even
though overall accuracy is higher. The next model should address class imbalance
and evidence conflict directly; these baseline numbers must not be presented as
the final APV-RAG result.

The Tesla records were not used for these metrics. No result here authenticates a
Tesla quotation or a “3-6-9 code.”
