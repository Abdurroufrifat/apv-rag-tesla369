# Phase 2H: repetition-induced belief shift

Phase 2H measures how much the verifier's support probability changes when
derivative copies are added without adding independent evidence. The fixed copy
counts are 0, 1, 5, 10, and 25.

Two systems are compared. The standard system receives every injected copy.
The family-collapsed system keeps one answer per normalized source domain before
classification. Both systems use the Phase 2E text-plus-disagreement pipeline,
`C=1.0`, class-balanced logistic regression, five-fold training-only sigmoid
calibration, and seed 369.

RIBS is the stressed support probability minus the zero-copy support
probability. The experiment reports signed RIBS, absolute RIBS, copy-count-
normalized RIBS-AUC, and 2,000 paired-bootstrap intervals. It uses only the
frozen Phase 2B training and internal-validation records. The official AVeriTeC
development set remains sealed.

Run from PowerShell in `D:\apv-rag-tesla369`:

```powershell
.\.venv\Scripts\python.exe scripts\run_repetition_stress.py
.\.venv\Scripts\python.exe scripts\validate_repetition_stress.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Frozen result

Run `run_20261001T064153Z` used 2,458 training and 609 internal-validation
records. It used zero official-development records.

| Copies | Standard mean absolute RIBS | 95% interval | Standard mean signed RIBS | Family-collapsed absolute RIBS |
|---:|---:|---:|---:|---:|
| 0 | 0.0000 | -- | 0.0000 | 0.0000 |
| 1 | 0.0336 | [0.0318, 0.0355] | -0.0300 | 0.0000 |
| 5 | 0.0970 | [0.0899, 0.1041] | -0.0840 | 0.0000 |
| 10 | 0.1455 | [0.1348, 0.1561] | -0.1115 | 0.0000 |
| 25 | 0.1794 | [0.1666, 0.1920] | -0.1095 | 0.0000 |

The standard model's copy-count-normalized RIBS-AUC was 0.1329. The
family-collapsed model's RIBS-AUC was 0 because injected copies from an already
represented domain are removed before inference. At 25 copies, the largest
individual absolute support-probability shift in the standard model was 0.7648.

The negative mean signed RIBS values matter: on average, these repeated answer
copies reduced the probability of the `Supported` label. The result therefore
shows sensitivity to repetition, but it does not show that repetition generally
causes false support or overconfidence. The zero shift for family collapse is a
deterministic consequence of the preprocessing rule and should be presented as
an invariance guarantee under this exact perturbation, not as an independently
learned performance gain.

These are internal-validation stress-test measurements used during method
development. They are not final test-set results.
