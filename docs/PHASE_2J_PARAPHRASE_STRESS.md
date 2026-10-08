# Phase 2J: controlled claim-paraphrase robustness

Phase 2J applies conservative, auditable phrase substitutions to benchmark
claims at mild and strong settings. It does not use a language model or claim
that the rules produce natural paraphrases for every record. Names, numbers,
labels, evidence answers, questions, URLs, and split membership remain fixed.

The experiment evaluates the frozen Phase 2E text-plus-disagreement verifier
at `C=1.0`. It reports transformation coverage, all-record and changed-only
Macro-F1, prediction flips, confidence changes, and 2,000 paired-bootstrap
intervals. Every original and transformed claim is stored in the artifact for
inspection.

The official AVeriTeC development set remains sealed. Tesla material is not
used for training, tuning, or quantitative accuracy claims.

Run from PowerShell in `D:\apv-rag-tesla369`:

```powershell
.\.venv\Scripts\python.exe scripts\run_paraphrase_stress.py
.\.venv\Scripts\python.exe scripts\validate_paraphrase_stress.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Frozen result

Run: `run_20261001T072234Z`

| Setting | Changed claims | Coverage | All-record Macro-F1 | Changed-only Macro-F1 | Flip rate | Changed-only mean absolute confidence change |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 / 609 | 0.00% | 0.5193 | not applicable | 0.00% | not applicable |
| mild | 30 / 609 | 4.93% | 0.5193 | 0.5838 | 0.00% | 0.0037 |
| strong | 30 / 609 | 4.93% | 0.5193 | 0.5838 | 0.00% | 0.0036 |

For both nonzero settings, the paired-bootstrap 95% interval for prediction
Macro-F1 change was `[0.0000, 0.0000]` in the full validation set and in the
changed-only subset (2,000 resamples). Macro-F1 did not change.

These results support only a narrow robustness statement: the frozen verifier
was invariant to the substitutions that fired in this rule set. The 4.93%
coverage is too small to establish broad semantic-paraphrase robustness. The
complete transformation audit is retained so readers can inspect exactly which
claims changed, and unchanged claims are reported separately rather than being
allowed to dilute the changed-only analysis.

The frozen run used 609 AVeriTeC validation records, five-fold sigmoid
calibration, seed 369, and no official development records.
