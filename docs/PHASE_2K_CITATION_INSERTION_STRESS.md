# Phase 2K: fabricated-citation insertion stress test

Phase 2K measures how the frozen verifier reacts when synthetic citations that
repeat the claim are appended to validation evidence. The inserted records use
the reserved `.invalid` domain and carry an explicit
`synthetic_test_citation` marker. They are experiment fixtures, not sources.

The experiment uses insertion counts 0, 1, and 5. Claims, labels, genuine
evidence, training data, and frozen split membership do not change. It reports
Macro-F1, prediction flips, support-probability movement, confidence movement,
and 2,000 paired-bootstrap intervals. The audit file records every insertion.

The official AVeriTeC development set remains sealed. Tesla material is not
used for training, tuning, or quantitative accuracy claims.

Run from PowerShell in `D:\apv-rag-tesla369`:

```powershell
.\.venv\Scripts\python.exe scripts\run_citation_insertion_stress.py
.\.venv\Scripts\python.exe scripts\validate_citation_insertion_stress.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Frozen result

Run: `run_20261001T074029Z`

| Inserted citations | Macro-F1 | Prediction flips | Mean support-probability shift | Mean absolute support shift | Mean confidence change | Confidence increase rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5193 | 0.00% | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| 1 | 0.4766 | 7.23% | +0.0037 | 0.0364 | -0.0174 | 27.26% |
| 5 | 0.4475 | 17.90% | -0.0569 | 0.1072 | -0.1555 | 9.20% |

The paired-bootstrap 95% interval for Macro-F1 change was
`[-0.0725, -0.0149]` with one inserted citation and
`[-0.1208, -0.0273]` with five. Both intervals exclude zero. The interval for
mean support-probability shift included zero at one insertion
(`[-0.0008, 0.0079]`) and was negative at five insertions
(`[-0.0674, -0.0463]`). Each interval used 2,000 resamples.

The result shows that repeated claim-like synthetic evidence can materially
change the frozen verifier's decisions and lower accuracy. It does not show
uniform confidence inflation or a uniform shift toward the Supported class.
At five insertions, confidence and Supported-class probability fell on average,
even though the absolute probability movement and prediction-flip rate grew.

This is a controlled insertion test, not an estimate of real-world fabricated
citation prevalence. The inserted text follows one deterministic template.
Broader conclusions require additional insertion styles and external systems.

The run used 609 AVeriTeC validation records, five-fold sigmoid calibration,
seed 369, and no official development records. The audit contains 3,654 marked
synthetic citation rows: 609 for the one-citation setting and 3,045 for the
five-citation setting.
