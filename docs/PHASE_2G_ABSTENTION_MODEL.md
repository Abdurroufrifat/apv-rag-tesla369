# Phase 2G: calibrated abstention and selective prediction

Phase 2G converts the frozen Phase 2F probability vectors into explicit
`machine_candidate` or `abstain` decisions. It compares maximum probability,
the top-two probability margin, and one minus normalized predictive entropy.
The score with the lowest internal-validation area under the risk-coverage
curve is selected, with a fixed score order used only to break exact ties.

The analysis uses the 609-record internal validation split. It does not read
the official AVeriTeC development set. The artifact stores its Phase 2F
prediction input and manifest so later validation is independent of mutable
project files. Percentile intervals use 2,000 seeded paired bootstrap samples.

Run from PowerShell in `D:\apv-rag-tesla369`:

```powershell
.\.venv\Scripts\python.exe scripts\run_abstention_model.py
.\.venv\Scripts\python.exe scripts\validate_abstention_model.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Frozen result

Run `run_20261001T061710Z` analyzed all 609 internal-validation records and
used zero official-development records. Normalized entropy produced the lowest
area under the risk-coverage curve and was selected:

| Confidence score | AURC | Paired-bootstrap 95% interval |
|---|---:|---:|
| Normalized entropy | **0.1881** | [0.1547, 0.2234] |
| Maximum probability | 0.1916 | [0.1577, 0.2284] |
| Probability margin | 0.2036 | [0.1685, 0.2407] |

The intervals come from the same 2,000 seeded resamples for every score. Their
overlap means the small AURC difference between normalized entropy and maximum
probability should not be described as a confirmed superiority claim.

The frozen normalized-entropy policy produced these operating points:

| Target coverage | Retained | Abstained | Selective accuracy | Selective risk |
|---:|---:|---:|---:|---:|
| 50% | 305 | 304 | 0.7902 | 0.2098 |
| 60% | 366 | 243 | 0.7705 | 0.2295 |
| 70% | 427 | 182 | 0.7635 | 0.2365 |
| 80% | 488 | 121 | 0.7213 | 0.2787 |
| 90% | 549 | 60 | 0.6958 | 0.3042 |
| 100% | 609 | 0 | 0.6667 | 0.3333 |

These are internal-validation estimates used to freeze the abstention policy.
They are not final test-set results. The official development split remains
sealed for later evaluation.
