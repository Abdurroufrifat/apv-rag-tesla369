# Phase 2F: provenance-family and duplicate-evidence modeling

Phase 2F tests whether source dependence adds useful signal beyond Phase 2E.
The model unwraps Wayback URLs, removes tracking parameters, groups evidence by
source domain, and measures repeated URLs and near-identical normalized answers.

The frozen comparison contains the Phase 2E feature set, provenance-only
features, and their union. Labels, justifications, speakers, and reporting-source
metadata are excluded from provenance features. Model selection uses validation
Macro-F1, conflict-class F1, balanced accuracy, and lower C, in that order.

The official AVeriTeC development set remains sealed. Every run stores its exact
split files inside the run directory so validation does not depend on the
current operating system or later changes to the project split files.

## Frozen result

Run `run_20260915T085322Z` used 2,458 training and 609 validation records. It
used zero records from the official development set. The frozen selection rule
retained the Phase 2E representation at `C=1.0`:

| Representation | C | Macro-F1 | Balanced accuracy | Conflict F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Phase 2E | 1.00 | **0.5193** | **0.4971** | **0.1702** | 0.0485 |
| Provenance only | 0.25 | 0.3225 | 0.3575 | 0.0000 | 0.0328 |
| Phase 2E + provenance | 0.25 | 0.4962 | 0.4828 | 0.0889 | 0.0598 |

The table reports the best Macro-F1 setting within each representation. The
simple provenance features did not improve the Phase 2E classifier, so they are
not promoted into the selected predictive model. This is a negative ablation
result rather than evidence that provenance is irrelevant: domain-family
counts, duplicate URLs, and normalized-answer repetition may be too coarse to
represent source independence or evidential quality. Later phases should test
provenance through retrieval constraints, calibrated abstention, or richer
source graphs instead of assuming that repetition is an additive label signal.

At the selected operating point, validation accuracy was 0.6667. Selective
prediction reduced empirical risk from 0.3333 at full coverage to 0.2098 at
approximately 50% coverage. These values are validation estimates, not final
test-set claims.

Run from PowerShell in `D:\apv-rag-tesla369`:

```powershell
.\.venv\Scripts\python.exe scripts\run_provenance_model.py
.\.venv\Scripts\python.exe scripts\validate_provenance_model.py
.\.venv\Scripts\python.exe -m pytest -q
```
