# Phase 2D: imbalance-aware evidence models

## Method

Phase 2D addresses the Phase 2C failure on the minority
`Conflicting Evidence/Cherrypicking` class. Every candidate uses the same
claim-plus-evidence renderer, balanced logistic-regression class weights,
training-only sigmoid calibration, and the frozen Phase 2B split.

Three representations are compared:

- word TF-IDF with one- and two-word features;
- character-boundary TF-IDF with three- to five-character features; and
- a hybrid union of both feature spaces.

Each representation is evaluated with `C = 0.25, 1.0, 4.0`. Selection uses
validation Macro-F1, then conflict-class F1, balanced accuracy, and lower C.
The official AVeriTeC development split is not read.

## Verified result

| Representation | C | Macro-F1 | Balanced accuracy | Conflict F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Word | 0.25 | 0.4559 | 0.4693 | 0.0000 | 0.0669 |
| Word | 1.00 | 0.4663 | 0.4759 | 0.0000 | 0.0577 |
| Word | 4.00 | 0.4694 | 0.4773 | 0.0000 | 0.0585 |
| Character | 0.25 | 0.4490 | 0.4346 | 0.0952 | 0.0561 |
| Character | 1.00 | 0.4610 | 0.4485 | 0.0952 | 0.0496 |
| Character | 4.00 | 0.4633 | 0.4500 | 0.0976 | 0.0753 |
| Hybrid | 0.25 | 0.4609 | 0.4626 | 0.0488 | 0.0610 |
| Hybrid | 1.00 | **0.4840** | **0.4845** | 0.0488 | 0.0496 |
| Hybrid | 4.00 | 0.4801 | 0.4801 | 0.0488 | **0.0463** |

The frozen rule selected the hybrid model with `C = 1.0`. Compared with the
Phase 2C evidence baseline, Macro-F1 increased from 0.4627 to 0.4840 and
balanced accuracy increased from 0.4649 to 0.4845. Conflict-class F1 increased
from 0.0000 to 0.0488. The character-only model detected more conflict cases,
but its overall Macro-F1 was lower.

The selected model achieved 0.7902 accuracy at approximately 50% coverage,
0.6988 at approximately 80% coverage, and 0.6634 at full coverage. These are
internal-validation results and are not final test results.

## Windows installation

Extract the Phase 2D ZIP directly into:

```text
D:\apv-rag-tesla369
```

Choose **Replace the files in the destination** if Windows asks. Do not create a
second nested `apv-rag-tesla369` folder.

In VS Code, open `D:\apv-rag-tesla369`, then open a new PowerShell terminal.
Run each command separately:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& D:\apv-rag-tesla369\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
python scripts\validate_averitec.py
python scripts\validate_averitec_splits.py
python scripts\validate_imbalance_baseline.py
python -m pytest -q
```

The ZIP already contains the verified Phase 2D run. To produce a new local run,
use:

```powershell
python scripts\run_imbalance_baseline.py
python scripts\validate_imbalance_baseline.py
```

The expected output directory is:

```text
D:\apv-rag-tesla369\artifacts\phase2d_imbalance_baseline\run_YYYYMMDDTHHMMSSZ
```

The run contains metrics, candidate comparisons, aligned predictions, a
confusion matrix, a representation-comparison figure, and an integrity manifest.

## Interpretation limit

Phase 2D is still a classical baseline. The low conflict recall shows that class
weighting and surface-form features are insufficient by themselves. The next
model must compare evidence units directly and represent disagreement between
answers. Tesla records remain an unlabeled archival stress test. These metrics do
not authenticate any Tesla quotation or “3-6-9 code.”
