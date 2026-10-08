# Phase 2E: evidence-disagreement modeling

Phase 2E tests whether label-free evidence structure improves the rare
`Conflicting Evidence/Cherrypicking` class. It compares the frozen Phase 2D
hybrid word/character representation, twelve structural disagreement features,
and their union. Models use class-balanced logistic regression and training-only
sigmoid calibration.

Structural features use only questions, answer text, answer type, and source
medium. They exclude labels, justifications, speakers, reporting sources,
fact-check article URLs, source URLs, and the official AVeriTeC development set.

Selection is frozen: highest validation Macro-F1, then conflict-class F1,
balanced accuracy, and lower C. A gain is reported only if measured; the
ablation remains valid if text-only wins.

Run in PowerShell from `D:\apv-rag-tesla369`:

```powershell
.\.venv\Scripts\python.exe scripts\run_disagreement_model.py
.\.venv\Scripts\python.exe scripts\validate_disagreement_model.py
.\.venv\Scripts\python.exe -m pytest -q
```

Each run creates a new timestamped directory under
`artifacts\phase2e_disagreement_model` and never overwrites an earlier run.

## Frozen validation result

Run `run_20260915T074043Z` selected `text_plus_disagreement` with `C=1.0`.
It achieved Macro-F1 0.5193 and balanced accuracy 0.4971. Conflict-class F1
was 0.1702 (precision 0.5000, recall 0.1026, support 39), compared with
0.0488 for the strongest text-only Phase 2D setting. The disagreement-only
model was weak (best Macro-F1 0.2547), showing that structural features help
when combined with semantic text features rather than replacing them.
