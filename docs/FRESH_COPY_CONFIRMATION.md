# Fresh supplied-evidence component confirmation

The earlier internal-validation benefit did not transfer to this small climate
cohort. The fixed primary macro-F1 difference at 25 added copies was -0.0285;
the descriptive group-bootstrap 95% interval was [-0.1053, 0.0539]. Neither this
primary result nor the secondary clean-evidence result establishes superiority.
The candidate is not promoted or retuned using these outcomes.

## Cohort and method

Selection was frozen before model scoring. All 48 eligible claims were retained,
in 46 article-connected groups. The complete climate claim/article graph was
used to exclude components touching the 300 prior climate claims, pages used
in earlier climate retrieval, and prior benchmark/training Wikipedia pages.
Previous AVeriTeC, SciFact, FEVER and English XFEVER input claim texts were
excluded by normalized exact match and token-set Jaccard >=0.8. All retained
claims are separate from those inputs under these checks. Near-duplicate
exclusion is heuristic; complete semantic independence cannot be guaranteed.
The dataset was inspected earlier, and the global excerpt pool was previously
indexed. This is fresh claim/page-group confirmation within that public dataset,
not a newly blinded dataset or proof of no model pretraining exposure.

No labels were used to rank or choose claims. All selected claims were scored,
including two DISPUTED cases. Four-label mapping: SUPPORTS to Supported,
REFUTES to Refuted, NOT_ENOUGH_INFO to Not Enough Evidence, DISPUTED to
Conflicting Evidence/Cherrypicking. This is an explicit benchmark adapter;
DISPUTED does not establish historical misattribution or cherrypicking.

The two original fitted classifiers and their argmax rules were frozen.
No model training, calibration, threshold search or language model inference
was performed. Each claim's supplied annotated sentences, with all annotation
labels and votes removed, became ordered extractive answers. Wikipedia URLs
identify the supplied articles; they do not authenticate publisher origin.
All interventions add exactly 0, 1, 5, 10 or 25 copies using the existing
same-source injection. The evidence is supplied by the benchmark, not retrieved.

## Results

| Added copies | Control macro-F1 | Augmented macro-F1 |
| --- | --- | --- |
| 0 | 0.1253 | 0.1381 |
| 1 | 0.1130 | 0.1417 |
| 5 | 0.1151 | 0.1429 |
| 10 | 0.1282 | 0.1568 |
| 25 | 0.1829 | 0.1543 |

Clean macro-F1 difference was +0.0128, with interval [-0.0011, 0.0442].
Accuracy at 25 copies was 15/48 for the control and 11/48 for the augmented
model. Mean absolute support-probability shift fell from 0.1376 to 0.0692;
less score movement did not produce higher verdict accuracy here.
Neither model predicted the Not Enough Evidence or conflicting class in the
clean condition. Poor transfer under domain and evidence-format shift remains
an observed failure, not a validated explanation of its cause.

Uncertainty uses 2,000 paired article-group bootstrap draws, seed 369, with
fixed four-label macro-F1. Rare classes, small groups and one training seed
limit the interpretation. No statistical result is used to select another model.
This comparison does not evaluate retrieval, explanations, abstention policy,
source authentication, historical Tesla labels or the complete final RAG method.
Those remaining charter requirements are not marked complete.

## Verification and Windows command

All 353 tests passed with 66 existing sklearn warnings. Local fitted-model
replay reproduced all 480 probability vectors to absolute tolerance 1e-12.
Independent sklearn recalculation matched the saved accuracy and macro-F1.
The supplied verifier checks frozen inputs/code/model hashes, row coverage,
probability validity, metrics and group-bootstrap intervals without loading models.

Extract APV-RAG_Fresh_Copy_Confirmation.zip directly into
`D:\apv-rag-tesla369`, retaining the previous trained-copy update. Run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_fresh_copy_confirmation.py --verify
```

No retraining is needed. Outputs are in
`artifacts/fresh_copy_confirmation_v1`. Existing experiments are preserved.
Manuscript drafting and GitHub publishing remain paused.
