# Training with copied evidence

This fixed single-seed exploratory experiment compares two matched logistic
classifiers on the original 2,458/609 AVeriTeC split. Both use Phase 2E text and
disagreement features, balanced class weights, C=1.0 and seed 369. These
probabilities are uncalibrated. The control differs from the earlier calibrated
Phase 2F baseline and must not be described as its exact reproduction.

The control trains on original evidence. The augmented arm trains on each
original plus variants with 5 and 10 exact same-source answer copies. Each
parent's weights sum to one. Labels remain unchanged; labels and justifications
are removed from model inputs. Claim/article connected groups are checked for
train/validation overlap. Vocabulary and scaling use training inputs only.

The two models are evaluated on all 609 original validation claims with 0, 1,
5, 10 and 25 injected copies. Macro-F1, accuracy and support-probability shift
are saved along with both fitted models and every probability vector. No
official development record is read and no threshold or candidate is selected.
Because internal validation was already observed, these results are exploratory.
Copy invariance, if improved, does not establish provenance independence,
historical authentication, calibrated confidence or general RAG efficacy.

## Results and verification

| Added copies | Control macro-F1 | Augmented macro-F1 |
| --- | --- | --- |
| 0 | 0.5373 | 0.5443 |
| 1 | 0.5488 | 0.5453 |
| 5 | 0.4436 | 0.5408 |
| 10 | 0.3587 | 0.5425 |
| 25 | 0.2581 | 0.5524 |

These are single-seed exploratory results, not independent confirmation.
Full suite: 351 tests passed. Fitted-model replay reproduced all 6,090
probability vectors and summary metrics in the recorded training environment.

Extract the update directly into `D:\apv-rag-tesla369`. Check saved results
without training, model loading or additional packages:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\check_trained_copy_results.py
```

This checks source/code/output hashes, validation coverage, upstream labels,
probabilities and all saved metrics. Optional fitted-model inference replay:
`python scripts/run_trained_copy_robustness.py --verify`. That replay requires
compatible recorded packages (scikit-learn 1.8.0, NumPy 2.3.5, SciPy 1.17.0,
joblib 1.5.3); do not change your existing environment merely to check the JSON.
Models and outcomes are in `artifacts/trained_copy_robustness_v1`.
