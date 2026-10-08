# Archive capture timing ablation

This exploratory AVeriTeC experiment asks whether five label-free archive
snapshot features add predictive information to the already selected Phase 2F
text-and-disagreement classifier. It does not add source publication dates:
`claim_date` dates the claim and Wayback URL timestamps date archived captures.
Neither authenticates a publisher, proves an article existed on the claim date,
or establishes the source's first publication date.

The input is the pinned upstream training JSON and the original 2,458/609
Phase 2B training/internal-validation indices. The frozen Phase 2F `phase2e`
`C=1.0` predictions supply the control. The only new fit uses the same
word/character TF-IDF, disagreement features, five-fold sigmoid-calibrated
balanced logistic regression, seed 369 and `C=1.0`, plus claim-date presence,
archive-capture fraction, share of captures after the claim, median signed lag,
and median absolute lag. Dates that are absent or malformed stay missing; an
invalid Wayback timestamp is never interpreted as a valid source date.

No official development record is used by the code. That set and the internal
validation set were both observed in earlier work; any comparison here is
exploratory and cannot confirm a revised method. AVeriTeC supplies the
answer-bearing evidence excerpts. This experiment does not evaluate retrieval,
open-web source independence, Tesla attribution, or the generative controller.

On 609 internal-validation records, 599 have a parseable claim date and 533
have at least one parseable Wayback capture. The frozen control obtains macro-F1
0.5193 and balanced accuracy 0.4971. The augmented classifier obtains macro-F1
0.4796 and balanced accuracy 0.4742. Its decision changes on 32 records:
13 become correct, 17 become incorrect, and 2 switch between wrong labels.
This negative result does not support adding archive-capture timing to the
selected predictive model. The existing Phase 2F selection stays unchanged.

The checked run and verification receipt are stored in
`artifacts/capture_timing_ablation_v1`. To verify the included result after
extracting the package directly into `D:\apv-rag-tesla369`:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_capture_timing_ablation.py --verify
```

The verification hashes inputs, code and outputs, aligns validation indices,
and replays the recorded macro-F1 from the saved prediction probabilities.
Running without `--verify` attempts a new fit and refuses to overwrite the
included output directory.
