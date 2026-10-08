# Fixed RAG on fresh pipeline claims

This run uses the existing constrained three-class climate RAG algorithm without
retuning. It evaluates all 46 compatible claims from the previously frozen
48-claim/page-group cohort. The two DISPUTED claims are excluded because the
existing generator has no disputed class, matching the earlier climate protocol.
No filtering uses the component's prediction correctness or probabilities.
Claim-only inputs are separate from scoring labels. Supplied claim-associated
sentences from the component test are not used as model contexts: BM25 retrieves
from the unchanged common 1,344-article annotation-derived pool.

The claims have not been used in the earlier neural climate RAG runs, but their
labels and component outcomes are already observed. This is a fixed-method
transfer check with fresh pipeline outcomes, not a blind independent dataset.
Article connections and prior text/page exclusions come from the previous
frozen protocol; semantic independence and pretraining independence are not
established. The pool is limited and annotation-derived, not full Wikipedia.

## Unchanged inference settings

Pinned Qwen2.5 1.5B and English DeBERTa NLI; CPU float32; seed 369; four threads.
BM25 top three positive-score articles; per-article BM25 top three sentences;
96 generator tokens per article; 64 claim tokens; 1,024 input-token ceiling;
128 output tokens; greedy constrained three-label verdict; free explanation.
The existing numeric-integrity v2 guard is preserved. NLI explanation scores
are diagnostic and do not gate output. Source IDs are retrieved context IDs,
not verified explanation citations or source authentication.

Report raw and structurally guarded macro-F1, all-claim accuracy, coverage,
covered accuracy, rejection reasons and explanation NLI diagnostics. Abstentions
are errors for all-claim accuracy. Do not tune on these outcomes or call the
complete original charter satisfied. These classifiers do not integrate the
trained-copy model, which remains an unpromoted component.

## Windows command

Extract the update directly into `D:\apv-rag-tesla369`, keeping previous
updates and your downloaded models. Run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_fresh_climate_rag.py
```

The command checks frozen inputs, model identities and prompt budgets, then runs
retrieval, generation, explanation checks and scoring. Interrupted generation
resumes using a separate cache only when all identities match. Existing runs
are preserved. A completed run refuses overwrite.

Send `D:\apv-rag-tesla369\fresh_climate_rag_outputs.zip` after completion.
It contains the small JSON outputs and receipts, excluding weights and the
SQLite cache. No training, model download, manuscript or GitHub push is needed.

For a cheap input check without neural inference:
`python scripts/run_fresh_climate_rag.py --check-inputs`.
The model weights are absent from the preparation workspace, so no neural
performance result or generation runtime is claimed here.
