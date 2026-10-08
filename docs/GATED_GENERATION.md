# Gate before generation

The controller collapses context copies, applies a fixed gate and requests a verdict and explanation only if the gate passes. It then applies numeric extractor v2. Rejected claims have no generation requests. A no-gate control follows the same prompts and numeric policy. Keep all NLI, embedding and combined variants at threshold 0.5; no threshold tuning or winner selection.

Inputs are the 900 verified frozen-context records, with their verified gate probabilities. Retrieval and neural feature extraction are not rerun. The earlier controller replay applied gates after saved generation; this controller checks the gate before invoking the generator callback. Probabilities still measure a constructed annotation-completeness target, whose transfer remains unvalidated.

Cached mode tests execution order using the previously generated answers. It does not run a neural model. Live mode loads the pinned local Qwen generator and runs the same constrained verdict and explanation prompts. CPU float32, seed 369, four threads, 1024 input tokens, 128 new tokens, greedy decoding. It reuses gate features; no NLI or embedding model is loaded. Model bytes, inputs, code, packages and protocol are recorded or checked. Completed outputs cannot be overwritten. Interrupted live runs resume from a prompt-keyed SQLite cache protected by the run identity.

All four policies share identical generations for a passing context. Live prompt caching avoids repeated identical neural calls across policies. Per-policy request counts describe the controller's requests; they are not measured latency or compute savings. Running the no-gate control makes every eligible context reach generation, even when every learned gate rejects it. Choosing one policy permits only that policy's requests, in a separate output folder.

From `D:\apv-rag-tesla369`:

```powershell
python scripts\run_gated_generation.py --backend cached
python scripts\run_gated_generation.py --backend live
```

Default is all four policies. Optional `--policy nli`, `embedding`, `combined`, or `no_gate`. Output directories include backend and policy: `artifacts/gated_generation_live_all_v1` for the main comparison. No new model downloads.

This integrates execution control on frozen contexts. It does not establish source authenticity, provenance-conditioned explanation quality, fresh retrieval, multilingual generation, or safe deployment. Live results are another exploratory run on observed cohorts, not independent confirmation.
