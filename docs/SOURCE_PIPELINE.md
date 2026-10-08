# Source-aware pipeline implementation

This module runs lexical retrieval over a supplied local document collection, filters declared language mismatches, collapses transitive shared-family and exact-text dependencies, and aggregates NLI stance with inverse declared source-rank weights. It emits only `machine_candidate` or `abstain`, with selected evidence and reasons. It is not a completed generative RAG system or an authenticated historical verifier.

Input JSON requires `claim`, `claim_language`, and `documents`. Each document requires `id`, `text`, and `language`; optional fields are `source_url`, `source_rank` (1–5), `is_primary` (boolean), and `provenance_family_id`. Use actual captured source text, not an evidence summary as a substitute for the underlying document. Missing source rank defaults to 5. Unknown provenance is not assumed to be authenticated.

Run on Windows with your own existing input file:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_source_pipeline.py --input evidence_input.json --output artifacts\source_pipeline\candidate.json --require-primary
```

`evidence_input.json` is the name of the user-supplied collection; no fictional source collection is included. `--require-primary` is appropriate for archival attribution checks. The declared primary flag does not authenticate a source. Supported and Refuted are machine stance candidates, never historical gold labels.

The fixed engineering defaults are retrieval pool 50, five retained representatives, two distinct provenance components, maximum stance score at least 0.60, and top-two score margin at least 0.15. These thresholds are heuristics, not trained sufficiency thresholds or calibrated probabilities. Neutral dominance also causes abstention. Missing evidence and incompatible declared language cause abstention without NLI inference. Language codes are compared exactly ignoring case; this is not automatic language detection or multilingual inference.

Shared provenance-family IDs and exact normalized text connect documents transitively within the retrieved pool. The highest ranked lexical representative is retained. Dependencies outside the pool, paraphrased copies without shared metadata, and incorrectly declared metadata are not detected. Repetition invariance tests apply to copies contained in the pool; large floods can displace independent evidence before collapsing. Source weighting uses 1/rank and does not verify dates, URLs, independence, or archival access.

Ablation switches `--no-family-collapse` and `--no-source-weights` are implemented. Disabling collapse allows copies to affect aggregated stance, but the minimum independent-component check remains active. These are software controls; benchmark evaluations of this pipeline and the remaining multilingual and missing-primary-source study requirements are still pending. No manuscript has been written.
