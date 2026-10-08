# Fresh retrieval and feature extraction integration, v1

This engineering run connects raw frozen corpus retrieval, learned context features and the existing generation controller. It evaluates the same 300 SciFact and 300 retrieved-climate claims already observed in previous development experiments. It is not an untouched test or a new confirmatory performance comparison. All earlier results remain unchanged. No new human annotation, manuscript or GitHub publication is authorized by this protocol.

## Frozen computation

1. Verify raw corpus/claim files, original result receipts, frozen learned heads, pinned local model bytes and the original five-package feature environment before neural execution. Record source, code, settings and protocol hashes. Refuse identity changes on resume or completed-run overwrite.
2. Build BM25 anew from sorted raw corpus abstracts. Retrieve three positive-score documents using the full claim. Select up to three lexical BM25 sentences in rank order per document, with the existing first-three fallback. Clip the claim to 64 and each selected passage to 96 tokens using the pinned Qwen tokenizer. Require exact agreement with original frozen baseline contexts. This checks implementation equivalence; it does not produce new evidence or authenticate sources.
3. Collapse declared source families and identical normalized text before extracting features. On the first run, infer every nonempty context with the pinned English DeBERTa NLI model (256 pair tokens) and MiniLM embedding model (384 tokens), CPU float32, seed 369, four threads and deterministic algorithms. Do not import the earlier feature cache. Resume only features produced by this run, bound to the exact shown claim and collapsed context digest.
4. Aggregate C/E/N mean, maximum and standard deviation (nine features), plus cosine mean, maximum, minimum and standard deviation (four features). Apply the unchanged serialized train-only logistic heads for NLI, embedding and combined features, with their original scaling and threshold .5. Do not retrain, tune thresholds or calibrate on these claims. These heads predict the constructed matched-rationale target, not universal evidence sufficiency.
5. Execute all four policies, including the no-gate control. Empty context and a gate below .5 abstain before requesting verdict or explanation. The existing controller applies constrained verdict generation and the numeric-value guard after explanation generation. The numeric guard is not a semantic explanation validator.

## Response reuse and memory

Only generation responses are seeded from the original received baseline records, indexed by response kind and the SHA-256 of the complete rendered prompt. A conflicting response is refused. Exact matches reuse their recorded answer and reported token count, explicitly labelled `prior_exact_prompt`. New prompts invoke the existing local Qwen backend with its frozen greedy decoding, label-token trie, 1024 input and 128 output limits. New/current-run resumed responses are labelled `current_run_live_or_resume`. No internet model download is performed.

Feature models leave scope and garbage collection runs before Qwen is lazily loaded for a prompt miss. If every prompt matches, Qwen inference is unnecessary. Model bytes and Qwen clipping are still checked on the user PC. Identical cached generations do not demonstrate independent generation reproducibility or quality improvement. Request counts and cache counts are not measured latency savings. Resumed current-run responses are not counted as newly executed calls in a later invocation.

## Output and verification

The runner writes `artifacts/fresh_pipeline_v1`: input identity, context-bound feature cache, execution progress, 2400 policy records, distinct used responses, metrics and a complete JSON output receipt. SQLite is resumable locally and excluded from the shared export. `summary.json` compares fresh features, gate probabilities and threshold decisions with the prior feature cache descriptively; numerical differences are reported and do not cause automatic threshold adjustment. Retrieval/clipping differences stop the run for inspection.

The non-neural verifier checks source/code/protocol identities, raw BM25 IDs/scores and sentence selections, prior clipped bytes, feature shapes/normalization/aggregates/digests, frozen head calculations, every gate-before-generation flow, original labels, prompt/response binding, exact cached-response provenance, numeric guards, output coverage and metrics. It does not independently rerun neural inference, read model weights or SQLite, or recount tokenizer tokens. The runner's user-PC checks and the local verifier's checks must remain distinct.

The Windows launcher installs the editable local package if necessary, resumes an incomplete run or verifies an existing completed one, then creates `fresh_pipeline_outputs.zip` containing the six JSON outputs, receipt and verification reports. It excludes model weights and SQLite. Run `scripts\run_fresh_pipeline_windows.cmd` from the existing project. Input-only preflight is available through `python scripts/run_fresh_pipeline.py --preflight`; it is not a neural result.

## Remaining scope

Source authentication, factual explanation validation, complete integrated multilingual retrieval/gating and gate robustness/calibration remain open against the binding charter. These observed English integration results cannot establish general historical authenticity or journal acceptance. Tesla outputs remain machine candidates or abstentions. No manuscript is written before the user's permission, and GitHub remains paused.
