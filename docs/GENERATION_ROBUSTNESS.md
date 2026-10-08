# Repeated-source and order generation stress

Use all 300 frozen SciFact contexts and 300 retrieved-climate contexts from the verified live controller. Preserve the same clipped claim/evidence, pinned Qwen generator, package versions and prompt templates. No retrieval or new source search is performed. Labels score results only.

Four conditions per claim: baseline; three copies of the first source appended without collapse; the copied context after the existing first-family/exact-text collapse; original sources in reverse order. Copies preserve the first source's text and declare its family ID, with unique synthetic IDs. The collapsed condition must reconstruct the baseline exactly. Declared families and identical text do not establish source authenticity or independent evidence.

Full verdict comparison: 600 claims times four conditions, 2400 records. Constrained English three-label greedy decoding. For explanations, choose 60 claim IDs per cohort by SHA-256 of `APV-generation-robustness-v1:{cohort}:{id}` without reading labels or scores. Include the same IDs in all four conditions: 480 explanation records. Numeric extractor v2 guards only that subset. Full verdict metrics and explanation-subset metrics stay separate.

Baseline and collapsed-control answers reuse verified frozen responses for identical prompts. Seed 600 verdict and 120 selected explanation prompts into the protected SQLite cache. Only new copy/order prompts invoke the generator. Baseline and collapsed invariance is enforced by construction and cache reuse; it is not an independent neural robustness result. Raw-copy and reversed-order responses are new model inference. Report changed-verdict rates and paired grouped accuracy/F1 differences; a flip need not be an error, and a stable wrong answer remains wrong.

CPU float32, seed 369, four threads, greedy one-beam decoding, 1024 input tokens and 128 new tokens. Reject budget overflow; do not silently truncate copied prompts. Require the original controller's model and package versions. Hash sources, baseline receipt, model declarations, code, protocol and settings before inference. Refuse stale-cache reuse and completed-run overwrite; resume interrupted runs. No downloads.

This is a synthetic, ungated generator comparison with an exact-copy normalization control. It does not measure paraphrased-copy defenses, unknown provenance, independent-source grouping, noisy retrieval, source authentication, explanation correctness, latency or safe abstention. Keep all outcomes without prompt/threshold tuning on these observed cohorts. Learned-gate behavior is not evaluated here.

From the activated environment in `D:\apv-rag-tesla369`:

```powershell
python scripts\run_generation_robustness.py
Compress-Archive -Path .\artifacts\generation_robustness_v1\*.json -DestinationPath .\generation_robustness_outputs.zip -Force
```

Send `generation_robustness_outputs.zip`. Rerun the generation command to resume an interruption. This stage writes no manuscript and publishes nothing.
