# Fixed-context verdict decoder diagnostic

Post-hoc comparison on the two already observed300-claim climate runs. Load the exact saved verdict prompt text and verify its token count against the original receipt. Same pinned Qwen generator, CPUfloat32, seed369, threads4. No retrieval or generation changes; no explanation generation. Score all three exact label token sequences followed by EOS through causal teacher forcing. Primary decision: largest sum token log likelihood including EOS. Secondary diagnostic: largest mean token log likelihood including EOS. Fixed label order breaks ties. No fitting, threshold search or calibration. Raw verdict metrics only: previous explanations were conditioned on previous verdicts and cannot be attached to changed verdicts or used to evaluate a new guarded answer.

Unlike greedy constrained generation, likelihood scoring uses raw model logits, without generation-time repetition penalties or token masking. This compares two decoder policies; it does not isolate search alone. Scores are not calibrated verdict probabilities. Different label lengths can bias summed or averaged scores. Preserve both measures, regardless of outcome; do not select the better one afterward and call it confirmation. This is not an independent benchmark or source authentication check.

Separate resumable score cache with input/code/protocol identity checks. No new model download. Neural inference must run on Windows. Inputs artifacts/climate_rag_frozen_v1 and artifacts/climate_supplied_evidence_v1. Output artifacts/climate_decoder_comparison_v1.

Run python scripts\compare_climate_verdict_decoders.py from D:\apv-rag-tesla369. Upload JSON outputs as climate_decoder_outputs.zip. No manuscript or GitHub push.
