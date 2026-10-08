# Supplied-evidence diagnostic

Post-hoc experiment after observing poor frozen CLIMATE-FEVER transfer. Same300 claims and pinned models. Bypass pooled article retrieval and sentence selection. Supply all five claim-associated dataset evidence sentences, preserving original order; omit evidence labels, votes and entropy. Do not select only supporting/refuting annotations. Each sentence is clipped to96 generator tokens. Claim64tokens, prompt1024 maximum, output128; prompts exceeding the maximum stop explicitly. Same greedy constrained verdict, free explanation, numeric extractorv2 and diagnostic English NLI.

Five passages replace the original three, so this is an evidence-presentation diagnostic with a changed context size, not a controlled causal isolation or ordinary retrieval performance. Association with claims is supplied from annotated data. Annotation evidence may be incomplete or disputed. Clipping and incorrect reasoning may remain. No source-authentication or explanation-grounding claim. Results cannot replace frozen scores or be described as independent confirmation.

Separate output/cache: artifacts/climate_supplied_evidence_v1. Input/code/protocol hashes freeze the diagnostic before inference and prevent changed-input cache reuse. UTF-8 reads explicit. No new model download. No manuscript or GitHub push.

Run python scripts\run_climate_supplied_evidence.py in D:\apv-rag-tesla369. Upload its JSON outputs zipped as climate_supplied_outputs.zip.
