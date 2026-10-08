# Replacement generator smoke test

Three failed development configurations are preserved. This is a separate model comparison using Qwen/Qwen2.5-1.5B-Instruct, pinned revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, with its own cache-free smoke directory. No claim that this replacement will pass has been established. Model choice follows observed failures; this is adaptive development, not independent confirmation.

Use the same four synthetic fixtures. Three nonempty-evidence cases use the tokenizer's official chat template, CPU float32, four threads, seed 369, greedy generation and maximum 128 new tokens. Save complete rendered chat prompts and raw completions. The unchanged structural answer parser requires known labels, explanations and valid source IDs. Additionally reject numeric strings in explanations absent from evidence, ignoring bracketed citations. This conservative lexical guard may reject equivalent numerical notation and cannot detect all factual or negation errors. It is not semantic authentication. Empty evidence bypasses generation and abstains explicitly; that outcome is from the pipeline, not model reasoning.

All three generated verdicts must match their fixture expectations and pass the guards; the empty-evidence case must abstain. Do not weaken the criterion after inspecting outputs. Passing fixtures does not establish research performance or explanation entailment. Full benchmarking remains blocked until the export is inspected. No human review or manuscript is added.

One new model download is required. It is larger than FLAN-T5-base; download only one safetensors weight file. Float32 inference requires more RAM than the downloaded weights and may run slowly on CPU. The script records exact model hashes. Existing models and outputs stay untouched.

From the activated environment in D:\apv-rag-tesla369:

```powershell
python scripts\download_instruction_model.py
python scripts\smoke_test_instruction_model.py
Compress-Archive -Path .\artifacts\instruction_model_smoke_v4\*.json -DestinationPath .\instruction_model_smoke_v4.zip
```

Send instruction_model_smoke_v4.zip. This is only a four-case test, not another 300-claim run.
