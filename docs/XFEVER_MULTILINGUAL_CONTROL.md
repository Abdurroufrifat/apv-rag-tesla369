# Multilingual NLI model control

This exploratory comparison was specified after inspecting the English-model XFEVER results. It is not an independent confirmation or a full multilingual retrieval evaluation.

Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, revision 8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c. Download only this revision and preserve the generated model manifest. Model training overlap with FEVER-derived claims cannot be ruled out; report this limitation when interpreting scores.

Use the same checksummed eleven XFEVER files, 600 aligned rows each. One supplied evidence premise per claim, maximum 256 tokens, CPU four threads, seed 369, deterministic float32 inference, direct three-class argmax. Read declared model labels and reorder to contradiction/entailment/neutral before scoring. No fitting, translation, threshold tuning or label-based evidence selection. Cache identity includes inputs, model, package versions, code and protocol hashes. Completed outputs cannot be overwritten.

After receiving outputs, verify all 6,600 predictions and compare with the archived English-model outputs. Use connected groups sharing claim ID or English page, paired group bootstrap intervals and group swaps; correct eleven model comparisons with Holm. Do not select languages or settings based on these results. Marginal intervals are not simultaneous intervals. Results remain exploratory and may reflect both training differences and language coverage.

Windows commands from the activated project environment:

```powershell
python scripts\download_multilingual_nli.py
python scripts\run_xfever_multilingual.py
python -m pytest -q
Compress-Archive -Path .\artifacts\xfever_multilingual_fp32_v2\*.json -DestinationPath .\XFEVER_multilingual_outputs.zip
```

The download requires Hugging Face access and sufficient disk space. Evaluation uses local files only. Send XFEVER_multilingual_outputs.zip for verification and paired analysis. No manuscript is written in this step.


## Inference repair v2

A user run stopped at probability validation. Raw probabilities were not supplied, so its precise numeric cause remains unconfirmed. The pinned model config declares torch_dtype float16. The runner now explicitly requests float32 weights, avoiding model-config-dependent precision, and corrects only row-sum drift within 0.001. Nonfinite, negative, out-of-range or substantially unnormalized values still fail with their actual values in the error. This preserves argmax. A separate output/cache directory prevents mixing the failed run with repaired inference. The original English runner and its probability checks are unchanged. No model redownload is required.
