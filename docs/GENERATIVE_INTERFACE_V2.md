# Generative interface repair smoke test

The first SciFact run failed its combined answer-format contract on all 300 claims. Preserve that result and its frozen code. Version 2 uses a separate output directory and separates two genuine generation calls: a verdict constrained to the three legal label token sequences, followed by a free-form explanation conditioned on that verdict. The application joins these fields with the delimiter. It does not append, infer or fabricate citation IDs. The unchanged citation parser still rejects absent required citations, unknown IDs and empty explanations. Constrained output prevents invalid label strings; it does not establish correct labels.

Run only four hand-constructed synthetic sanity cases first: simple support, contradiction, unrelated evidence and empty evidence. These are software fixtures, not historical annotations or research benchmark results. Save all prompts, model-generated fields and parser decisions. All four verdicts must match the fixtures and all four answers must pass structural checks before a full-run readiness flag is true. Explanation entailment remains unverified even when this flag is true. This is a post-failure development repair, not independent confirmation.

Reuse the pinned local FLAN-T5-base model, CPU float32, four threads and seed 369. No model download or neural retraining is required. Do not run another 300-claim experiment until the smoke export is inspected. Do not modify or overwrite the failed run.

From the activated environment in D:\apv-rag-tesla369:

```powershell
python scripts\smoke_test_generative_interface.py
Compress-Archive -Path .\artifacts\generative_interface_smoke_v2\*.json -DestinationPath .\generative_interface_smoke_v2.zip
```

Send generative_interface_smoke_v2.zip. Manuscript writing remains paused. Full cited generative RAG is not complete.
