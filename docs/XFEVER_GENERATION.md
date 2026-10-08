# Multilingual supplied-evidence generation

Evaluate all eleven frozen XFEVER test.6h files: English and machine/human translations in Spanish, French, Indonesian, Japanese and Chinese. Each contains 600 aligned rows, including repeated claim IDs. Reuse the upstream labels and translations; no new human review, label fitting or translation is performed. This is generation from supplied evidence. It does not evaluate retrieval or source authentication.

Use the existing pinned Qwen generator and constrained three-label greedy verdict decoder. Every row receives an English label: Supported, Refuted or Not Enough Evidence. All 6600 verdict records contribute to full-set metrics. English is the aligned comparison. Preserve every language/translation result, including failures.

Explanations are limited to 60 unique claim IDs ranked by SHA-256 of `APV-multilingual-generation-v1:{id}` from the 585 English IDs. Selection does not read labels or scores. Include every row with a selected ID in every file, preserving duplicate claim rows. The aligned explanation cohort can therefore exceed 60 rows per file. Ask for an explanation in the input language. Language compliance and correctness are not verified by this request.

Apply numeric extractor v2 only to the explanation subset and score guarded results only on that subset. Never mix missing explanations in the other rows with explanation rejection. The numeric check does not establish semantic correctness, convert units or cover all non-Latin number words/numerals. There is no learned completeness gate: the existing head was trained on English constructed rationale coverage; multilingual transfer is unvalidated.

CPU float32, seed 369, four threads, greedy decoding, one beam, at most 128 new tokens. Claim budget 192 Qwen tokens, evidence budget 512, total input budget 1024. Record original/shown token counts and clipping flags. Reject total-budget overflow without silent truncation. Language tokenization differences and clipping are evaluation limitations. No setting changes based on target scores.

Cache responses by prompt/kind in SQLite. Record dataset/model/code/protocol/package hashes before inference. Reject cache reuse after identity changes and refuse completed-run overwrite. Model files are local; no downloads. The generation backend is shared with the verified gate controller, but this experiment does not apply a gate. Request/cache counts do not measure runtime savings.

After receiving results, verify all 6600 outputs and explanation selection, then use paired claim/page groups for language comparisons. Report translation-origin and clipping effects. The cohorts and benchmark have already been observed in NLI experiments; this is exploratory generator evaluation. Pretraining overlap cannot be ruled out. Do not claim independent confirmation or multilingual explanation quality.

From the activated environment in `D:\apv-rag-tesla369`:

```powershell
python scripts\run_xfever_generation.py
Compress-Archive -Path .\artifacts\xfever_generation_v1\*.json -DestinationPath .\xfever_generation_outputs.zip -Force
```

Send `xfever_generation_outputs.zip`. If interrupted, rerun the same generation command to resume. Do not change input/code/package versions while a run is incomplete.
