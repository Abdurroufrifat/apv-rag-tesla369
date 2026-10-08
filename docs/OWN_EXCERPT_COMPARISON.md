# Own-excerpt input substitution

Extract the updated consolidated ZIP into `D:\apv-rag-tesla369`, preserving
models, `.venv` and previous inference caches. Run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_own_excerpt_comparison.py --threads 4 --batch-size 8
```

This uses the existing NLI model offline. No embedding inference is required.
Original cached NLI pairs are copied into a separate diagnostic cache if
available; missing pairs are computed. The original experiment is preserved.
Interrupted inference can resume. Completed results cannot be overwritten.

Only the original train.json and its internal train/validation indices are
read. Official dev is never opened. Each validation claim receives its first
five distinct (answer text, source URL) excerpts in original source order.
Empty answers are skipped. Shared text with different URLs remains distinct,
matching the original corpus deduplication rule.

Each original calibrated classifier is refitted on its original cached training
features with the same parameters, folds and seeds. Its validation probabilities
must reproduce the saved original probabilities before comparison proceeds.
The classifier and selected decision exponent then stay fixed while the
validation features are replaced with own-excerpt NLI features.

The summary compares Macro-F1, per-class measures, raw Brier and argmax ECE,
and corrected/new errors across both retrieval methods and all five seeds.
Both ordinary argmax and the previously selected rule are reported. No rule
is tuned from these results.

This is a descriptive input-substitution experiment. Benchmark-attached
excerpts are not guaranteed sufficient, and the first five may omit useful
evidence. Feature distributions change while aggregators remain fixed.
Therefore a score difference does not isolate retrieval as the sole cause.
It is neither an upper bound nor a new confirmation result.

After completion, upload these files from
`D:\apv-rag-tesla369\artifacts\own_excerpt_comparison`:

- `own_excerpt_summary.json`
- `predictions.json`
- `output_manifest.json`

Neural inference remains pending the Windows run. The authoring environment
tests cached feature comparisons and original-probability reproduction.
