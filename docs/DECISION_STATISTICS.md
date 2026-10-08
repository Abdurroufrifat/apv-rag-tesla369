# Statistical check of the cached decision comparison

The two comparisons are selected-rule versus original probability argmax,
separately for BM25+NLI and dense+NLI. The statistic is the mean Macro-F1
difference across five fixed seeds. Claims are not treated as independent:
609 validation claims are clustered into 375 frozen connected provenance/claim
groups. Bootstrap samples draw whole groups with replacement, keeping methods
and seeds paired. Randomization swaps the two rules within whole groups and
uses the same swap for all five seeds.

| Method | Mean Macro-F1 gain | Group-bootstrap 95% interval | Holm-adjusted two-sided p |
|---|---:|---|---:|
| BM25+NLI | +0.0773 | [0.0342, 0.1147] | 0.0079 |
| Dense+NLI | +0.0827 | [0.0477, 0.1116] | 0.0036 |

The analysis used 2,000 bootstrap draws, 10,000 randomization draws, and seed
369. Randomization p-values use the plus-one correction. Holm correction
covers these two decision-rule comparisons only; it does not cover all prior
experiments, method choices, or exploratory analyses.

Both intervals exclude zero and both corrected p-values are below 0.05.
This supports a decision-rule improvement on this development sample. It
does not establish a final-test improvement: validation outcomes had already
been inspected when this experiment was proposed. The evidence corpus also
consists of benchmark-selected answer excerpts, not retrieved original pages.
No superiority over external published systems is established.

The official development set remained unused. No NLI inference was repeated.
The reproducible analysis and full numerical results are included in
`scripts/analyze_decision_statistics.py` and
`artifacts/cached_decision_statistics/decision_statistics.json`.

No Windows experiment rerun is required. Inspect the included JSON or run the
project test suite. The analysis runner refuses to overwrite its completed
artifact directory.
