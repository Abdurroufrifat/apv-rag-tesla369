# SciAttr-369 Annotation Guide v0.1

## Two different questions

Annotators must separate:

1. **Occurrence:** does this webpage/book/newspaper contain the wording?
2. **Authentication:** does reliable evidence establish that Tesla originated or expressed it?

A page can support occurrence while providing no authentication.

## Claim verdict

| Label | Use when | Do not use when |
|---|---|---|
| `authenticated` | Traceable evidence supports Tesla’s authorship or the material proposition | Many dependent websites merely repeat it |
| `misattributed` | Evidence identifies another origin or a later transmission inconsistent with Tesla authorship | The search simply found nothing |
| `contradicted` | Reliable evidence directly conflicts with the proposition | Only part of the wording differs |
| `insufficient` | Evidence is absent, inaccessible, ambiguous, or unable to justify a definitive label | An annotator wants to avoid a difficult decision despite sufficient evidence |

## Evidence stance

| Stance | Definition |
|---|---|
| `supports` | The evidence directly supports the canonical claim |
| `refutes` | The evidence directly conflicts with the canonical claim |
| `mentions_only` | It repeats, discusses, or contextualizes the claim without independently proving it |
| `unrelated` | Retrieved material does not materially address the claim |

## Source rank

| Rank | Definition |
|---:|---|
| 1 | Primary/contemporaneous document with stable identity and inspectable content |
| 2 | Curated archive, critical transcription checked against a scan, or strong scholarly historical source |
| 3 | Scholarly secondary source with traceable citations |
| 4 | Edited tertiary or reputable journalistic source useful for context |
| 5 | Uncurated web/social source useful mainly for propagation analysis |

Source rank is not a verdict. A Rank-1 document may mention a claim without supporting it.

## Sufficiency decision

Evidence is `sufficient` only if an annotator can explain why the selected material justifies the verdict and identify the decisive passage or record. Otherwise use `insufficient` and specify the missing evidence.

## Provenance relations

- `original`: earliest located member; not necessarily the true historical origin
- `quotes`: reproduces wording explicitly
- `cites`: names or links a parent source
- `paraphrases`: preserves the proposition with changed wording
- `mirrors`: republishes substantially the same document
- `syndicates`: distribution through a known publisher/feed relationship
- `unknown`: dependence is plausible but the direction is unresolved

## Confidence

- `high`: multiple independent metadata/content signals agree
- `medium`: evidence favors one judgment but an important uncertainty remains
- `low`: tentative pilot decision requiring adjudication

Numerical confidence is recorded from 0 to 1 only for analysis. It does not replace the written rationale.

## Special rules for Tesla 3-6-9 material

- The viral “magnificence of 3, 6 and 9” quotation and Tesla’s documented divisible-by-three habits are different claims.
- A description of repeated acts divisible by three does not prove belief in a cosmic 3-6-9 code.
- Later numerology explanations are propagation evidence, not primary historical evidence.
- Do not infer `contradicted` from archive silence.

## Annotation quality check

Before finalizing a record, answer:

1. Is the claim atomic?
2. Did I inspect the underlying document rather than a search snippet?
3. Did I separate occurrence from authentication?
4. Is the decisive page/section recorded?
5. Are derivative sources grouped rather than counted as independent?
6. Would another annotator be able to reproduce my search?
7. Does the rationale support the selected verdict and sufficiency label?

