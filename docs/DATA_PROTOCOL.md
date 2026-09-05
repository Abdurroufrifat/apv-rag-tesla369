# SciAttr-369 Data Protocol (Draft 0.1)

This protocol defines the pilot dataset. It must be revised after independent annotation of the first 20 claims and then frozen before large-scale collection.

## 1. Unit of analysis

One record represents one **canonical attribution claim**: a proposition that a named person authored, said, discovered, believed, or practiced something.

Paraphrases that preserve the same proposition share a `canonical_claim_id`. Claims that sound related but require different evidence must remain separate. For example:

- “Tesla said a quotation about the magnificence of 3, 6, and 9.”
- “Tesla described compulsive habits involving actions divisible by three.”

These are two different canonical claims.

## 2. Verdict definitions

### `authenticated`

Use only when the attribution is traceable to a primary source or a sufficiently authoritative, transparent record. The evidence must support the material wording or proposition—not merely repeat the attribution.

### `misattributed`

Use when reliable evidence identifies another origin, later author, or documented transmission path inconsistent with the claimed attribution.

### `contradicted`

Use when reliable evidence directly conflicts with the attributed proposition. Do not use this label merely because a search found nothing.

### `insufficient`

Use when the available evidence is incomplete, inaccessible, ambiguous, or incapable of supporting a definitive verdict. This is a valid target class, not a temporary error.

## 3. Source hierarchy

Source rank is a feature and audit aid, not an automatic verdict.

| Rank | Source type | Examples | Typical role |
|---:|---|---|---|
| 1 | Primary contemporaneous | authored work, letter, patent, interview transcript, archival document | Direct attribution evidence |
| 2 | Curated institutional | museum archive, national archive, critical edition | Authentication/context |
| 3 | Scholarly secondary | peer-reviewed history, academic book with traceable citations | Interpretation and provenance |
| 4 | Reputable tertiary | edited encyclopedia, established fact-check | Discovery and synthesis |
| 5 | Uncurated web | blog, quote site, social post, generated page | Claim occurrence; rarely authentication |

OCR output is not independently authoritative. When a verdict depends on a passage, compare it to the scanned page or a verified transcription.

## 4. Evidence records

Each evidence item should include:

- stable evidence ID;
- claim ID;
- title, author/creator, publisher/archive, and date;
- URL or archival identifier and access date;
- source type and rank;
- quoted span or exact page/section pointer;
- stance: `supports`, `refutes`, `mentions_only`, or `unrelated`;
- whether the source is primary;
- provenance-family ID;
- OCR status and transcription confidence;
- license/access notes; and
- annotator rationale.

Quote only the minimum passage required for research auditing and respect source licenses.

## 5. Provenance families

Assign the same `provenance_family_id` when sources appear to derive their claim content from a shared upstream source. Signals include:

- identical unusual wording or errors;
- explicit hyperlinks or citations;
- matching publication chronology;
- identical quotation truncation;
- near-duplicate passages; and
- shared archived snapshots or syndication metadata.

Do not infer copying solely from semantic similarity. Record `provenance_confidence` as `high`, `medium`, or `low`, plus a rationale.

## 6. Search and negative-evidence log

For every `insufficient` claim, preserve a reproducible search log:

- repositories/databases searched;
- exact search strings;
- spelling and language variants;
- date range;
- search date;
- inaccessible collections; and
- relevant results reviewed.

The correct wording is “not located in the documented search,” not “does not exist,” unless direct contrary evidence supports the stronger statement.

## 7. Claim collection

Collect claims from varied channels, then verify them against independent evidence. Candidate channels may include quotation sites, social platforms, videos, popular articles, biographies, archives, and scholarly discussions.

Sampling should stratify by:

- scientist and discipline;
- era;
- claim type (quotation, discovery, belief, behavior, prediction);
- popularity/repetition count;
- apparent evidence quality; and
- anticipated verdict.

Avoid building a dataset dominated by famous false quotes; it would encourage shortcut learning.

## 8. Annotation procedure

1. Normalize the candidate into one atomic canonical claim.
2. Search for the earliest traceable occurrence and primary evidence.
3. Add evidence items without viewing a model verdict.
4. Label evidence stance independently.
5. Assign source metadata and provenance family.
6. Judge sufficiency.
7. Assign the claim verdict and confidence.
8. Have a second annotator repeat Steps 4–7 independently.
9. Adjudicate disagreements and record the decision.

Annotators may use retrieval tools to find sources. They must not accept a chatbot response, search snippet, or unattributed quotation page as ground truth.

## 9. Required pilot fields

The starter CSV contains claim-level fields only. Evidence will later move to a separate one-to-many table. Required claim fields are:

- `claim_id`
- `canonical_claim_id`
- `claim_text`
- `claimed_person`
- `claim_type`
- `language`
- `verdict`
- `verdict_confidence`
- `sufficiency`
- `source_urls`
- `provenance_family_ids`
- `review_status`
- `annotator_id`
- `notes`

Multiple values in the starter CSV use a semicolon delimiter. The production dataset should use JSON Lines or relational tables to avoid ambiguous cells.

## 10. Split policy

- Group all variants of a canonical claim in one split.
- Prevent provenance-family leakage across splits when copied wording could reveal the answer.
- Reserve scientists absent from training for a cross-person generalization slice.
- Reserve time-based slices for emerging web repetitions.
- Freeze and hash each corpus snapshot.

## 11. Synthetic transformations

Generate transformations only after the human-grounded base claim is frozen:

- paraphrase without label change;
- add 1, 5, 10, or 25 derivative copies;
- inject a fabricated citation string;
- rewrite in more authoritative language;
- remove the highest-ranked source;
- degrade archival evidence with controlled OCR noise.

All variants inherit the canonical claim’s split. Synthetic text must be flagged and never represented as real historical evidence.

## 12. Quality and reporting

Report:

- class distribution and missingness;
- source-type and person distribution;
- duplicate and near-duplicate analysis;
- inter-annotator agreement with confidence intervals where feasible;
- adjudication rate;
- unavailable-source rate;
- provenance-confidence distribution; and
- all protocol changes with dates.

## 13. Ethics and legal care

- Preserve uncertainty and avoid reputational claims beyond the evidence.
- Store only material needed for research and respect copyright/license limits.
- Release links, metadata, and short evidence spans where full text cannot be redistributed.
- Identify synthetic claims clearly.
- Document model and search-engine versions because retrieval results change over time.

