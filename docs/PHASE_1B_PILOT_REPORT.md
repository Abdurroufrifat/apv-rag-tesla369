# Phase 1B pilot report: five Tesla attribution claims

**Protocol:** 0.1  
**Evidence access date:** 2026-09-05  
**Status:** machine-assisted evidence reconnaissance; not gold annotation

## Result in one sentence

The pilot found contemporaneous support for the material meaning of one quotation
(`T369-004`, in a 1927 Serbian newspaper interview) and insufficient evidence for
the other four; every result still requires independent human review.

## Scientific status

These are pre-annotations, not final dataset labels. An annotator must not convert
them directly into gold labels. The frozen Phase 1A search plan remains unchanged,
and every executed search is recorded in a separate log with its access limitation.

The label `insufficient` means that the recorded evidence cannot justify a stronger
verdict. It does **not** mean “Tesla never said it.” Likewise, the word
`authenticated` for `T369-004` applies to the materially equivalent proposition;
the familiar English sentence is a translation from Serbian.

## Pilot outcomes

| Claim | Provisional outcome | Evidence finding | Required human gate |
|---|---|---|---|
| `T369-001` — magnificence of 3, 6 and 9 | `insufficient` (0.86) | Tesla's 1919 autobiography documents repeated acts divisible by three, but not the viral cosmic-key quotation. The earliest directly inspected dated web occurrence in this pilot is 1 December 2009 and has no source. | Rerun direct Tier-A searches; inspect the 1919 scan; verify archived pre-2009 captures. |
| `T369-002` — energy, frequency, vibration | `insufficient` (0.80) | The located trail is Ralph Bergstresser's late recollection of an unrecorded private 1942 conversation, reproduced in product-related web material. | Obtain the complete 1996 Bergstresser booklet and search for a contemporaneous record. |
| `T369-003` — brain as receiver / universal core | `insufficient` (0.86) | Later pages repeat it, but the source commonly cited through Wikiquote does not contain the receiver continuation. | Audit the Wikiquote history and cited target; search pre-2012 print corpora independently. |
| `T369-004` — present is theirs / future is mine | `authenticated` (0.90), provisional | The National Library of Serbia exposes the original *Politika* issue of 27 April 1927. A later *Politika* archive article identifies pages 1–2 and reproduces the materially equivalent Serbian passage. | Two Serbian-English annotators must record the exact line, literal translation, and normalized English claim. |
| `T369-005` — others stole my idea | `insufficient` (0.82) | A 2009 school literary magazine contains a Croatian variant without a source; a 2010 institutional page and later English quote sites repeat it. | Search early biographies/correspondence and trace the South Slavic and English variants separately. |

Confidence is confidence in the provisional evidence judgment, not the probability
that Tesla did or did not speak the words.

## High-value provenance findings

### 1. Repetition must not be counted as corroboration

The 3-6-9 and Bergstresser pages form small families of dependent or plausibly
dependent sources. A naive RAG system can retrieve many pages and become more
confident because it mistakes copies for independent witnesses. APV-RAG must
aggregate at the provenance-family level before evidence scoring.

### 2. A citation can exist and still fail to support the claim

Two citation-target mismatches are encoded explicitly:

- Wikiquote links the combined receiver passage to a purple-plate page that contains
  the energy-frequency sentence but not the receiver continuation.
- A Tesla Universe quote page links the full present-versus-future quotation to a
  1905 article whose displayed text contains only the opening thought about past
  greatness. The supported path for the later sentences is the 1927 *Politika*
  interview.

This motivates a separate **citation-target entailment** component. Link presence is
not the same as textual support.

### 3. Source prestige is not primary-source independence

A 2022 scholarly editorial repeats the present-versus-future sentence as an
epigraph without giving its historical source. It is useful as an occurrence and a
hard negative, but it belongs downstream of the attribution rather than serving as
independent authentication.

### 4. Translation must be modeled explicitly

For `T369-004`, exact-string matching in English is the wrong criterion. The system
needs distinct fields for original language, literal translation, normalized claim,
and translation-review status. Otherwise a correct Serbian primary source can be
missed, or a later English rendering can be mistaken for Tesla's verbatim wording.

## What this adds to the machine-learning paper

The five-claim pilot supports four testable components rather than a generic chatbot:

1. **Provenance-family collapse** — compare raw document voting with one vote per
   independent family.
2. **Citation-target entailment** — predict whether the linked source actually
   supports the attributed wording.
3. **Quality- and time-aware evidence scoring** — favor inspectable contemporaneous
   records over later high-ranking pages.
4. **Calibrated abstention** — select `insufficient` when evidence cannot support an
   authenticated, misattributed, or contradicted verdict.

The central ablation should compare:

| System | Provenance collapse | Source quality | Citation-target check | Abstention |
|---|---:|---:|---:|---:|
| BM25 / dense retrieval + majority vote | No | No | No | No |
| Quality-weighted RAG | No | Yes | No | Optional |
| Provenance-aware RAG | Yes | Yes | No | Yes |
| Full APV-RAG | Yes | Yes | Yes | Yes |

Primary metrics should include macro-F1 over the four verdict labels, selective risk
versus coverage, expected calibration error, citation precision, evidence-family
recall, and accuracy on citation-mismatch hard negatives. Split the final dataset by
canonical claim and provenance family to prevent near-duplicate leakage.

## Files produced

- `data/pilot/tesla_phase1b_evidence_v0_1.csv` — 20 evidence records
- `data/pilot/tesla_phase1b_search_log_v0_1.csv` — 35 executed-search records
- `data/pilot/tesla_phase1b_verdicts_v0_1.csv` — five provisional verdicts
- `data/pilot/tesla_phase1b_provenance_edges_v0_1.csv` — 11 dependency edges
- `data/pilot/phase1b_manifest_v0_1.sha256` — integrity hashes for all four tables
- `schemas/` — record schemas
- `scripts/validate_phase1b.py` and `tests/test_phase1b.py` — automated checks

## Remaining limitations

- Most negative Tier-A searches were domain-targeted web searches, not exhaustive
  archive-interface sessions. The log says so on every affected row.
- The February 1919 magazine scan needs line-level comparison with the transcription.
- The Bergstresser booklet is known from metadata but has not been inspected in full.
- The 1927 Serbian line needs independent bilingual review.
- “Earliest located” is bounded by this search protocol and access date; it never
  means “first ever.”
- No inter-annotator agreement can be reported until two humans annotate independently.

## Phase 1B completion rule

Phase 1B is complete only after:

1. the validator and all tests pass;
2. Annotators A and B independently complete the human-review sheet;
3. disagreements are preserved and adjudicated;
4. Cohen's kappa or Krippendorff's alpha is reported with raw agreement;
5. the bilingual decision for `T369-004` is documented; and
6. the decision is made whether to revise protocol 0.1 before claims 6–20.
