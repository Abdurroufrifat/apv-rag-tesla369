# Phase 1A Archival Search Protocol

**Version:** 0.1  
**Freeze status:** pilot only  
**Pilot scope:** 20 Tesla attribution claims

## 1. Objective

For each claim, identify:

1. the earliest located occurrence of the wording or proposition;
2. any primary or contemporaneous evidence;
3. later sources that repeat, paraphrase, cite, or distort the claim;
4. whether sources are independent or members of the same provenance family; and
5. whether the collected evidence is sufficient for a verdict.

The procedure verifies historical attribution. It does not evaluate spiritual or numerological interpretations as scientific hypotheses.

## 2. Fixed source order

Search each claim in this order and log every stage, including unsuccessful stages.

### Tier A: primary and contemporaneous records

- Tesla-authored articles, lectures, books, patents, and correspondence
- scanned newspapers and magazines published during Tesla’s lifetime
- court documents and patent-office records
- archival finding aids and dated manuscript collections

Preferred repositories include the Library of Congress, Internet Archive scans, Google Patents with linked patent-office records, Project Gutenberg transcriptions checked against scans, and the Nikola Tesla Museum catalog.

### Tier B: scholarly historical analysis

- peer-reviewed history of science and engineering
- scholarly books from academic presses
- critical editions with page-level citations

### Tier C: curated institutional summaries

- museums, professional engineering societies, national archives, and established reference works

### Tier D: occurrence and propagation sources

- news features, quote sites, blogs, videos, social posts, and image macros

Tier-D material can establish that a claim circulated. It cannot, by repetition alone, authenticate Tesla as its originator.

## 3. Query sequence

For every candidate claim:

1. Search the exact wording in quotation marks.
2. Search distinctive 5–10-word fragments.
3. Search punctuation, spelling, and translation variants.
4. Search the proposition without quotation marks.
5. Restrict searches by year ranges ending in 1943.
6. Search Tesla-authored collections and scans directly.
7. Search the earliest located page’s citations, hyperlinks, and archived predecessors.
8. Repeat backward until no earlier traceable source is located.

Use `data/pilot/tesla_search_plan_v0_1.csv` as the frozen pilot query plan. Record executed searches separately; never overwrite the plan.

## 4. Minimum search requirement

Before assigning `misattributed` or `insufficient`, an annotator must search:

- at least two Tier-A repositories;
- at least one scholarly index or scholarly historical source;
- exact wording and fragment variants; and
- the earliest visible web occurrence’s upstream references.

An inaccessible archive must be marked `inaccessible`; it must not be counted as searched evidence.

## 5. Negative-search statement

Use this form:

> “No authenticating primary source was located under protocol version X, using the recorded repositories, query variants, and access date.”

Do not write “Tesla never said this” unless direct historical evidence justifies that stronger conclusion.

## 6. Evidence capture

For each evidence item, record:

- stable ID and claim ID;
- title, creator, publisher/archive, publication date, URL/identifier, and access date;
- exact page, column, section, patent number, or archival folder when available;
- a short evidence summary rather than unnecessary copied text;
- stance toward the claim;
- source rank and primary-source status;
- OCR/transcription status;
- provenance-family ID and parent-source ID;
- relation to the parent (`quotes`, `cites`, `paraphrases`, `mirrors`, `syndicates`, `unknown`);
- verification status and annotator.

## 7. Provenance-family assignment

Two documents may share a family when at least two of the following support dependence:

- one explicitly links or cites the other;
- publication chronology permits the proposed direction;
- unusual wording, truncation, spelling error, or punctuation is shared;
- near-duplicate analysis shows substantial passage overlap;
- archive or syndication metadata identifies a common source.

Semantic similarity alone is insufficient. Record `high`, `medium`, or `low` provenance confidence and explain borderline cases.

## 8. Human annotation

- Annotator A and Annotator B work independently.
- Neither annotator sees an LLM verdict while assigning the gold label.
- A third reviewer adjudicates disagreements where possible.
- Report agreement for verdict, stance, source rank, and family assignment.
- Preserve the pre-adjudication labels.

## 9. Pilot workflow

1. Validate the 20 claim candidates.
2. Execute the frozen search plan for claims `T369-001` through `T369-005`.
3. Review and revise the protocol before searching the remaining claims.
4. Complete independent annotations for all 20.
5. Calculate agreement and adjudication rate.
6. Audit provenance families and claim leakage.
7. Decide whether to expand to 100 Tesla claims or immediately add other scientists.

## 10. Initial authoritative entry points

- Library of Congress Tesla guide: https://guides.loc.gov/chronicling-america-nikola-tesla
- Library of Congress Tesla correspondence record: https://www.loc.gov/item/mm82050302/
- *My Inventions* transcription: https://en.wikisource.org/wiki/My_Inventions
- Tesla remote-control patent US613809A: https://patents.google.com/patent/US613809A/en
- Tesla turbine patent US1061206A: https://patents.google.com/patent/US1061206A/en
- Nikola Tesla Museum, life and work: https://tesla-museum.org/en/nikola-tesla-2/life-and-work/
- Project Gutenberg, Martin’s 1894 collection: https://www.gutenberg.org/ebooks/39272

These links are starting points, not automatic proof. Transcriptions must be checked against scans when exact wording determines the verdict.

