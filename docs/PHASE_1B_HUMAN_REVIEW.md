# Phase 1B independent human-review sheet

> **Archived:** No human-review stage will be conducted. Existing benchmark labels
> replace project-specific annotation for all quantitative experiments.

Do not inspect `data/pilot/tesla_phase1b_verdicts_v0_1.csv` while assigning an
independent label. That file is machine-assisted pre-annotation and may bias you.

## Roles

- **Annotator A:** works independently.
- **Annotator B:** works independently and cannot see A's labels.
- **Adjudicator:** sees both only after both are frozen.
- **Bilingual reviewer:** required for Serbian-English evidence in `T369-004`.

One person may be both an annotator and bilingual reviewer, but may not adjudicate
their own unresolved disagreement alone.

## Before labeling each claim

- [ ] Read the canonical claim, not a shortened social-media variant.
- [ ] Run the frozen exact query and fragment query.
- [ ] Search at least two Tier-A repositories directly.
- [ ] Search at least one scholarly index.
- [ ] Inspect the earliest located occurrence and its upstream references.
- [ ] Record inaccessible sources as inaccessible, not searched.
- [ ] Inspect the underlying document, not only a search snippet.
- [ ] Separate occurrence from authentication.
- [ ] Record page, column, section, or archival identifier.
- [ ] Group dependent copies into provenance families.
- [ ] Use `insufficient`, not `misattributed`, when only evidence absence is known.

## Claim-specific review

### T369-001

- [ ] Compare the *My Inventions* transcription with the February 1919 scan.
- [ ] Confirm that divisible-by-three behavior is not treated as proof of a 3-6-9 code.
- [ ] Repeat Chronicling America and Internet Archive searches in their own interfaces.
- [ ] Verify the date/capture of the December 2009 Nerd Business page.

### T369-002

- [ ] Obtain the complete 1996 Bergstresser booklet.
- [ ] Record whether the exact sentence appears and on which page.
- [ ] Search for any 1942 transcript, diary, letter, or independent witness.
- [ ] Treat product-related mirrors as one family unless independence is established.

### T369-003

- [ ] Compare the receiver wording with the cited Biblioteca Pleyades target.
- [ ] Audit when the receiver continuation entered Wikiquote.
- [ ] Verify the 18 February 2012 post and search earlier print occurrences.
- [ ] Do not assume that its proximity to T369-002 gives both sentences one origin.

### T369-004

- [ ] Open the National Library of Serbia issue dated 27 April 1927.
- [ ] Locate “Посета г. Николи Тесли” on pages 1–2.
- [ ] Transcribe the decisive Serbian sentence exactly.
- [ ] Record page and column.
- [ ] Produce a literal translation and a normalized English rendering separately.
- [ ] Have a second bilingual reviewer confirm both.
- [ ] Document why the Tesla Universe 1905 source link is not used for the later sentences.

### T369-005

- [ ] Inspect PDF page 126 of *POZICA 2009*.
- [ ] Search the Croatian/Serbian wording and English wording separately.
- [ ] Search early biographies and correspondence catalogs.
- [ ] Verify whether the 2010 IRB page names an upstream source elsewhere on the site.

## Independent annotation form

Copy this block once per claim and once per annotator:

```text
Claim ID:
Annotator ID:
Date:
Verdict: authenticated / misattributed / contradicted / insufficient
Sufficiency: sufficient / insufficient
Confidence (0-1):
Decisive evidence IDs:
Evidence stance(s):
Source rank(s):
Provenance family ID(s):
Exact page/column/section:
Rationale:
Missing evidence:
Search limitations:
```

## Freeze and adjudication

1. Save A and B in separate files before comparison.
2. Compute raw agreement and chance-corrected agreement.
3. Flag disagreements in verdict, evidence stance, source rank, and family assignment.
4. Adjudicate with a written reason; never delete the original labels.
5. Only the adjudicated table may become a gold-label dataset.
