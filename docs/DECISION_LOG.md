# Decision Log

Record decisions that affect scientific validity or reproducibility.

| Date | Decision | Evidence/reason | Alternatives considered | Consequence | Owner |
|---|---|---|---|---|---|
| YYYY-MM-DD | Example: keep `insufficient` separate from `contradicted` | Absence of located evidence is not direct refutation | Binary true/false | Requires selective evaluation | Researcher |
| 2026-09-05 | Keep all Phase 1B outputs as `machine_assisted_provisional` | Human annotators have not yet worked independently, so gold-label agreement cannot be measured | Promote the reconnaissance result directly to gold | Prevents circular evaluation and LLM-label leakage | Research lead |
| 2026-09-05 | Provisionally authenticate only the material proposition in `T369-004` | The National Library of Serbia exposes the 27 April 1927 *Politika* issue; a same-publisher retrospective reproduces the Serbian passage | Treat familiar English punctuation as verbatim Tesla wording | Requires bilingual line-level verification and separate literal/normalized translations | Research lead |
| 2026-09-05 | Add `citation_mismatch` as a provenance edge | Two high-value cases link to sources that do not contain the full attributed wording | Treat any citation link as evidence support | Enables citation-target entailment experiments and hard-negative evaluation | Research lead |
| 2026-09-05 | Preserve domain-targeted negative searches with explicit limitations | Web indexing is not equivalent to an exhaustive archive-interface search | Report broad searches as exhaustive | Human annotators must repeat Tier-A searches directly before adjudication | Research lead |
