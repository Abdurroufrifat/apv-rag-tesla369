# Phase 1 Checklist: Protocol and 20-Claim Pilot

Do not begin mass scraping or model fine-tuning before this checklist is complete.

## 1. Environment

- [ ] Local environment report saved
- [ ] Unit tests pass
- [ ] Colab notebook runs with a GPU, or CPU fallback is documented
- [ ] Project folder is placed under version control

## 2. Search protocol

- [ ] Define source hierarchy and exclusion criteria
- [ ] Define database/search-engine queries before collecting results
- [ ] Record exact query, engine/database, date, result URL, and access status
- [ ] Define how archived pages and dead links are handled
- [ ] Define how OCR text is checked against page images

## 3. Tesla pilot

- [ ] Create 20 distinct canonical claims, not 20 paraphrases
- [ ] Include the viral 3-6-9 quotation as an `insufficient` candidate until evidence review is complete
- [ ] Keep the documented “multiples of three” habit as a separate claim
- [ ] Attach at least one evidence item or explicit no-evidence search log to every claim
- [ ] Assign source type, publication date, and provenance family
- [ ] Record the annotator’s rationale without using an LLM-generated verdict as ground truth

## 4. Annotation quality

- [ ] A second human independently labels the 20-claim pilot
- [ ] Disagreements are adjudicated and logged
- [ ] Cohen’s kappa or Krippendorff’s alpha is calculated for categorical labels
- [ ] Ambiguous cases lead to protocol revisions, not forced labels

## 5. Literature positioning

- [ ] Complete at least 25 high-relevance papers in `literature_matrix.csv`
- [ ] Cover RAG factuality, claim verification, source provenance, citation verification, abstention/calibration, and historical archives
- [ ] Write a one-page gap statement grounded in the literature matrix
- [ ] Confirm that the claimed novelty is not already published

## Phase-1 deliverables

1. versioned search protocol;
2. 20-claim pilot dataset;
3. annotation guide revision 1;
4. inter-annotator agreement report;
5. literature matrix and gap statement; and
6. go/no-go decision for full data collection.

