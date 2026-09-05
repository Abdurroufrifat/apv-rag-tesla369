# APV-RAG: Tesla 3-6-9 Scientific-Attribution Verification

This repository is the starter workspace for the proposed paper:

> **Repetition Is Not Evidence: Archive- and Provenance-Aware RAG for Verifying the Tesla 3-6-9 Claim and Other Viral Scientific Attributions**

The project treats the popular “Tesla 3-6-9 code” as a **misattribution-verification case study**, not as an established scientific law. The central machine-learning question is whether a retrieval-augmented system can distinguish independent archival evidence from many websites that repeat the same unsupported attribution.

## Proposed contribution

The planned system, **APV-RAG** (Archival Provenance Verification RAG), combines:

1. hybrid retrieval from primary archives, scholarly sources, and the open web;
2. source-type and source-quality estimation;
3. a temporal provenance graph that collapses copied or derivative sources;
4. claim–evidence stance and evidence-sufficiency prediction;
5. calibrated abstention when the available evidence cannot justify a verdict; and
6. citation-constrained explanations.

The proposed dataset is **SciAttr-369**, a benchmark for scientific quotations and attributions, with Tesla 3-6-9 claims as a focused case study and claims about other historical scientists included to support generalization.

## Important scientific rule

Do not write that Tesla discovered a mystical “369 code.” The defensible starting position is:

- the widely shared “magnificence of 3, 6 and 9” quotation is not currently authenticated by a traceable primary source;
- Tesla’s autobiographical writing does report repetitive habits involving actions divisible by three; and
- those are different claims and must have separate evidence records.

Every conclusion in the paper must follow from documented evidence and uncertainty labels.

## Start here

Read [`START_HERE.md`](START_HERE.md), then run the environment check and tests. Do not download large datasets or train a model until Phase 1 is complete.

## Repository map

| Path | Purpose |
|---|---|
| `START_HERE.md` | Exact setup instructions for Windows and Colab |
| `docs/PROJECT_CHARTER.md` | Research questions, novelty, experiments, and publication plan |
| `docs/DATA_PROTOCOL.md` | Dataset construction and annotation rules |
| `docs/PHASE_1_CHECKLIST.md` | First milestone and completion criteria |
| `docs/literature_matrix.csv` | Structured literature-review template |
| `data/templates/claims_template.csv` | Claim-record template |
| `schemas/claim_record.schema.json` | Machine-readable record schema |
| `src/apv_rag/` | Lightweight validation package |
| `scripts/check_environment.py` | Reports local hardware and Python setup |
| `scripts/validate_claims.py` | Validates a claim CSV before experiments |
| `notebooks/00_colab_setup.ipynb` | Cloud/GPU environment check |
| `config/project.yaml` | Reproducible project settings |

## Planned compute split

- **Local Windows laptop (16 GB RAM):** writing, source collection, annotation, preprocessing, unit tests, classical baselines, evaluation, plots, and paper assembly.
- **Google Colab/cloud GPU:** sentence embeddings at scale, transformer fine-tuning/inference, and optional local-LLM experiments.

This split keeps the project reproducible and avoids spending cloud compute before the data protocol is stable.

