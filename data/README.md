# Data Directory

- `raw/`: immutable source exports and manually saved metadata; never edit in place.
- `interim/`: normalized or deduplicated intermediate tables.
- `processed/`: versioned model-ready datasets and split manifests.
- `templates/`: schemas and starter rows for the pilot.
- `pilot/tesla_phase1b_evidence_v0_1.csv`: inspected evidence and source metadata.
- `pilot/tesla_phase1b_search_log_v0_1.csv`: executed queries with honest access limitations.
- `pilot/tesla_phase1b_verdicts_v0_1.csv`: machine-assisted provisional outcomes only.
- `pilot/tesla_phase1b_provenance_edges_v0_1.csv`: dependent-source and citation-mismatch edges.
- `pilot/phase1b_manifest_v0_1.sha256`: frozen SHA-256 hashes for the four pilot tables.
- `pilot/tesla_phase1c_source_register_v0_1.csv`: metadata and SHA-256 fingerprint
  for the private `T369-004` evidence capture; it does not contain the scan bytes.
- `templates/t369004_bilingual_review_template_v0_1.csv`: archived Phase 1C form;
  retained only for provenance and no longer used by the project.

Do not commit copyrighted full-text documents, private information, API keys, or credentials. Prefer stable identifiers, URLs, access dates, lawful short spans, and scripts that reproduce permitted downloads.

Every generated dataset release should have a manifest containing file hashes, code version, collection dates, and license notes.

Never use the Phase 1B provisional verdict file as gold training data. The project
performs no new human annotation; only unchanged labels from published benchmarks
may be used as quantitative ground truth. Tesla records remain unlabeled.

Files below `raw/` and generated reviewer packets below `artifacts/` are private
working material excluded from Git. Do not force-add them.
