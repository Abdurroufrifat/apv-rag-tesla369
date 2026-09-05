# Data Directory

- `raw/`: immutable source exports and manually saved metadata; never edit in place.
- `interim/`: normalized or deduplicated intermediate tables.
- `processed/`: versioned model-ready datasets and split manifests.
- `templates/`: schemas and starter rows for the pilot.

Do not commit copyrighted full-text documents, private information, API keys, or credentials. Prefer stable identifiers, URLs, access dates, lawful short spans, and scripts that reproduce permitted downloads.

Every generated dataset release should have a manifest containing file hashes, code version, collection dates, and license notes.

