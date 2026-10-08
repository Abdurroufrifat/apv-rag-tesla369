# Explanation preflight receipt restore

The received Windows export has 322 passing tests and five successful checks. Project status alone fails with `Explanation preflight hash mismatch`. ZIP integrity, output receipts and current producer code hashes pass.

This update restores only the older derived `preflight.json` and its matching receipt from the checked project snapshot. The restored file replays exactly against the existing source outputs, task identities and literal-quote counts: 840 explanations and 2,520 passage pairs. No cached model output, benchmark label, gate decision or threshold is changed. No checksum is refreshed over an unvalidated file.

Extract into `D:\apv-rag-tesla369`, replacing the two files under `artifacts/explanation_diagnostic_preflight_v1`. Then rerun `.\.venv\Scripts\python.exe scripts\check_current_release.py` and upload the new `current_release_checks_outputs.zip`.

The export does not include the differing Windows preflight bytes, so their reason for divergence is unknown. The received failure is retained. Windows repair is pending rerun; historical authentication and semantic explanation truth remain unresolved. No neural inference, manuscript or GitHub push is involved.
