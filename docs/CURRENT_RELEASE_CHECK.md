# One-command current export check

Extract this update into the existing `D:\apv-rag-tesla369` folder, then run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\check_current_release.py
```

This command uses the same interpreter for all eight checks. It adds the existing
`src` and `scripts` folders to the child-process import path, runs pytest, verifies
the fixed multilingual confirmation, replays the source-origin and evidence-display
audits, checks the saved guard trade-offs, verifies the latest clean-confirmation,
capture-timing and pool-mismatch receipts, replays the component ablation, then
regenerates consolidated status.
It performs no neural inference or new benchmark tuning.

Each run creates a separate timestamped folder under `artifacts/current_release_checks`.
Logs, environment versions, command results and current open requirements are saved
in `current_release_checks_outputs.zip` at the project root. The ZIP contains only
this run. An unsuccessful check is recorded, the other checks continue, and the
command exits with failure if any check failed. `--output` accepts a new empty
directory for a named local run and refuses to overwrite existing logs.

A successful run means the current export checks passed. It does not authenticate
Tesla quotations, verify semantic explanation truth, show broad robustness benefit,
complete all planned ablations, or reproduce neural inference from a clean
installation. The existing project charter and user restrictions are preserved.
No manuscript or GitHub publication is performed.

The included `latest_20261005` run passed all eight checks and 350 tests.
The regenerated status now records the received scoped English/multilingual
clean-confirmation reproduction and the negative timing result. Full training
and source-index reproduction remain outside those received confirmation runs.
