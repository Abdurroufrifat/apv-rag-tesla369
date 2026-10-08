# Clean multilingual confirmation reproduction

Extract this update directly into `D:\apv-rag-tesla369`. Retain the successful English clean run directory, including its `environment` subfolder. The command finds the latest completed English reproduction with intact receipts. It reuses that isolated environment, checks its package inventory and runs `pip check`; no package installation or model download occurs.

Run in VS Code PowerShell:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_clean_multilingual_confirmation.py
```

Upload `D:\apv-rag-tesla369\clean_multilingual_confirmation_outputs.zip` when it finishes. Setup/inference errors after run creation are also exported with logs. An error before run creation prints without producing a new ZIP. The original English environment, inference caches and result archives are retained.

The original multilingual producer is invoked with a new empty output directory beneath `artifacts/clean_multilingual_confirmation_runs/run_<timestamp>/inference`. Fresh tokenization, English NLI/embedding features, multilingual NLI features and generation run for all 600 original queries. These are 100 underlying claim IDs represented in six parallel variants, not 600 independent claims. The existing excerpt pools, four frozen policies and development-only calibrators remain unchanged. No fitting or new model selection is performed.

Only the original producer's final export callback is temporarily replaced: its shared project-root ZIP destination is suppressed. Its inference main, source files, settings, inputs and model checks remain unchanged. The wrapper restores the callback and CLI arguments even if inference fails. Results are packaged separately by this wrapper. Before starting, the wrapper refuses existing inference files; the original generator receives an empty seed mapping and no prior answers are imported.

Both original and fresh results must pass the existing multilingual feature/prompt/decision/calibration replay. Comparison includes prepared contexts, both feature caches, response bindings, policy predictions, direct NLI predictions, frozen correctness confidence and summary metrics. Float values use the same predeclared absolute tolerance of `1e-5` as the English reproduction. IDs, labels, text, ordering, prompts and other discrete values must match exactly. Differences are saved and cause failure; original outputs and the tolerance are not adjusted afterward.

The fixed four model assets must be present and match the frozen reference. CPU inference may take hours; leave the terminal open and do not start overlapping runs. Each invocation uses a new output directory. It does not resume an interrupted multilingual run or reuse its inference caches.

This step checks multilingual confirmation repeatability in the already-tested Windows environment. It does not perform a second independent installation, reproduce development training, create human translations, test full-page/open-web retrieval, authenticate historical quotations or validate explanation truth. The original charter remains open. No manuscript or GitHub action is included.

Local development verification covers wrapper behavior and saved-output replay. Actual neural execution requires the Windows assets and remains pending until its result export is received.
