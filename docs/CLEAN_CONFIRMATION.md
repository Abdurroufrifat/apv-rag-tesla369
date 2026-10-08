# Clean English confirmation reproduction

Extract the update directly into `D:\apv-rag-tesla369`. Keep the existing `.venv`, models, datasets and artifacts.

In VS Code PowerShell, run:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_clean_confirmation.py
```

Upload `D:\apv-rag-tesla369\clean_confirmation_outputs.zip` when the command finishes. A setup or inference failure after the new run directory is created also exports its logs to that ZIP. A prerequisite failure before run creation prints its error without exporting a ZIP. Send the ZIP instead of a screenshot. The command prints its unique run directory.

The command validates the original FEVER cohort/reference and saved export, checks existing model files, and requires the original five neural package versions. It snapshots every installed distribution in this project's existing `.venv`, excluding the editable `apv-rag` distribution. It downloads binary wheels at those exact versions, records their SHA-256 hashes, creates an isolated environment beneath a new `artifacts/clean_confirmation_runs/run_<timestamp>` directory, and installs from those local wheels with online dependency resolution disabled during installation. `pip check` and an exact installed-package inventory comparison must pass. The project imports from its existing `src` and `scripts` directories; its original producers are unchanged.

The existing pinned Wikipedia index and three local models are reused as frozen input assets. Model weights are not downloaded. Wheel downloads need Internet access and additional disk space. If a recorded version has no compatible wheel, setup fails and preserves the error. It does not substitute another version or modify reference receipts. The wheel directory and environment remain on your computer and are excluded from the result ZIP; wheel hashes and installation logs are included. Retain these wheels for offline installation later. This is a snapshot of the current Windows package environment, not a pre-existing universal dependency lock.

Fresh retrieval, tokenization, NLI and embedding features, verdict generation, explanations and four frozen gate policies run for the original 300 English confirmation claims. The stage starts with no prior feature or answer cache. Its original generator receives `seeds={}`. No development training, calibration fitting, confirmation tuning or new model selection occurs. The original corpus/index, inputs, model files, responses and result exports are preserved. The saved stage is checked with the existing feature/prompt/decision verifier.

Comparison covers retrieved contexts, prepared contexts, features, prompt-bound responses and all 1,200 policy rows. Labels, IDs, ordering, text, evidence and prompt bindings must match exactly. Floating values have an absolute tolerance of `1e-5`, fixed before this run; invalid/nonfinite values fail. All differences are saved. They cannot be repaired by changing tolerance, original outputs or reference hashes after observing the comparison.

The neural work can take hours on a CPU. Leave the terminal open. Every invocation starts a new run and does not import another run's caches. An interrupted run is retained but is not a successful reproduction. Do not repeatedly invoke the command while another run is active.

`--preflight` checks original inputs and local assets without creating an environment or running neural inference:

```powershell
.\.venv\Scripts\python.exe scripts\run_clean_confirmation.py --preflight
```

Even a successful comparison closes only the scoped English confirmation inference check on that Windows machine. It does not reproduce development training, all earlier experiments, multilingual inference, index construction from the upstream archive or a different operating system. It cannot establish historical attribution, explanation truth, gate superiority or completion of the original charter. No manuscript or GitHub operation is included.

Development verification here uses unit tests and existing frozen-export checks. This environment lacks the neural weights and Wikipedia index; actual clean installation and neural reproduction remain pending until the Windows result is received.
