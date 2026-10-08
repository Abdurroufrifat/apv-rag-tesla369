# Current project status

The saved English explanation diagnostic is received and verified: 840 explanations, 2520 scored passage pairs and zero token-budget skips. Windows and local verification content agrees after line-ending normalization. The bounded analysis is in `artifacts/explanation_diagnostic_analysis_v1/RESULTS.md`; no rerun is needed.

The next saved-data step is complete: 6,600 multilingual verdicts were aligned and a Qwen/NLI agreement rule replayed across eleven files. It rejects 1,169 otherwise correct Qwen answers; results are in `artifacts/multilingual_agreement_replay_v1/RESULTS.md`. It is an exploratory supplied-evidence diagnostic, not a full multilingual retrieval or gate result.

The saved multilingual NLI probability audit is now in `artifacts/xfever_reliability_audit_v1/RESULTS.md`. It reports fixed-bin reliability on the same eleven supplied-evidence files; no new model run or fitted calibrator is claimed.

The optional source-snapshot control is now connected to all four policies in a separate wrapper. Its cached replay and per-policy results are in `artifacts/bound_policy_integration_v1/RESULTS.md`. This update does not require a new model run.

A fresh local reproducibility audit now passes the full test suite and eight other commands. See `artifacts/reproducibility_audit_2026_10_04/RESULTS.md` for 30 still-missing files from older exports; keep those originals in your Windows project.

The next external evaluation input is prepared: 300 FEVER claims selected without using labels and checked against prior XFEVER IDs. The Wikipedia archive is not included, and no FEVER model run has been done. See `docs/FEVER_EXTERNAL_EVALUATION.md`.

Extract this update directly into your existing `D:\apv-rag-tesla369` folder, replacing matching files and preserving `.venv`, models and caches. No command is needed for this update.

Original-cohort mean maximum passage entailment scores are 0.2701 for SciFact and 0.3306 for retrieved climate; these scores do not measure factual explanation accuracy. Source authentication, explanation truth, multilingual gate/retrieval integration and confidence calibration remain open. No human annotation is requested. Manuscript writing awaits your permission and GitHub publication remains paused.
