# APV-RAG with a Tesla archival stress test

This project studies claim verification using existing benchmark labels, evidence retrieval, source-family handling, learned evidence-coverage models and selective generation. The Tesla material is an unlabeled archival stress test. Its machine outputs do not authenticate historical attributions.

The full project remains unfinished against [the project charter](docs/PROJECT_CHARTER.md). No new human annotation is required. Manuscript writing and GitHub publication are paused.

## Current checked work

- AVeriTeC retrieval/verifier experiments, internal partitions and an official-development evaluation with overlap exclusions. Older received folders contain selected exports; several original caches and input identities are absent from this package.
- SciFact and retrieved/supplied CLIMATE-FEVER generation. Original document retrieval, sentence selection, labels, prompts, numeric guards and metrics have non-neural replay checks. Model inference is recorded from the Windows runs.
- Constructed complete/missing-rationale coverage targets, parent-group partitions and learned NLI, embedding and combined heads. These targets do not establish universal semantic sufficiency.
- A live controller export with 3600 policy records for 900 claims and four gate policies. A subsequent fresh-retrieval and model-feature export checks 600 claims and 2400 policies, with exact feature/probability agreement and 652 early rejections. All 1200 distinct generator responses in that integration export reuse identical prior prompts. This remains partial integration against the charter.
- XFEVER generation with 6600 verdict and 660 explanation records across eleven aligned sets. Eight translated sets have lower macro F1 than English after grouped correction. Supplied-evidence generation does not establish multilingual RAG or gate transfer.
- Copy/order generation with 2400 verdict and 480 explanation records over 600 claims, plus a post-hoc 9600-record gate policy replay. About 16% of verdicts change under raw repetition or source reversal. Repetition sometimes improves aggregate accuracy; deduplication accuracy gains are not established. Collapsed controls reuse identical baseline prompts and responses.

Results are mixed and include negative outcomes. Numeric and NLI explanation diagnostics do not verify factual explanations. Source-family metadata and exact-copy removal do not authenticate sources. No overall system superiority or publication-readiness claim is established.

## Start and inspect

Follow [START_HERE.md](START_HERE.md) using the existing `D:\apv-rag-tesla369` folder. The fresh integration check is complete: received Windows verification and independent local replay agree for 600 claims and 2400 policy records. No rerun is requested. Exact prompt responses were all reused; fresh feature values and probabilities exactly match their earlier cache. See [the verification](artifacts/fresh_pipeline_verification_v1/RESULTS.md) and [the protocol](docs/FRESH_PIPELINE.md).

The existing Windows audit is closed: 172 tests, nine checks, 341 matching references and zero missing. The integration build previously passed 185 local tests. The subsequent gate copy/order policy replay is checked for 9600 records, with 193 passing local tests and no new neural inference. It transforms fixed per-document scores and reuses 2160 verified responses. The received changed-context batch is verified for 60 claims and 1440 policy records, covering OCR, fabricated citations, authoritative wording, swapped evidence and evidence absence. It reports 240 new feature computations and includes 480 current-run/resumed responses. Windows and local checks agree; no rerun is required. The code suite passed 200 tests before the selective diagnostic; two focused tests pass for its ranking and row-binding logic. Its descriptive results are in artifacts/selective_diagnostic_v1/RESULTS.md. A source/explanation trace audit binds all 1800 saved baseline passages to frozen corpus snapshots and records missing or unknown literal references; see artifacts/source_trace_audit_v1/RESULTS.md. The optional source-snapshot controller replay is in artifacts/snapshot_guard_replay_v1/RESULTS.md. It keeps 600 original outputs and refuses 239 of 240 synthetic text modifications before cached generation; it was not run with live models. The received explanation-to-evidence NLI diagnostic is verified for 840 saved English explanations and 2520 passage pairs, with no new labels or tuning. Its paired descriptive analysis is in artifacts/explanation_diagnostic_analysis_v1/RESULTS.md; no rerun is needed. See docs/EXPLANATION_DIAGNOSTIC.md. See docs/TEXT_STRESS.md. Source authentication, factual explanations, multilingual integration and calibration remain open.

The saved multilingual agreement replay aligns 6,600 Qwen and multilingual NLI outputs. A verdict is retained when the two agree; accepted coverage is 295–393/600 per file and 1,169 correct Qwen answers are rejected across all files. This exploratory supplied-evidence diagnostic uses no new inference and does not establish multilingual retrieval, gate transfer or calibration. See [multilingual agreement results](artifacts/multilingual_agreement_replay_v1/RESULTS.md).

The frozen XFEVER reliability audit measures multilingual NLI top-label ECE and multiclass Brier per supplied-evidence file. Mean confidence is .903–.964 against accuracy .663–.727, with no calibrator fitted; see [reliability results](artifacts/xfever_reliability_audit_v1/RESULTS.md). This does not calibrate the integrated APV-RAG pipeline.

An opt-in four-policy source-snapshot controller now checks frozen benchmark passages before feature use or generation. Its cached replay preserves 2400 original policy outputs and refuses 299 of 360 stress contexts per policy, including the 60 evidence-absence cases. One swapped climate context still passes matching excerpts; this is not publisher authentication. See [bound controller results](artifacts/bound_policy_integration_v1/RESULTS.md). The earlier completed runner is unchanged.

The current non-neural reproducibility audit is in [artifacts/reproducibility_audit_2026_10_04/RESULTS.md](artifacts/reproducibility_audit_2026_10_04/RESULTS.md): the full test suite and eight other commands pass. It verifies 275 present file references and lists 30 missing files from older exports; those gaps are not repaired by the consolidated ZIP. Neural model weights and clean-environment reproduction were not checked.

A new 300-claim external FEVER evaluation input is frozen in `data/external/fever/heldout_v1` after excluding previously observed XFEVER IDs and matching texts. This contains model-only inputs and separate gold. The received 300-claim English NLI baseline has been verified against those frozen inputs. A disk-backed FTS5 adapter can index the separately downloaded official Wikipedia archive and save claim-only retrieval contexts; the received Windows NLI outputs and scorer results now pass probability/cache binding and full metric replay. Accuracy is 47.33%, macro F1 is 0.4694, and complete gold sentence-group recall is 54.08% (106/196 verifiable claims). Mean confidence is 76.83%. These are baseline diagnostics, not integrated APV-RAG accuracy. See [verified FEVER results](artifacts/fever_audit_v1/RESULTS.md). This cohort is now observed; future model or policy changes evaluated on it are exploratory. See [FEVER external evaluation](docs/FEVER_EXTERNAL_EVALUATION.md).

Current completion requirements are in [PROJECT_STATUS.md](artifacts/project_status_consolidated/PROJECT_STATUS.md). The engineering history is in [COMPLETION_AUDIT.md](docs/COMPLETION_AUDIT.md). Earlier protocol documents preserve the original plans and may describe then-pending stages that have since run; use the current status and verified outputs for present claims.

## Main paths

| Path | Contents |
|---|---|
| `src/apv_rag` | Validation, retrieval, features, gates, generation interfaces and analysis helpers |
| `scripts` | Experiment runners, validators and the current audit command |
| `tests` | Synthetic regression and integrity checks |
| `data/external` | Frozen benchmark inputs included in the package |
| `artifacts/*_received` | Received experiment result exports |
| `artifacts/*_verification` | Non-neural replay and analysis outputs |
| `docs/PROJECT_CHARTER.md` | Binding questions, planned system and evaluation requirements |
| `docs/MACHINE_ONLY_PROTOCOL.md` | Existing-label evaluation and unlabeled Tesla boundary |

Model weights, SQLite inference caches and some earlier large arrays are not included. Preserve the original Windows files and their checksums. The package alone does not reproduce every historical experiment.

A development-only temperature-calibration and fresh-claim confirmation runner is now provided in `scripts/run_fever_calibration.py`. Its frozen inputs contain 600 development and 300 confirmation claims disjoint by ID and normalized text from each other and the project's prior FEVER/XFEVER outcomes. The fitted scalar is saved before confirmation inference; the received Windows run is now verified for 900 predictions and 2692 cached premise scores, with exact confirmation metric replay. On 300 confirmation claims, ECE falls from 0.2539 to 0.0227, NLL from 1.5623 to 0.9303, and Brier from 0.6638 to 0.5606. Accuracy stays 53%; top-confidence subset accuracy at 50% and 80% coverage decreases. See [verified calibration confirmation](artifacts/fever_calibration_audit_v1/RESULTS.md). This calibrates the retrieved-evidence English NLI baseline and preserves its predicted labels. That temperature fit applies only to the NLI baseline. The later integrated English experiment below uses separate development-only final-answer correctness fits. See [calibration protocol](docs/FEVER_CALIBRATION_PROTOCOL.md).

The integrated English FEVER runner is now provided in `scripts/run_fever_pipeline.py`: 300 new development and 300 new confirmation claims, all four established gate policies, fresh context-bound features, Qwen verdict/explanation generation, numeric checks and policy-specific correctness-confidence fitting. Settings and gate heads are fixed; development fits are frozen before confirmation inference. The Windows run and all 2,400 policy records are now verified; details follow below. This evaluates the current English controller and does not establish historical publisher authentication, explanation truth or multilingual retrieval/gate transfer. See [pipeline protocol](docs/FEVER_PIPELINE_PROTOCOL.md).


### Verified integrated English FEVER results

The Windows pipeline export has been independently audited without rerunning neural models. All 600 claims, 2,400 policy decisions, 600 context-bound feature records, 1,789 premise scores and 1,200 distinct cached responses passed replay. Selection and upstream gold bindings replay exactly; the confirmation summary reproduces exactly. Development-only correctness calibration coefficients differ by at most 1.11e-16 in the local refit.

On the 300 confirmation claims, all-claim accuracy (abstentions counted as errors) is 53.33% without a gate, 36.00% with the NLI gate, 50.67% with the embedding gate, and 51.67% with the combined gate. The combined gate removes five correct and six incorrect accepted control answers. Final-answer correctness calibration improves binary Brier and NLL for all four policies but does not change verdicts. These findings do not establish gate superiority.

Read `artifacts/fever_pipeline_audit_v1/RESULTS.md` and `verification.json` for verified metrics and uncertainty limitations. `scripts/verify_fever_pipeline.py` can reproduce this audit using the saved export and frozen inputs; choose a new `--output` path to preserve existing audit files. Neural inference, tokenizer clipping/token counts and full-index retrieval were not independently repeated. Historical publisher authentication, semantic explanation truth and full multilingual retrieval/gate transfer remain unestablished; verification of this English experiment does not establish completion of the full original project. No manuscript or GitHub push was performed.


### Equal-coverage gate controls

The frozen English FEVER results now include two simple selection controls at each gate's accepted-answer count. At 192 answers, NLI gating achieves 56.25% accuracy and generated-label NLI ranking achieves 63.02%. No learned gate advantage is established after correction across six exploratory comparisons. The analysis preserves all responses and settings and uses 269 connected page/claim groups, largest size four. See `artifacts/fever_selective_analysis_v1/RESULTS.md`. Updated experiment status and unresolved charter requirements are recorded in `artifacts/project_status_consolidated/PROJECT_STATUS.md`. No new inference, policy tuning, manuscript or GitHub push is included.


### Multilingual excerpt-pool retrieval

`artifacts/xfever_retrieval_pool_v1` contains claim-only retrieval for 6,600 queries per analyzer across eleven aligned XFEVER files. The two fixed BM25 analyzers search pools of 538–571 evidence excerpts per file. Every target is present by construction; this is an optimistic closed-pool diagnostic on already observed data. On the 400 support/refute pairs per file, CJK character tokens raise paired target hit@3 from 29.75% to 68.50% for machine-translated Japanese and from 21.75% to 71.50% for machine-translated Chinese. These rates are not factual-verdict accuracy. Full-page/open-web retrieval and downstream multilingual NLI/generator/gate/calibration transfer remain open.

The stage uses no neural models or new human annotation. `scripts/run_xfever_retrieval.py --verify` reproduces corpus/query bindings, document IDs, rankings and all metrics, with 1e-12 absolute tolerance on BM25 scores for platform math differences. No new model download or manuscript is included.


### Next run: multilingual retrieved controller

`scripts/run_multilingual_retrieved_pipeline.py` is prepared for the existing sixty aligned claim IDs in eleven variants: 660 retrieved contexts, two separate direct-NLI baselines and 2,640 generator/gate policy records. Gates retain their original English features. There is no target fitting, imported generator response, or reused English confidence calibrator. The resumable Windows command and protocol are in [MULTILINGUAL_RETRIEVED_PIPELINE.md](docs/MULTILINGUAL_RETRIEVED_PIPELINE.md). Actual neural outputs remain pending; local preflight and synthetic replay tests do not establish model performance. Upload the automatically created `multilingual_retrieved_pipeline_outputs.zip` after the Windows run.


### Verified multilingual retrieved controller

The previously pending Windows run is now received and audited: 660 queries, 2,640 policy records, 660 feature records per model, 1,978 passage scores per NLI model and 1,316 distinct cached responses. Multilingual NLI accuracy is numerically higher on all ten translated variants and lower on English. No gate improves all-query accuracy over its common ungated answers; acceptance and correct/wrong removals are reported separately. Correctness confidence remains unavailable. See [verified results](artifacts/multilingual_retrieved_pipeline_audit_v1/RESULTS.md). The earlier protocol/pending notes preserve the preparation history; current stage results are authoritative.

This closes the scoped retrieved-context comparison on the already observed sixty-claim sample and target-derived excerpt pools. The neural models and tokenizer were not independently rerun here. Full-page/open-web search, historical publisher authentication, factual explanation validation and multilingual correctness calibration remain unestablished. The full charter is unfinished. No manuscript or GitHub push was performed.


### Grouped multilingual correctness calibration

The five-fold analysis is completed on the saved sixty-claim/eleven-variant controller run. All variants of each evaluated claim are excluded together from its policy-specific fit. Twenty logistic calibrators and a development-prevalence control produce held-out confidence estimates without changing any verdict, gate or threshold. Brier/NLL/ECE decrease versus raw NLI scores in all 44 file/policy cells; versus prevalence alone, logistic Brier/NLL improve in 33 cells and ECE in 6. The pooled prevalence control has lower ECE for every policy. All mixed results are retained in [the report](artifacts/multilingual_correctness_calibration_v1/RESULTS.md).

This is exploratory grouped validation on already observed outcomes, not independent confirmation or calibrated deployment. No neural model is rerun, no human labels are created and no all-data production fit is supplied. `scripts/analyze_multilingual_calibration.py --verify` reproduces the fold fits, confidence estimates and metrics. The original charter and independent new-claim confidence confirmation remain open; no manuscript or GitHub push was performed.

### Frozen new-claim multilingual confirmation: preparation history

The larger official XFEVER test files supply 100 newly frozen project-held-out claim IDs across six variants (600 queries). Selection ignores labels, excludes 2,385 previously used IDs plus normalized prior claim texts, source pages and paired excerpts, and prevents shared source pages among new claims. Four fixed logistic correctness models and four prevalence controls use only the earlier sixty-claim development outcomes. Preparation, retrieval and development fitting have been replayed locally; confirmation neural inference has not run.

Extract the update into `D:\apv-rag-tesla369`, then run `scripts/run_multilingual_confirmation.py` with the existing virtual-environment interpreter. It checks local model bytes, resumes identical caches, applies the exact frozen models without a Windows refit, and produces `multilingual_confirmation_outputs.zip`. Upload that output for audit. `--preflight` checks preparation without inference. See [the fixed protocol](docs/MULTILINGUAL_CONFIRMATION.md). The larger target-derived excerpt pools and upstream machine translations limit the scope; project separation does not prove absence of model pretraining exposure. No manuscript or GitHub push is included.

### Verified new-claim multilingual confirmation

The Windows export is received and replayed for 600 queries, 2,400 policy records and 1,200 model responses. The fixed development-only calibrators remain unchanged. Compared with raw generated-label NLI scores, logistic Brier, NLL and ECE decrease in all 24 language/policy cells. Against constant development-prevalence confidence, logistic Brier and NLL decrease in 21 cells; prevalence has lower ECE in every cell. No gate improves all-query accuracy over common no-gate answers in any language. [The verified report](artifacts/multilingual_confirmation_audit_v1/RESULTS.md) preserves all comparisons.

This experiment is complete within its machine-translated closed excerpt-pool scope. Do not rerun or tune it on these observed claims. `scripts/verify_multilingual_confirmation.py --verify` replays the received export and report without neural inference. The full charter, historical authentication, factual explanation verification and full-page/open-web transfer remain open. No manuscript or GitHub push is included.

### Fixed institutional source-origin guard

Three fixed HTTPS pages from the existing Tesla sources are captured and replay-bound: the Politika issue catalogue, the Library of Congress guide, and the Tesla Museum navigation page. Eight altered or missing-source cases per page are refused before NLI scoring; a whitespace control passes. Run `scripts/audit_source_origin.py --verify` to replay the shipped audit offline. See [the protocol](docs/SOURCE_ORIGIN_PROTOCOL.md) and [capture results](artifacts/source_origin_audit_v1/RESULTS.md).

This completes a metadata/navigation admission layer. Unsigned receipts assume a trusted collector; they cannot independently prove past TLS exchanges. Newspaper images, Tesla quotations, author/date truth, embedded resources and publisher honesty remain unverified. No model accuracy or provenance benefit is established. No new human labels, manuscript or GitHub push are involved.

### Exact cited-excerpt evidence displays

The new display layer replays 3,840 saved English policy records from 960 claim/condition contexts. It produces 5,637 literal cited excerpts bound to source document IDs, selected-sentence indices and frozen document hashes. A mismatched context withholds the displayed candidate; upstream abstentions stay abstentions. Generated rationale text is excluded, and original model outputs and reported benchmark results remain unchanged.

Run `scripts/audit_evidence_display.py --verify` offline. See [the display protocol](docs/EVIDENCE_DISPLAY_PROTOCOL.md) and [the replay report](artifacts/evidence_display_audit_v1/RESULTS.md). These evidence packets check citation traceability. They do not establish the verdict-evidence relationship, semantic explanation truth, relevance, completeness or publisher authenticity. No new model inference, human labels, manuscript or GitHub push are involved.


## Saved guard trade-off audit

All 3,840 saved policy records are paired with existing benchmark labels. The cited-evidence guard preserves original candidates but withholds correct and incorrect answers on altered contexts. The no-gate OCR control loses 28 correct and 32 incorrect answers among 60 cases; one swapped no-gate context still passes incorrectly. These descriptive, overlapping-context results do not establish general robustness or accuracy benefit. Original outputs remain unchanged. Run `scripts/analyze_guard_tradeoffs.py --verify` offline; see `artifacts/guard_tradeoffs_v1/RESULTS.md`. No neural inference, new human labels, manuscript or GitHub push is involved. The full charter remains unfinished.


## One-command current export checks

`scripts/check_current_release.py` runs pytest and five current export/status checks with the same interpreter and explicit project import paths. It saves per-check logs, installed package versions, open research requirements and a single `current_release_checks_outputs.zip`. Each run keeps separate logs; failed or timed-out checks are recorded and cause a nonzero exit. This is a current export check, not clean neural reproduction or completion of the original charter. See `docs/CURRENT_RELEASE_CHECK.md`. No neural inference, manuscript or GitHub push is performed.
