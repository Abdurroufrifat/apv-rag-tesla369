# Project completion audit

## Current status, 2026-10-04

The dated entries below preserve the project history. The current checklist is `artifacts/project_status_consolidated/PROJECT_STATUS.md`. The English FEVER NLI baseline, separate NLI calibration, integrated generator/gate confirmation and equal-coverage controls are now evaluated and verified. None of six exploratory equal-coverage comparisons establishes a learned-gate advantage after Holm correction. Final-answer correctness calibration improves probability losses on accepted English answers. Historical source authentication, explanation truth and full multilingual retrieval/gate/calibration transfer remain unestablished. The original charter is not marked complete, and no manuscript or GitHub push is performed.

Audit date: 2026-10-01. This package consolidates the existing implementation
through Phase 2K. It is not a completed journal evaluation.

## Completed implementation

- Pinned AVeriTeC acquisition and leakage-controlled internal split.
- Evidence, imbalance-aware, disagreement, and provenance-feature baselines.
- Calibration and selective prediction.
- Repetition, OCR, conservative paraphrase, and synthetic citation stress tests.
- Saved predictions, checksum manifests, audit logs, bootstrap intervals, and figures.
- Machine-only protocol; Tesla claims remain unlabeled archival observations.

## Publication requirements still open

The existing benchmark plan and binding protocol require the following work.
None is satisfied merely by passing software tests.

| Requirement | Current status |
|---|---|
| Untouched primary evaluation | Frozen official-dev excerpt evaluation completed; 461 independent claims, 39 training-linked exclusions; selected-rule gains were not statistically significant |
| BM25+NLI, dense retrieval+NLI, standard and source-weighted RAG | Five-seed BM25/dense NLI oracle-excerpt results completed; full RAG comparisons remain open |
| Full APV-RAG and planned component ablations | Classical feature baselines exist; full retrieval-system comparison remains open |
| Five fixed seeds | NLI comparison and held-out excerpt evaluation use 369, 1369, 2369, 3369, 4369 |
| Missing-primary-source and multilingual-mismatch tests | Not completed |
| SciFact and XFEVER transfer tracks | SciFact fixed BM25 abstract transfer completed: 300 claims, 247 groups; expanded features significantly worse under both rules. XFEVER supplied-evidence English-model transfer stress completed (11 files, 600 rows each); multilingual-model supplied-evidence control completed: ten translated-file gains significant after Holm correction; English gain not significant; exploratory comparison |
| Paired randomization and Holm correction | Completed for the two frozen excerpt decision comparisons; planned full-system comparisons remain open |
| Licenses, compute reporting, paper assembly | Require a final audit and manuscript work |

Do not rename the current classical classifier as a evaluated full RAG system.
Do not use internal-validation improvements as final held-out test claims.
Do not assign historical gold verdicts to the Tesla collection.

Official dev has now been observed. Revised methods require separate new
confirmation data. Descriptive errors are recorded in
`artifacts/heldout_error_analysis/ERROR_ANALYSIS.md`; this does not establish
retrieval or NLI failure causes. The received held-out files are a partial
export in `artifacts/heldout_received`, not a full local inference cache.

Post-evaluation development includes own-excerpt substitution/refitting,
an eighteen-feature representation, group ablations and fixed-retrieval
representation comparisons. These are internal-validation diagnostics.
The four paired representation tests are saved in
`artifacts/retrieved_feature_statistics/STATISTICS.md`. Only dense argmax
has Holm-adjusted p below 0.05 within that family; correction does not cover
all preceding adaptive feature searches. No revised method has new independent
confirmation, and these additions do not complete open-web APV-RAG.

## Corrections and interpretation

Phase 2J's bootstrap implementation computes Macro-F1 differences, not
prediction-flip-rate intervals. Its zero intervals therefore refer to Macro-F1
change. Only 30 of 609 claims changed under those rules.

Phase 2K inserts claim-like text plus synthetic citation metadata. The current
text verifier does not authenticate the URL. Its measurements concern influence
of appended synthetic evidence, not a standalone fake-URL detection capability.

Provenance-family collapse's zero repetition shift follows from removing exact
same-family copies. It is a preprocessing invariance result, not proof that
the learned model has higher accuracy.

## Windows installation

Extract the consolidated ZIP directly into `D:\apv-rag-tesla369`, preserving
the project-root layout. Back up local changes first; do not overwrite newer
local work without checking it. No virtual environment is included.

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\validate_citation_insertion_stress.py
```

The saved experiment outputs are included. These commands do not retrain models.
No GitHub push is required or performed.

## Updated project boundary after SciFact

SciFact transfer is completed and archived in `artifacts/scifact_received` and
`artifacts/scifact_statistics`. It does not support cross-domain improvement
from the expanded representation. The untuned direct NLI control has been
implemented but awaits the local cached-audit run. It is a post-result diagnostic,
not a new preregistered confirmation.

The project still requires full retrieval-system comparisons, planned component
ablations, missing-primary-source and multilingual tests, XFEVER evaluation, and
license/compute/reproducibility auditing. Existing classifier experiments do not
complete these requirements. If scope changes are proposed, record them explicitly;
do not silently mark unimplemented requirements complete.

Manuscript work is paused until the user explicitly approves writing it.
GitHub publication remains paused until project completion.

## XFEVER transfer update

All 6,600 predictions and scores were verified. Ten prespecified grouped paired
comparisons use 393 connected components sharing English claim IDs or pages.
Indonesian, Japanese, and Chinese variants decline significantly after Holm
correction; Spanish and French do not. This is English-model transfer on supplied
evidence, not multilingual retrieval or full RAG. Results are archived in
`artifacts/xfever_statistics`. No project human review was required.


## Multilingual model control update

The received float32 control produced 6,600 verified predictions on the same eleven input files. Both models' probabilities, decisions and metrics were checked for 13,200 predictions. Paired analysis used 393 connected groups, 2,000 bootstrap samples, 10,000 group swaps and Holm correction across eleven model comparisons. All ten translated-file gains were significant; the English-file gain was not. Multilingual macro-F1 ranged from 0.6927 to 0.7267 on translated files, versus 0.4806 to 0.6253 for the English model.

This comparison was specified after observing the English results. Potential FEVER-derived training overlap, different model architectures and training data limit interpretation. Inference was not independently rerun, and local model weights were unavailable for inspection. This closes the supplied-evidence multilingual model comparison only. Full generative RAG, learned sufficiency and source-quality efficacy remain unfinished. No manuscript was written.


## Learned NEI proxy update

The learned benchmark-label proxy experiment is complete and verified: 12,180 exported predictions from four method/feature combinations across five seeds on 609 internal validation claims, 375 groups. There are 57 NEI and 552 non-NEI claims. Expanded BM25 features produced mean macro-F1 0.5312 and AUROC 0.6018; expanded dense features produced 0.5231 and 0.6014. The training-prior baseline macro-F1 was 0.4755. No statistical superiority has been established by this descriptive comparison.

The learned probabilities had substantially worse Brier scores (0.2382–0.2475) than the training-prior baseline (0.0848). Positive AP is partly explained by 90.64% non-NEI prevalence. Do not treat these probabilities as calibrated confidence or deploy the proxy as a sufficiency gate. Labels concern the benchmark claim, not the quality of retrieved passages. Raw OOF probabilities, model fitting and feature matrices were not independently replayed during export verification. Actual retrieved-evidence sufficiency and downstream selective performance remain open. Official development records used by the runner: zero. Manuscript writing remains paused.


## Generative baseline failure

The first generative SciFact run completed and its 300 outputs were verified. All 300 failed the frozen answer-format contract, causing 0% coverage and zero all-claim accuracy/macro-F1. Raw output was exactly Not Enough Evidence for 295 claims; the remaining five were malformed variations. Parser tests did not establish that the selected generator would follow the contract. This is a failed baseline, not completion of cited generative RAG. Keep its outputs unchanged. Any prompt, decoder or model repair must use a new output identity and separate cache. Neural inference and tokenizer clipping were not independently replayed during verification. Model-interface repair and subsequent grounded-output evaluation remain open.


## Generative interface v2 smoke failure

The four-case synthetic smoke export was verified: two correct verdicts, two structurally valid answers, full-run readiness false. The support answer omitted its citation. The contradiction case was predicted NEI and its explanation repeated the contradicted claim. The unrelated-evidence case was predicted NEI. Empty evidence was predicted Supported with an uncited explanation. Constrained decoding fixed label vocabulary but did not fix evidence use or explanation grounding. Keep the full generative benchmark blocked. Preserve this result; do not weaken the smoke criterion or count synthetic fixtures as research evaluation. Neural inference and semantic explanation support were not independently replayed. Next repair must address evidence-conditioned verdicts and grounded explanations before a new full run.


## Hybrid generation v3 smoke failure

The four-case synthetic hybrid smoke export was verified. NLI predicted three of four expected stances; two generated summaries passed the NLI entailment check, but full-run readiness remained false. On support evidence the generator changed 20 degrees Celsius to 0.005; the guard rejected it. Contradiction passed. Unrelated evidence was wrongly labeled Refuted and its factual summary passed entailment. Empty evidence abstained without generation. Thus summary entailment cannot establish claim relevance or correct stance. Preserve the failed export. Do not treat heuristic NLI as gold semantic verification or relax readiness. Generation fidelity and evidence relevance require a separate, explicit repair. No full generative benchmark should run with this configuration. Manuscript writing remains paused.


## Replacement instruction model v4 smoke failure

The received pinned Qwen2.5-1.5B-Instruct smoke export was verified. One of four expected actions passed: the application empty-evidence abstention. All three actual model generations failed the frozen structural contract. Support returned a literal verdict placeholder with an unknown citation format; contradiction and unrelated evidence omitted the delimiter and explanation. No usable cited-generation improvement is established. Full-run readiness is false. Preserve the failed export and do not spend another full benchmark run on this configuration. Next work should inspect the model/prompt interface with isolated component diagnostics before further end-to-end changes. Model inference was not independently replayed. Full generative RAG remains unfinished; manuscript writing remains paused.


## Separated instruction v5 verification

All three generated verdicts match their synthetic fixtures and the empty-evidence case abstains. The original contract passed three of four actions: its source-only numeric guard rejected the NEI explanation for mentioning 20 degrees supplied in the claim. A separate numeric-provenance check now distinguishes evidence values, claim-only references and values absent from both inputs. Offline replay of existing outputs passes four of four structural actions under the revised check; original results are preserved. This is adaptive guard development, not an independent model run or semantic grounding validation. Input-present numbers can still be misrepresented. The corrected module is not automatically deployed into a full benchmark runner. Multi-passage source selection, explanation grounding and full generative evaluation remain open. Manuscript writing remains paused.


## Separated RAG: received full run

300 claims verified by scripts/verify_separated_rag.py: raw coverage 181/300, correct 95/300, macro F1 .3684; guarded coverage 172/300, correct 91/300, macro F1 .3627. Invalid verdicts:119; novel numeric values:9. Frozen parser preserved. BM25 IDs/scores, integrity and scoring replay passed. Neural generation and token clipping were not rerun. Output reliability remains unresolved; this is exploratory, not project completion. See artifacts/separated_rag_scifact_verification/RESULTS.md.


## Constrained RAG received results

300 records passed non-neural integrity, retrieval and scoring replay. All verdicts valid. Guarded coverage284/300, correct144/300, macro F1 .4395; 16 numeric rejections. Original correct91/300 all retained plus53 newly correct. Exploratory comparison; not independent confirmation. Neural inference not rerun. See artifacts/constrained_rag_verification/COMPARISON.md. Project requirements remain unresolved.


## Context visibility audit

Gold document top-three hit159/188 (84.57%); complete gold sentence set exact-text visibility21/188 (11.17%). Diagnostic only; no gold-driven retrieval or generation and no NLI threshold chosen. Suggest claim-relevant sentence excerpts as exploratory follow-up. Full grounding remains unresolved. See artifacts/constrained_rag_audit/RESULTS.md.


## Sentence-selected RAG received run

All300 received rows passed hashes, BM25 IDs/scores, selected sentence indices, guard and metric replay. Guarded correct152/300, coverage284/300, macro F1 .47994. Previous correct144/300:43 newly correct,35 newly wrong. Gold sentence-set exact visibility110/188 vs21/188. No independent confirmation or significance claim. Neural generation/token clipping not rerun; explanation accuracy remains unverified.


## Numeric guard audit

14/16 rejections explained entirely by numbered list markers. Two remain:690 unit-attached5mmol/L missed in original extraction;1339 generated baseline1 absent from supplied inputs. Post-hoc list-marker sensitivity correct162/300, coverage298/300, macroF1 .4972; frozen original results unchanged. Correction: rejected correct verdicts are not automatically false explanation rejections. See artifacts/numeric_rejection_audit/RESULTS.md.


## Numeric extractor v2

Separate extractor excludes list markers, recognizes specified unit-attached numbers, canonicalizes decimal forms without rounding and excludes digits inside gene identifiers. Five regression tests and nine boundary checks passed. Post-hoc replay:163/300 correct,299/300 coverage,macroF1 .49907; claim1339 still rejected. Original generation and frozen scores unchanged. Units/semantic meaning remain unchecked.


## Frozen CLIMATE-FEVER received result

300 records passed non-neural replay. Raw114/300 correct,guarded108/300,coverage290/300,macroF1 .36325 guarded. Majority-class descriptive accuracy46.67%; model raw38%,guarded36%. Cross-domain strength not supported. Preserve frozen failure; diagnostic follow-up only. See artifacts/climate_rag_verification/ASSESSMENT.md.


## Climate evidence-loss diagnostic

194 support/refute claims:matching-stance article absent92,sentence notselected17,selected sentence notfullyvisible13,visible72. Visible cases guardedcorrect25/72. NEI106 analyzedseparately. Stages are textvisibility diagnostics,notcausal proof. Frozenresults unchanged; retrieval and reasoning require investigation.


## Supplied climate evidence diagnostic

All300 passed non-neural checks. Raw120/300,guarded110/300,coverage279/300,macroF1 .37677. Retrieval guarded108/300;36 newlycorrect,34 newlywrong. Suppliedevidence doesnotresolveweakperformance. Contextcount/clipping/annotations/greedydecoding limitcausalinterpretation. Frozenresult preserved.


## Climate decoder comparison

600 records passed non-neural identity/arithmetic/score replay. Retrieved greedy38%,sum35.67%,mean36.33%;supplied greedy40%,sum35.67%,mean36.33%. Alternativedecoders do notresolveweaktransfer. Stopdecodertuning onobservedcohort. Noindependentconfirmation; neural logits notrerun.


## Matched semantic sufficiency verification

580features,294validationpredictions passed non-neural feature/classifier/metrics replay. 98validationexamples:embeddingaccuracy .8878,combined .8776,NLI .6327. Length/claimonlycontrols .5. Constructedannotation-completeness discrimination demonstrated; generalrealretrievedsufficiency notvalidated. Experimentalintegration only; retainallfeaturevariants/no-gatecontrol.


## Experimental gate integration replay

Added a runner comparing NLI, embedding and combined gates with a common numeric-checked no-gate policy on 900 saved RAG records. Synthetic repeated-source checks measure gate sensitivity only. The helper reproduced 294 serialized validation probabilities; context/copy, numeric policy, syntax, import and boundary checks passed. Neural integration results are pending the user model run. Full generation integration and source authentication remain open.


## Integrated gate results and matched-coverage audit

All 900 uploaded records passed non-neural identity, cache, feature, classifier, numeric policy, decision, copy and metric replay. At threshold 0.5, NLI answered accuracy/coverage: SciFact 64.7%/22.7%, retrieved climate 53.1%/16.3%, supplied climate 32.8%/19.3%. At those accepted counts, NLI exceeds simple cosine/verdict-NLI baselines in the first two cohorts; supplied climate is worse than verdict-NLI mean (53.4%). Embedding/combined answered accuracy is lower than the common no-gate policy in all cohorts. Copy collapse preserves all inputs; without collapse, combined/embedding/NLI scores cross threshold in 52/46/11 of 900 cases. This is mixed exploratory selective-prediction evidence, not a general sufficiency, source authentication or explanation-quality result. Thresholds remain frozen. Full generation integration, multilingual generation and final reproducibility checks remain open.


## Gate before generation controller

Added cached/live frozen-context execution with gate before all generator requests and numeric checks after explanation. Seven new flow tests passed. All 3600 cached outcomes (900 claims times four policies) match the verified replay, and all early rejects have zero generation requests. No neural inference was run locally. The full pytest command could not run because pytest is absent. Live Qwen execution on the user PC remains pending. Retrieval/features remain frozen, requests are not measured runtime savings, and source authentication, explanation-quality validation and multilingual generation remain open.


## Live gate controller output verification

All 3600 records (900 claims, four policies) passed non-neural source/code/protocol identity, controller, prompt/context, reported token budget, shared-answer, numeric, metric and count checks. All 934 early policy rejections have zero generation requests. All 3600 decisions match the prior replay; all 900 no-gate verdicts and explanations match original saved generation. The receipt reports 1800 cached live answers, consistent with distinct prompt keys; SQLite and model inference were not independently rerun/inspected. Reused source NLI explanation diagnostics for exact matching text/context. These scores do not establish explanation correctness, source authenticity or safe abstention. Live controller reproducibility on frozen contexts is supported; multilingual generation, fresh retrieval integration and final audits remain open.


## Multilingual generation runner prepared

Added full eleven-file XFEVER Qwen verdict evaluation (6600 records) and a fixed label-independent 60-claim-ID explanation subset (660 records). Five helper tests and full input hash/alignment/subset/syntax/import checks passed. Actual model inference remains pending on the user PC. Reuses the pinned live generation backend and local Qwen model, with resumable cache and separate full-verdict/subset-guard metrics. No multilingual retrieval, learned completeness gate or explanation-quality validation is claimed.


## Multilingual generation output verification and grouped analysis

All 6600 verdicts and 660 fixed-subset explanations passed non-neural export/source/code/protocol, exact text/prompt, reported token budget, subset, shared-answer, numeric and metric checks. All source text is unaltered; no clipping. Generator declarations match the prior controller receipt; local model bytes, tokenizer counts, SQLite and neural inference were not independently checked. Paired analysis uses 393 connected claim/page groups, 2000 bootstrap samples and 10000 swaps. Holm confirms lower macro F1 than English in eight translated sets; French differences are not confirmed. Qwen has no confirmed gain over multilingual NLI and is lower in seven of eleven sets in exploratory model comparisons. English accuracy 71.0%, macro F1 .6963. Japanese machine/human accuracy 55.0%/62.0%. Keep all results and settings frozen. Generation robustness, source authentication, fresh retrieval integration, explanation-quality validation and final reproducibility checks remain open.


## Generation robustness runner prepared

Added three-copy raw/collapsed and reversed-order generation controls on 600 frozen SciFact/retrieved-climate claims, with baseline reuse. All 17 relevant generation tests and 600 baseline prompt/context checks passed. Prepared 2400 verdict records and 480 explanation records using fixed label-independent explanation subsets. Live perturbation inference remains pending. This evaluates ungated synthetic-copy/order sensitivity and exact normalization, not source authenticity or learned-gate robustness.


## Generation copy/order results and corrected status (2026-10-03)

The received synthetic stress export passed all 2400 verdict and 480 selected-explanation checks against 600 frozen claims. Exact source/context/prompt binding, generator/package declarations, source/code/protocol hashes, shared responses, deterministic selection, numeric policy, counts and metrics replayed. The export contains 2160 distinct prompt keys including 720 frozen baseline keys. Model weights, inference, tokenizer counts and SQLite were not independently checked here.

Repetition changed 47/300 SciFact verdicts and 48/300 retrieved-climate verdicts. Reversal changed 47/300 and 49/300. Raw repetition increased accuracy from 54.67% to 59.67% and from 38.00% to 40.00%; these changes do not demonstrate deduplication accuracy gains. Correct/wrong transitions and explanation text changes are preserved. Collapsed controls reconstruct baseline prompts and reuse responses, so equality is enforced rather than independently inferred.

Exploratory paired analysis uses connected normalized original claims or shared original retrieved IDs, 2000 group bootstrap samples, 10000 swaps and seed 369. SciFact has 182 groups; climate has 28, including a 262-claim component. None of four macro F1 comparisons passes Holm correction. Intervals are marginal and large components limit inference. The grouping rule was specified during export analysis, not preregistered. No full-system calibration or authentication claim follows.

Seventeen relevant existing unit tests passed. Four mutation probes rejected duplicate rows, changed source text, changed prompts and changed guard decisions; original received files remained unchanged. The full pytest suite was not run in this environment.

The consolidated status generator and checklist now include constructed sufficiency, live gating, multilingual generation and copy/order work, replacing obsolete open-stage descriptions. Fresh retrieval integration, source authentication, explanation correctness, complete integrated robustness/calibration and final reproducibility remain incomplete against the binding charter. No new annotation, manuscript, GitHub push or scope reduction was performed. The next bounded task is a current reproducibility/evaluation inventory, followed by the remaining integration gaps.


## Current reproducibility inventory and Windows setup (2026-10-03)

A single non-neural audit entry point now inventories output/audit receipts, records installed and received package versions, parses Python sources, runs the full test suite and executes the current semantic, retrieval, controller, multilingual, copy/order and status replay commands. The nine command checks passed here. The full suite passed 168 tests, including eight new integrity tests that first failed before the auditor was implemented. The receipt tests distinguish omitted files from corrupt bytes and reject invalid digests and paths outside their receipt folder.

The current audit verified 144 referenced files and found 30 missing references in older selected-result exports, with no mismatched or invalid checksums. The missing identities, retrieval/excerpt audits and arrays are listed individually in artifacts/reproducibility_audit_v1/missing_export_files.json. They are absent from the current package; they are not reconstructed or marked verified. Original Windows run folders may retain them. Checksums and logs support the current exported experiment chain, not complete reproduction of every historical run.

Windows setup now installs the editable package after lightweight dependencies. README.md and START_HERE.md now point to the current status and audit instead of obsolete starter stages or unused-dev claims. The optional Colab sentence-transformers upper bound was widened to permit the received feature-run version 6.1.0. Recorded top-level generator/feature package versions are saved separately from the current audit runtime; they are not complete locks or a tested clean neural installer.

The audit ran on Linux using Python 3.12.14, with pytest 8.4.2 temporarily available for this check. A fresh Windows installation, model weights, tokenizer recounts, neural inference and SQLite cache bytes were not tested here. The runner itself downloads nothing and performs no neural inference or model fitting. Original experiment inputs and expected digests were not altered. The full project remains unfinished against the charter. Next work is fresh retrieval/feature extraction integration with the live gate controller and the remaining integrated evaluations. No manuscript, GitHub push or new annotation was performed.


## Windows audit receipt and copied-export recovery (2026-10-03)

The Windows audit archive reports 168 tests and nine successful commands on Python 3.14.0, 275 matching file references and 30 missing references. Archive contents, current run code/setup hashes and four frozen package profiles passed verification; seven JSON replay logs match local results. Shared raw receipt hashes match. Windows-only receipts and derived receipt byte differences are recorded explicitly rather than treated as independently inspected bytes.

Every missing received-folder reference has a matching checksum in an original Windows experiment-folder receipt from the same audit. A 30-file recovery map now points to those originals. The standalone recovery command validates unchanged target-receipt hashes, rehashes every original before any copy, refuses corrupt existing targets and copies through temporary files without overwriting originals. The real Windows recovery has not run here. Four new synthetic recovery tests passed, and the complete local suite passed 172 tests. Expected hashes were not changed.

The updated commands restore copied exports and rerun the audit without neural inference. The package itself still lacks the large original arrays; it is not a complete standalone archive. Fresh retrieval integration and the other charter gaps remain open. No new annotation, manuscript or GitHub push was performed.


## Received-export recovery confirmed; Windows test fixture corrected (2026-10-03)

The next Windows audit reports 324 matching references, zero missing references and all 30 recovered alias targets matching their original expected checksums. Eight replay commands passed. Its test suite reported 171 passed and one failed. The failure comes from the new synthetic fixture serializing relative paths with Windows backslashes, while the recovery-map contract requires forward slashes. The two fixture fields now use Path.as_posix(); production recovery code and experiment digests are unchanged.

The complete local suite passed 172 tests after this correction. The uploaded failed Windows test report is preserved, and Windows confirmation of the fixed suite remains pending. No recovery repetition or neural rerun is required. The complete original array archive is still local to the Windows machine. No manuscript or GitHub push was performed.


## Existing Windows audit stage cleared (2026-10-03)

The final Windows audit reports 172 passed tests, nine successful commands, 341 matching file references and zero missing references. The fixture path-format failure is cleared and all 30 recovered aliases retain their original expected checksums. Archive/code/setup identity, four package profiles and seven replay JSON outputs match the corresponding project evidence. The raw final audit and verification summary are preserved in reproducibility_audit_windows_final_received and reproducibility_audit_windows_final_verification.

The current audit/recovery step is closed; no repeat audit is requested. This validates the existing Windows environment and supplied export evidence, not a clean neural installation or independent reading of model/cache bytes. Large arrays remain outside the consolidated package. Fresh retrieval/feature extraction integration and the remaining charter requirements are open. Manuscript and GitHub publication stay paused.


## Fresh retrieval and feature/controller runner prepared (2026-10-03)

Added a reusable raw-corpus retrieval/context component, strict context-bound 13-feature records, all four frozen gate policies, a resumable local-model runner, export verifier and one-command Windows launcher. The runner hashes pinned model bytes, requires the original package profile, rebuilds BM25/sentence selection and Qwen clipping, and infers fresh NLI/embedding features before invoking the generation controller. Feature models leave scope before a lazy Qwen load. Only complete exact original prompts seed generator responses; reused responses and current-run live/resumed responses are reported separately. Earlier model runners, learned heads, experiment inputs and results were not changed.

The full suite passed 185 tests, including 13 new retrieval, feature shape/digest, early-gate, label independence, cache conflict/lazy backend and export mutation tests. Input preflight checked 600 raw claims and frozen receipts. A separate non-neural smoke rebuilt 600 retrieval/sentence contexts from raw inputs using original clipped bytes, exercised 2400 policy records with received features and 1200 exact responses, and replayed the controller/response verifier. Those temporary predictions were discarded and are not a fresh neural experiment. Build evidence is in fresh_pipeline_build_verification_v1.

Fresh neural feature inference, Qwen tokenizer clipping and the Windows launcher have not run in this environment because local model weights and neural dependencies are absent. The next action is the provided Windows launcher in the existing project environment. It resumes an incomplete fresh run or verifies an existing completed one and exports fresh_pipeline_outputs.zip, excluding model weights and SQLite. This engineering equivalence check on observed English cohorts does not establish new quality gains, source authentication or factual explanation correctness. The full project, remaining multilingual/gated stress/calibration requirements, manuscript permission and paused GitHub publication remain unchanged.


## Fresh integration export received and verified (2026-10-03)

Preserved all ten files from fresh_pipeline_outputs.zip in fresh_pipeline_received and fresh_pipeline_windows_verification_received. The independent non-neural verifier passed source, model declarations, package, code, protocol, raw BM25/sentence selection, context, feature shapes/digests/aggregates, frozen-head probabilities, all 2400 policy decisions, exact prompt/response provenance, numeric guards and metric replay checks for 600 claims. The Windows verification files retain their original hashes, and its parsed verification payload exactly matches the local replay. Model bytes, tokenizer clipping, inference and SQLite were not independently read or rerun here.

The progress receipt reports 600 new feature computations. Fresh exported feature aggregates and gate probabilities exactly match the earlier cache, with zero threshold changes. All 1200 distinct responses were reused from prior identical prompts; there are zero current-run Qwen responses. There were 652 early gate rejections, leaving 1748 verdict and 1748 explanation policy requests. Counts are not measured runtime savings. This is engineering integration equivalence on already observed development cohorts, not new performance evidence.

The NLI gate retains covered accuracy 64.71% at 22.67% coverage on SciFact and 53.06% at 16.33% on retrieved climate, compared with no-gate covered accuracy 54.52% and 37.24%. Overall accuracy counts abstentions as errors and falls to 14.67% and 8.67%. Embedding and combined gates do not improve covered accuracy in these cohorts. These results repeat the earlier frozen decisions; they do not establish new superiority or universal sufficiency.

The named fresh integration check is closed. No full test rerun or new neural generation was included in the received archive, and none was needed for this export replay; the prior build evidence remains 185 local tests. Next work is integrated gate copy/order behavior, followed by the remaining authentication, explanation, multilingual and calibration requirements. The full charter remains unfinished. No manuscript, GitHub push or new human annotation was performed.


## Gate copy/order policy replay completed (2026-10-03)

The verified fresh integration and earlier generation stress export provide exact per-document score/response bindings, allowing the bounded gate copy/order task to run without another Windows model job. Added source-bound score transformation and early-gate policy execution, a frozen replay protocol, runner and independent export verifier. The raw-copy condition explicitly disables normalization; collapsed and reverse controls preserve their declared source handling. Earlier frozen code, heads, source inputs and experiment outputs were not changed.

The seven initial focused tests failed before implementation and now pass; an additional export-mutation test also passes. Together they cover copy weighting, parent/text and corrupt-cache rejection, reversed source-score alignment, threshold crossing, early rejection before response requests, explanation subset handling, numeric rejection and export mutations. The full suite passed 193 tests. The verifier recomputed and checked all 9600 policy records, 2400 transformed feature entries and 2160 distinct prior responses. No neural feature or generator call was made.

Raw copies change gate acceptance for SciFact NLI/embedding/combined in 3/16/17 of 300 claims, and retrieved climate in 4/14/16. Reversal changes no acceptance decisions, with completeness probability differences at arithmetic-rounding scale, while saved verdicts can change under source order. Collapsed controls have zero probability, acceptance or output changes by construction. Raw repetition sometimes increases covered accuracy; the recorded comparisons change both presentation and answered claims and do not isolate a normalization benefit. No significance or calibration claim was made.

All-claim metrics score gate-only verdicts. Numeric/explanation metrics score only the fixed label-independent sixty-claim subset per cohort, including early abstentions. Prior answers were generated earlier without these gates, so this is a counterfactual policy replay and request counts do not measure compute savings. Exact per-document feature copying/reordering does not independently test neural batch invariance or changed-text extraction.

The bounded post-hoc copy/order gate replay is closed. Results and scope are preserved in gated_stress_replay_v1 and gated_stress_replay_verification_v1. Next are changed-text, citation and absence controls; the broader authentication, factual explanation, multilingual and calibration charter requirements remain open. No manuscript, GitHub push, new annotation or scope reduction was performed.


## Batched changed-context stress runner prepared (2026-10-03)

Prepared one fixed label-independent sample of thirty claims per cohort, with baseline and five controls: deterministic character-level OCR noise, fabricated archive references, authoritative wording, donor context and complete evidence removal. The sample has sixty claims, 360 contexts and 1440 policy records planned. Sixty verified baseline feature entries and exact generator responses are reused; 240 changed nonempty contexts require fresh features. Original claims/labels, heads and threshold .5 remain unchanged. Source identity and context recipes are frozen before the pending inference.

Seven new focused tests passed. The initial five recipe tests first failed before implementation; two export tests verify binding, policy mutations, duplicate rows and missing prompts. The full suite passed 200 tests. Input preflight saved all selected IDs and exact synthetic contexts. Empty evidence abstains before generation and is not scored as benchmark accuracy. Donor evidence is not declared semantically unrelated; fake citation echo detection is a literal string check, not authentication. Added prefixes and NLI truncation can confound wording effects.

The runner verifies pinned local models and packages, checks all Qwen prompt budgets, extracts fresh features, releases feature models before lazy generation, refuses identity/cache changes, and resumes incomplete work. The one-command Windows launcher verifies the completed export and writes text_stress_outputs.zip without model weights or SQLite. Model inference, tokenizer recount and Windows shell execution have not run here. The current evidence is code/preflight verification only; no fabricated stress results were written.

Remaining source authentication, factual explanation, multilingual and calibration work stays open. Manuscript and GitHub remain paused. No new annotation, fitting or threshold tuning was performed.

## Changed-context stress export received (2026-10-03 UTC)

The latest and prior uploads are byte-identical. Preserved eleven received files and locally verified sixty claims, 360 contexts and 1440 policy records. Windows/local verification payloads agree. The progress receipt reports 240 new feature computations, with sixty baseline entries reused; 480 current-run/resumed responses and 120 exact prior responses are bound to prompts. No neural inference or tokenizer recount was repeated here.

Descriptive results are in text_stress_analysis_v1. All missing-evidence policies abstain without generation. Fabricated citations and authority wording still change some outputs; one distinct SciFact explanation repeats a synthetic reference. No source authentication, explanation truth or confirmatory superiority follows. This stage is closed; broader charter requirements remain open. No new human annotation, manuscript or GitHub publication was performed.

## Frozen-score selective diagnostic (2026-10-03 UTC)

Replayed 600 verified fresh-pipeline development claims using the original no-gate answers and the frozen NLI, embedding and combined scores. Source hashes and verification receipt were checked; no new inference, fit or threshold selection occurred. The report gives risk by ranked coverage and the fixed 0.5 policy counts. In climate retrieval NLI passes 52/300 gates, but numeric checks leave 49 final answers. Gate scores were trained for constructed completeness, not verdict correctness, so these rankings do not establish probability calibration. Source authentication, explanation correctness and multilingual gate integration remain open. Two focused diagnostic tests pass. No manuscript or GitHub push occurred.

## Frozen-corpus source and explanation trace (2026-10-03 UTC)

The raw SciFact and retrieved-climate corpus checksums match pinned source manifests. For all 600 ungated saved baseline claims, all 1800 retrieved passages use a known document ID, the selected sentence indices match claim-specific selection, and each saved excerpt begins with the selected corpus text. This binds the passages to the frozen dataset snapshot; tokenizer clipping lengths, publishers, original webpages and historical authorship were not independently checked.

Across 600 baseline explanations there are zero unknown bracket IDs and zero unseen literal URLs, but 157/158 SciFact and 195/197 climate decisive explanations lack a known bracketed source ID. The prompt did not require citations. Synthetic stress changes break exact current-claim excerpt binding as expected; one distinct SciFact fabricated-reference echo is observed. These are structural checks, not factual explanation judgments. Results and per-condition counts are in artifacts/source_trace_audit_v1. No new annotation or model inference was performed.

An offline exact-snapshot pre-generation rule passes all 600 original contexts and the 60 stress baselines. It refuses all 120 modified SciFact contexts, 119/120 modified climate contexts and all 60 missing-evidence contexts. One swapped climate donor retains three excerpts selected for the current claim, so it passes. The synthetic reference echo would be suppressed in this replay. This rule is not wired into the frozen live controller and has not been tested on real tampering; ordinary OCR differences would be refused. It does not authenticate publishers or prove the excerpts relevant.

## Executable source-snapshot controller replay (2026-10-03 UTC)

A separate optional controller checks each retrieved passage against the pinned corpus source ID, claim-specific selected sentence order and text prefix before calling the existing generation flow. It refuses mismatched or empty contexts without model requests. A cached replay of 600 original and 360 stress contexts preserves all 600 original ungated controller outputs and 60 stress baselines. Of 240 modified nonempty contexts, 239 are refused; one swapped climate context passes because all selected excerpts still match. The one synthetic archive-reference echo is suppressed in the refused context. Three new focused controller tests plus two trace and two selective tests pass. All accepted generation answers are exact prior responses, and no neural runtime saving or general attack-detection rate is measured. The controller is separate from the frozen four-policy experiment. Snapshot matching rejects benign OCR variants and does not authenticate publisher identity or explanation truth.

## Explanation-to-evidence diagnostic prepared (2026-10-03 UTC)

A single resumable Windows runner scores 840 saved English no-gate explanations against 2520 individual passages using the pinned local NLI model. It excludes 60 duplicated stress baselines, missing-evidence cases and all benchmark truth labels. The original frozen results remain unchanged. Oversize 256-token pairs are recorded without truncation; 3-way scores retain contradiction/entailment/neutral order. Inference is pending the existing Windows model environment. Four focused tests pass for selection, literal quote checks, malformed scores and skip handling.

Local preflight found 202 quoted spans: 58 occur in supplied evidence, 87 only in claims and 57 are unmatched under case/whitespace normalization. These are literal flags. A single-passage entailment score, when available, cannot certify full explanation truth, source independence or publisher authenticity. No new human review, manuscript or GitHub push was performed.

## Received explanation diagnostic (2026-10-03 UTC)

The Windows export contains 840 saved English explanations and 2520 NLI explanation–passage pairs, all scored with zero token-budget skips. The local verifier reconstructs inputs and checks hashes, scores and summary. Windows and local verification files have identical normalized text but different raw SHA-256 values because the Windows output uses CRLF line endings; both audit receipts validate against their own raw bytes. No neural inference or tokenizer count was repeated locally.

For the original 300 claims per cohort, mean maximum single-passage entailment is .2701 SciFact and .3306 retrieved climate. Thirty paired claims per cohort support descriptive stress differences; in retrieved climate fabricated citation +.0956 and authoritative wording +.0959 mean paired differences, without significance or causal isolation of prefix length versus regenerated answer. Literal quotes total 202: 58 in evidence, 87 claim-only, 57 unmatched. These are structural/NLI proxy diagnostics, not human-verified factual explanation outcomes. The source/results are preserved in explanation_diagnostic_received and explanation_diagnostic_analysis_v1. Source authenticity, multilingual gate integration and calibration remain open.

## Multilingual agreement replay (2026-10-03 UTC)

The existing XFEVER Qwen and multilingual NLI outputs were aligned by file and row with claim, English page and label equality, frozen input/output hash checks, and the prior verification receipt. The fixed post-hoc rule emits a raw Qwen verdict only when multilingual NLI agrees. Across eleven supplied-evidence files (600 each), 295–393 verdicts per file are accepted; 1,169 correct Qwen answers are rejected in total. All-claim accuracy with abstentions as errors is 0.4827 versus raw Qwen 0.6598 over these dependent rows; selective accuracy on the accepted subset is not a general accuracy gain. The replay and source bindings are in `artifacts/multilingual_agreement_replay_v1`. Qwen and NLI used different text limits; no new neural inference, multilingual retrieval, trained gate transfer, calibrated probability or factual explanation validation occurred. The full charter remains open.

## Frozen XFEVER reliability audit (2026-10-03 UTC)

All 6600 previously received multilingual NLI probability vectors were checked against their predicted labels, the Qwen alignment keys and the two source manifests. Per-file ten-bin top-label ECE is .228–.267, and mean confidence .903–.964 exceeds observed accuracy .663–.727. Per-file multiclass Brier and reliability-bin counts are in `artifacts/xfever_reliability_audit_v1/summary.json`. This is an observed supplied-evidence reliability check on dependent translations of the same claims. No calibrator was fitted on test data, no new neural inference was performed, and APV-RAG correctness calibration remains open.

## Four-policy snapshot controller integration (2026-10-03 UTC)

The new opt-in `execute_bound_policies` entry point checks passage IDs, selected sentence indices and excerpt prefixes against the frozen benchmark corpus before evaluating gate features or invoking a response backend. It leaves `run_fresh_pipeline.py` and its completed run identity unchanged. Using exact saved responses, the integration replay reproduces 2400 original policy results across 600 contexts and refuses 299/360 saved stress contexts per policy before generation. This includes 60 empty-evidence cases; one swapped climate context remains bound because selected excerpts match. Gate probability comparisons allow only 1e-12 numerical roundoff; verdicts, prompts, requests and reasons must match exactly. `artifacts/bound_policy_integration_v1` records decisions, source hashes and policy counts. It uses no live neural inference, cannot verify publishers and does not establish an accuracy gain.

## Current reproducibility audit (2026-10-04, Asia/Shanghai)

An isolated temporary `pytest` 8.4.2 installation enabled the complete local test suite. The preflight receipt writer was corrected to include the standard `files` map while retaining its existing `preflight_sha256` field. Its saved preflight was regenerated from 840 explanations and 2520 passage pairs; no neural model was run. The project-wide audit passes 220 tests plus eight other non-neural commands, verifies 275 existing file references and lists 30 missing references from older export folders. All current core exports are present. The audit status is `checked_with_export_gaps`; it does not imply a clean neural installation or completion of the research charter. See `artifacts/reproducibility_audit_2026_10_04`.

## New FEVER input freeze (2026-10-04, Asia/Shanghai)

The official 19,998-claim FEVER labeled development JSONL was downloaded and fixed to SHA-256 `e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7`. An ID-hash rule selected 300 distinct claim texts after excluding all 585 unique claim IDs and matching texts from the previously observed 600 XFEVER English rows. The model input contains no labels; gold labels and evidence are stored separately for future scoring. Selected label counts are 110 support, 86 refute, 104 insufficient. The full Wikipedia archive has not been downloaded or hashed; no retrieval, model inference or independent result is claimed. See `docs/FEVER_EXTERNAL_EVALUATION.md`.

After freezing the new FEVER input, the full local suite passed 223 tests. The current audit was rerun with the temporary test dependency and keeps the 30 older export gaps explicit.


## Received multilingual retrieved-controller run (2026-10-05 Asia/Shanghai)

The 660-query/2,640-policy Windows export passes pinned identity, source/gold binding, context-bound feature validation, SQLite prompt/response equality, controller decisions and metric replay. The received save-recovery receipt preserves 422 checked English feature records; the completed export contains 660 records per model, 1,978 premise scores per model and 1,316 distinct Qwen responses. Results are in artifacts/multilingual_retrieved_pipeline_audit_v1/RESULTS.md. This closes the bounded closed-pool transfer experiment; it does not establish multilingual calibrated correctness confidence, full-page/open-web retrieval, historical authentication or explanation truth. The current consolidated status preserves these limits. No new human annotation, manuscript or GitHub push was performed.


## Grouped multilingual correctness-confidence analysis (2026-10-05 Asia/Shanghai)

Twenty policy/fold logistic fits use forty-eight observed claim IDs and exclude all eleven variants of the other twelve IDs. Every evaluation confidence is out of fold. Raw-score Brier/NLL/ECE decrease in all 44 file/policy cells. The Beta(1,1) development-prevalence control has lower pooled ECE than logistic scoring for all policies; logistic Brier/NLL improve over this control in 33 cells and ECE in 6. See artifacts/multilingual_correctness_calibration_v1. Original source outputs, answers, gate decisions and thresholds are unchanged. This closes the grouped exploratory confidence diagnostic, not independent new-claim validation or deployment calibration. No neural inference, new human annotation, manuscript or GitHub push was performed.

## New-claim multilingual confirmation: prepared, not executed

The checksum-verified official archive contains six larger test files with 11,710 paired rows and 9,418 claim IDs each. A label-blind selector freezes 100 unused IDs / 600 queries after excluding prior IDs, texts, source pages and excerpts and separating new claims by shared pages. Full-test deduplicated excerpt pools contain 3,884 texts per language. Four logistic correctness fits and four prevalence controls use only the earlier observed sixty-claim development run; their exact bytes are pinned. Selection, exclusion, retrieval and local development refitting are replayed. Synthetic tests cover controller/response/confidence replay and reject semantically corrupted confidence even after checksums are recomputed. They are software tests, not neural benchmark results.

Actual confirmation inference requires the user's existing Windows models. Run `scripts/run_multilingual_confirmation.py` to produce `multilingual_confirmation_outputs.zip`, then audit the uploaded export. No new confirmation accuracy or reliability result is available yet. The new non-English variants are machine translations; target-derived pools remain optimistic, and upstream model pretraining exposure is unknown. The whole charter, historical authentication and explanation truth remain open. Manuscript writing still requires user permission; no GitHub push is included.

## Received new-claim multilingual confirmation

The completed Windows export now passes exact frozen identity, exclusion/selection/retrieval replay, feature-context validation, SQLite response binding, controller decisions, frozen confidence application and metric replay: 600 queries, 2,400 policy rows, 600 entries and 1,800 passage scores per model, and 1,200 responses. There are 100 underlying IDs shared across six variants. Local audit does not rerun neural inference or independently verify tokenizer counts or explanation truth.

Logistic Brier/NLL/ECE improve over raw scores in 24/24 language/policy cells. Logistic Brier/NLL improve over development-prevalence controls in 21/24; ECE improves in 0/24. All four pooled prevalence ECE values are lower than logistic ECE. No gate raises all-query accuracy over common no-gate outputs. These mixed outcomes close the fixed machine-translated excerpt-pool confirmation; they do not establish policy superiority or general deployment calibration. Do not refit, select policies or repeat this experiment using the observed outcomes. Historical source authentication, factual explanation verification, full-page/open-web transfer and the full charter remain unfinished. No manuscript or GitHub push was performed.

## Fixed institutional source-origin admission

Three institution-hosted HTTPS pages are captured with exact URL policies, default certificate/hostname verification, identity anchors and saved raw HTML. The frozen audit replays their original metadata/navigation excerpts and whitespace controls, and refuses eight altered/missing-source controls per page before scoring. The guard integrates through a wrapper around the unchanged `source_pipeline.verify_claim`. Software tests cover redirects, transport settings, payload limits, script-only text, mixed pools and corrupted audit outputs. No neural benchmark or new human labels are used.

This closes the bounded origin/capture admission step, not historical authentication. The Politika catalogue does not authenticate the private newspaper image or its transcription; the museum's embedded timeline and linked guide resources are outside the capture. Unsigned receipts assume a trusted collector and frozen archive. Coordinated archive replacement, compromised publishers, author/date truth and semantic explanation truth remain outside the guarantee. See `docs/SOURCE_ORIGIN_PROTOCOL.md` and `artifacts/source_origin_audit_v1/RESULTS.md`. The full charter remains unfinished. No manuscript or GitHub push was performed.

## Exact cited-excerpt display layer

The deterministic display guard replays all 3,840 original/stress English policy records over 960 claim/condition contexts. It preserves upstream abstention and withholds candidates whose retrieved passages or collapsed context fail the existing snapshot rule. The admitted packets contain 5,637 exact cited excerpts with document IDs, selection indices and excerpt/document hashes. Every packet, summary count and report is reconstructed offline. Gold labels, model confidence and generated rationale text do not enter the renderer; the original source outputs and benchmark results are unchanged.

This completes literal citation traceability for the new display mode. It does not provide a certified natural-language explanation or establish entailment, relevance, completeness, source truth or evidence absence. Existing donor-context and OCR limitations remain. The earlier NLI diagnostic has not been turned into a truth gate. Semantic explanation truth and the full charter remain open. See `docs/EVIDENCE_DISPLAY_PROTOCOL.md` and `artifacts/evidence_display_audit_v1/RESULTS.md`. No neural inference, new human labels, manuscript or GitHub push was performed.


## Saved guard trade-off audit

All 3,840 saved policy records are paired with existing benchmark labels. The cited-evidence guard preserves original candidates but withholds correct and incorrect answers on altered contexts. The no-gate OCR control loses 28 correct and 32 incorrect answers among 60 cases; one swapped no-gate context still passes incorrectly. These descriptive, overlapping-context results do not establish general robustness or accuracy benefit. Original outputs remain unchanged. Run `scripts/analyze_guard_tradeoffs.py --verify` offline; see `artifacts/guard_tradeoffs_v1/RESULTS.md`. No neural inference, new human labels, manuscript or GitHub push is involved. The full charter remains unfinished.


## One-command current export checks

`scripts/check_current_release.py` runs pytest and five current export/status checks with the same interpreter and explicit project import paths. It saves per-check logs, installed package versions, open research requirements and a single `current_release_checks_outputs.zip`. Each run keeps separate logs; failed or timed-out checks are recorded and cause a nonzero exit. This is a current export check, not clean neural reproduction or completion of the original charter. See `docs/CURRENT_RELEASE_CHECK.md`. No neural inference, manuscript or GitHub push is performed.

## Exploratory archive-capture timing ablation (2026-10-05 UTC)

One fixed `C=1.0` Phase 2F augmentation adds five claim-date/Wayback-capture
availability and lag features. It fits only the original 2,458 training
records and compares with frozen predictions on the previously observed 609
internal-validation records. Macro-F1 drops from 0.5193 to 0.4796 and balanced
accuracy drops from 0.4971 to 0.4742. Of 32 changed decisions, 13 are fixed
and 17 become wrong. The earlier selected Phase 2F model remains unchanged.
Wayback capture is not a publication date or source authentication; no official
development record was read for this run. The result is exploratory, not a
fresh confirmation of provenance efficacy. See `docs/CAPTURE_TIMING_ABLATION.md`.

## Controlled multilingual pool mismatch (2026-10-05 UTC)

An optional source guard binds saved multilingual excerpts by exact ID and text
to one of six pinned file-specific XFEVER excerpt pools. The unmodified
controller is delegated to for an admitted context. All 600 saved original
contexts bind. A fixed same-claim next-language substitution is refused in all
600 cases before a generation request, across four policies. These 600
parallel-variant interventions are not independent test claims. The check uses
no gold labels or neural inference and does not alter any earlier output.
Because pool membership is a constructed identity check, this result does not
establish natural-language mismatch detection, publisher authentication,
open-web retrieval, or improved answer quality. A passage found identically in
two pools can be admitted for both. See `docs/MULTILINGUAL_POOL_MISMATCH.md`.

## Updated current release check (2026-10-05 UTC)

The one-command checker now includes the latest clean-confirmation export
receipts, capture-timing and pool-mismatch results, and frozen component
analysis. The status generator retains these stages when rebuilding its
machine-readable requirements and report. The included run passes all eight
checks and 350 tests. Received scoped confirmation reproduction is recorded;
full development training and source-index reconstruction remain unverified.
This fixes stale reporting and does not add a new accuracy evaluation.

## Trained-copy robustness experiment

Single-seed training augmentation completed on 2,458 training parents, with 609 previously observed internal validation claims. At 25 copies, control/augmented macro-F1 was 0.2581/0.5524; clean scores were 0.5373/0.5443. All 351 tests passed, fitted model replay and saved-result verification passed. No official development records were used. This does not close independent confirmation, source authentication or the full project charter. See TRAINED_COPY_ROBUSTNESS.md and PROJECT_FINISHING_PLAN.md.

## Fresh trained-copy component confirmation

Frozen 48 fresh Climate-FEVER claims in 46 article-connected groups after prior claim/page and near-text exclusions. The primary 25-copy macro-F1 difference was -0.0285, descriptive 95% group-bootstrap interval [-0.1053, 0.0539]. No superiority established; candidate not promoted or retuned. 353 tests passed. Model replay and independent metric recalculation passed. This supplied-evidence component check does not complete source authentication or full RAG evaluation. See FRESH_COPY_CONFIRMATION.md.

## Fresh fixed RAG preparation

Prepared the unchanged three-class climate RAG on all 46 compatible claims from the frozen 48-claim parent cohort; two disputed claims excluded for taxonomy compatibility. Input hashes, claim-only layout and BM25 retrieval checks passed for all 46. All 355 tests passed. Model weights are absent here; neural generation, explanation checks and outcome scoring are pending Windows execution. This is a fixed-method transfer check on already component-observed claims, not a blind independent dataset. See FRESH_CLIMATE_RAG.md.

## Received fresh fixed RAG result

All 46 received claims in 44 groups passed export, source-prefix/retrieval, literal prompt, numeric-guard, NLI-score and metric checks. Raw: 19/46 correct, macro-F1 0.3414. Guarded: 17/46 correct, 43 answers retained, macro-F1 0.2881; two correct and one incorrect raw answers rejected. Guard superiority not established. All 356 tests passed. Neural rerun/tokenizer replay and source authentication are not claimed. Full charter remains open. See artifacts/fresh_climate_rag_verification_v1/RESULTS.md.

## Fresh literal evidence output

Reused the existing source-snapshot and evidence-display modules on all 46 fresh RAG contexts. Produced 43 machine-candidate packets with 129 cited literal excerpts; three upstream abstentions preserved. All 172 constructed packet alterations rejected; no verdict changed or model called. Gold labels and free-form explanations were excluded from builder inputs. This establishes stored excerpt integrity only, not publisher authentication or factual explanation truth. See artifacts/fresh_rag_evidence_packets_v1/RESULTS.md.

## Post-hoc domain-collapse transfer mechanism

On all 48 observed supplied-evidence claims, applying the existing domain-collapse rule to the frozen plain control reduced false Supported decisions under 25 copies from 24 to 13 of 31 non-support cases. Overall accuracy fell from 31.25% to 27.08%; 80% of evidence sentences were removed. Shared Wikipedia domain is not evidence of source dependence. No refitting or neural inference; no promotion or independent-confirmation claim. Manual metrics and hashes passed. See artifacts/domain_collapse_transfer_v1/RESULTS.md.

## Optional exact-copy preprocessor

On 48 observed supplied-evidence claims and five copy counts, all 240 record-identity checks restored the clean records. Removed all 1,200 injected answers at 25 copies; preserved all 240 originals. Missing-source answers and different questions/text/metadata remain intact. Existing clean probabilities reused explicitly; no new inference or accuracy-gain claim. All 358 tests passed. Not activated in frozen pipelines. See docs/EXACT_COPY_REMOVAL.md.

## Copy-policy calibration and matched coverage

Descriptive evaluation of four saved-probability policies at 0 and 25 copies on 48 observed supplied-evidence cases. At 25 copies, Brier scores 1.0745/0.9796/0.9118/1.0141 all exceed the fixed uniform reference 0.7500. At 50% coverage, raw/copy-trained policies get 5/24 correct; domain/exact-copy policies get 6/24. No useful calibration/selective-accuracy benefit, new sufficiency head, threshold fitting or independent-confirmation claim. Independent metric replay passed; no model calls. See artifacts/copy_calibration_diagnostic_v1/RESULTS.md.


## Temperature calibration development
Two frozen classifiers calibrated using group-disjoint portions of internal validation. Brier worsened in all six checks. Not adopted; no independent confirmation claimed. See artifacts/copy_temperature_v1/RESULTS.md.


## Retrieval evidence recall diagnostic
Top3 decisive article hits 17/22; top10 19/22; top20 19/22. Exact decisive sentences present in top3 context 16/22 before clipping. All46 rankings match received run. No verdict improvement claimed.


## Top10 lexical sentence reranking
Not adopted: decisive article hits12/22 versus baseline17/22; exact decisive sentence hits12/22 versus16/22. Same three-article budget. All360tests passed. No neural verdict improvement claimed.


## Semantic reranking verified
Uploaded46claim run verified; decisive article hits7/22 versus baseline17/22. Ranking policy not adopted. Neural scores not recomputed locally.


## DirectQuote existing-label component evaluation
100frozen cases:44correct;33/50known speakers correct;39/50unknown cases falsely attributed. Always-UNKNOWN balanced reference50%. No reliable attribution/abstention gain established. Export/code/protocol verified; neural generation not replayed.


## Trained quote relation baseline
Train3569/development875/test798, quote-shingle group split. Test671/798correct;known550/645;unknown false32/153. Original100observed cases excluded. All798saved predictions replayed;362tests passed. Filtered component only, article IDs unavailable.
