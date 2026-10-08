# SciFact transfer: preparation and fixed design

This stage downloads and seals the official SciFact corpus and labeled development file. It does not run transfer evaluation. The upstream download URL is mutable, so the manifest pins the exact first downloaded bytes; this is not a verified upstream release checksum.

Run in the existing Windows project:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\prepare_scifact.py
```

Provide `data\external\scifact\sealed_v1\manifest.json` after preparation. Existing sealed files are checked rather than replaced. No SciFact training data is downloaded.

## Design to implement before scoring

Compare the seven-feature and eighteen-feature classifiers trained on the current AVeriTeC training partition. Keep all five seeds, calibration, regularization, and exponent 1 fixed. Use BM25 over the entire SciFact abstract corpus and select five abstracts, without consulting rationale annotations or cited document identifiers. Each abstract is a premise; do not retrieve from gold rationale sentences.

Map SUPPORT to Supported, CONTRADICT to Refuted, and empty evidence to Not Enough Evidence. Claims with conflicting annotation labels are excluded with an explicit count, rather than assigned an invented label. Retain the original four-class classifier; predictions of Conflicting Evidence/Cherrypicking count as errors on the three-label target. Report target three-class macro-F1, accuracy, per-class recall, and the rate of out-of-target predictions. Do not silently renormalize away the fourth class.

Exclude exact normalized claim matches with any previously observed AVeriTeC training, internal validation, or official development claim. Report the exclusions. Build connected SciFact claim groups sharing cited documents for bootstrap and paired randomization only; cited document identifiers must not influence retrieval or prediction. Use marginal 95% group bootstrap intervals with 2,000 draws and 10,000 group swaps, seed 369. Compare expanded versus seven-feature performance for argmax and exponent 1; Holm correction covers these two planned comparisons. No SciFact labels may be used for fitting, threshold choice, or selecting a winning configuration.

This evaluates transfer with automatic retrieval from abstracts, not the official joint evidence-and-label SciFact metric. Exact matching cannot establish absence of semantic overlap or pretrained-model contamination. The retained four-class task and abstract premises differ from the source task and must be discussed. A successful preparation command is not a completed benchmark result.

Official source and schema:
https://github.com/allenai/scifact
https://github.com/allenai/scifact/blob/master/doc/data.md
