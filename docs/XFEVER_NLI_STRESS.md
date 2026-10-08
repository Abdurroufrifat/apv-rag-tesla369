# Fixed XFEVER English-model transfer stress

Use all 600 aligned rows in each of eleven official files: English, and upstream machine and human translations for Spanish, French, Indonesian, Japanese, and Chinese. The 585 distinct claim IDs include repeated evidence rows. Preserve every row; aggregate uncertainty using connected components sharing English claim ID or English page. No target training or model selection is permitted.

Use the existing pinned English NLI model, supplied claim/evidence pairs, a 256-token pair limit, direct contradiction/entailment/neutral mapping, and deterministic argmax. This is not a multilingual-model capability claim or a retrieval evaluation. Neutral is a heuristic NEI mapping. Human translations were provided by the benchmark creators; no new human review is requested.

Report three-class macro-F1, accuracy, Brier, per-class metrics, and disagreement with English. After prediction verification, run marginal 95% grouped bootstrap intervals (2,000 draws) and grouped paired randomization (10,000 swaps), random seed 369, for each translated file minus English. Holm correction spans all ten comparisons. These settings are recorded before the model outputs are observed.

Run from the existing Windows project:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\run_xfever_nli_stress.py
```

The script uses local model files, prints progress, and resumes pair inference through a checksum-bound cache. It rejects changed inputs and completed-run overwrites. Upload `stress_summary.json`, `predictions.json`, `input_manifest.json`, and `output_manifest.json` from `artifacts\xfever_nli_stress`. No new model download is needed. A multilingual-model control and the full generative retrieval study remain open. Manuscript work remains paused.
