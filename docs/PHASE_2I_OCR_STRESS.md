# Phase 2I: controlled OCR-corruption testing

Phase 2I measures verdict stability when evidence-answer text is corrupted at
fixed rates of 0%, 5%, 10%, and 20%. Claims, labels, generated questions,
source URLs, and split membership remain unchanged. A seeded transformation
replaces an exact rounded fraction of alphanumeric evidence characters with
common OCR confusions or a visible corruption marker.

Word, character, and hybrid TF-IDF verifiers use the same frozen training
records, `C=1.0`, balanced logistic regression, five-fold training-only sigmoid
calibration, and seed 369. The experiment reports Macro-F1 change,
prediction-flip rate, confidence change, and 2,000 paired-bootstrap intervals.

The official AVeriTeC development set remains sealed. Tesla material is not
used for training, model selection, or quantitative accuracy claims.

Run from PowerShell in `D:\apv-rag-tesla369`:

```powershell
.\.venv\Scripts\python.exe scripts\run_ocr_stress.py
.\.venv\Scripts\python.exe scripts\validate_ocr_stress.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Frozen result

Run `run_20261001T070531Z` used 2,458 training and 609 internal-validation
records. It used zero official-development records.

| Representation | Clean Macro-F1 | 5% OCR | 10% OCR | 20% OCR | 20% change | 20% flip rate |
|---|---:|---:|---:|---:|---:|---:|
| Word | 0.4663 | 0.4579 | 0.4372 | 0.4182 | -0.0481 | 0.1051 |
| Character | 0.4610 | 0.4536 | 0.4388 | 0.4116 | -0.0494 | **0.0657** |
| Hybrid | **0.4840** | **0.4759** | **0.4635** | **0.4230** | -0.0611 | 0.1018 |

At 20% OCR corruption, the paired-bootstrap 95% intervals for the Macro-F1
change were [-0.0941, -0.0030] for word features, [-0.0857, -0.0161] for
character features, and [-0.0962, -0.0281] for hybrid features. All three
intervals exclude zero at that rate. The 5% and 10% intervals include zero.

The hybrid model retained the highest absolute Macro-F1 at every tested rate,
but its 20% loss was the largest. The character model had the lowest
prediction-flip rate, suggesting greater verdict stability under this exact
character-substitution process. These findings concern synthetic evidence-text
corruption and should not be generalized to every OCR engine or document type.

These are internal-validation robustness measurements, not final test-set
results. The official development split remains sealed.
