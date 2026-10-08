"""Explicit NLI decisions and checked generated summaries; no generative verdict claim."""

import numpy as np

from apv_rag.multilingual_nli import normalize_model_scores


def evidence_decision(scores):
    if not scores:
        return {"label": "Not Enough Evidence", "selected": None, "reason": "no_evidence"}
    values = np.asarray(normalize_model_scores(scores))
    options = []
    for i, row in enumerate(values):
        stance = int(row.argmax())
        margin = float(np.sort(row)[-1] - np.sort(row)[-2])
        if stance in (0, 1) and row[stance] >= 0.6 and margin >= 0.15:
            options.append((float(row[stance]), i, stance))
    if not options:
        return {"label": "Not Enough Evidence", "selected": None, "reason": "no_clear_stance"}
    if len({stance for _, _, stance in options}) > 1:
        return {"label": "Not Enough Evidence", "selected": None, "reason": "conflicting_stance"}
    _, index, stance = max(options, key=lambda x: (x[0], -x[1]))
    return {
        "label": "Refuted" if stance == 0 else "Supported",
        "selected": index,
        "reason": "nli_candidate",
    }


def explanation_passes(text, scores):
    if not text.strip():
        return False
    row = normalize_model_scores([scores])[0]
    return row[1] >= 0.7 and row[1] - max(row[0], row[2]) >= 0.2
