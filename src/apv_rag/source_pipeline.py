"""Auditable lexical retrieval and heuristic provenance-aware NLI aggregation."""

from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import numpy as np

from apv_rag.direct_nli import TARGET_LABELS, direct_probabilities
from apv_rag.retrieval import BM25Index
from apv_rag.scifact import normalized_claim


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int = 5
    retrieval_pool: int = 50
    minimum_families: int = 2
    minimum_confidence: float = 0.6
    minimum_margin: float = 0.15
    collapse_families: bool = True
    weight_sources: bool = True
    require_primary: bool = False

    def validate(self):
        if self.top_k < 1 or self.retrieval_pool < self.top_k or self.minimum_families < 1:
            raise ValueError("Invalid retrieval or family-count setting")
        if not 0 <= self.minimum_confidence <= 1 or not 0 <= self.minimum_margin <= 1:
            raise ValueError("Invalid decision thresholds")


def verify_claim(claim, claim_language, documents, scorer, config=None, *, retrieval_index=None):
    """Scorer receives (claim, list of texts), returns C/E/N probability rows.

    Source ranks and families are declared metadata, not authenticated facts.
    Language tags must be supplied; no automatic language detector is claimed.
    """
    config = config or PipelineConfig()
    config.validate()
    if not isinstance(claim, str) or not claim.strip() or not claim_language:
        raise ValueError("Claim and declared language required")
    ids = [d["id"] for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document identifiers")
    for d in documents:
        if not isinstance(d["text"], str) or not d["text"].strip():
            raise ValueError("Empty document text")
        if not d.get("language"):
            raise ValueError("Declared document language required")
        if type(d.get("source_rank", 5)) is not int or not 1 <= d.get("source_rank", 5) <= 5:
            raise ValueError("Source rank must be an integer from one to five")
        if type(d.get("is_primary", False)) is not bool:
            raise ValueError("Primary-source flag must be boolean")
    index = retrieval_index or BM25Index([d["text"] for d in documents])
    if len(index.lengths) != len(documents):
        raise ValueError("Retrieval index document count mismatch")
    ranked = index.search(claim, config.retrieval_pool)
    eligible, rejected = [], []
    for position, score in ranked:
        d = documents[position]
        if d["language"].casefold() != claim_language.casefold():
            rejected.append({"id": d["id"], "reason": "declared_language_mismatch"})
        else:
            eligible.append((d, score))
    # Collapse transitive dependencies: shared declared families OR exact normalized text.
    parent = list(range(len(eligible)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, (d, _) in enumerate(eligible):
        keys = [("text", normalized_claim(d["text"]))]
        if d.get("provenance_family_id"):
            keys.append(("family", d["provenance_family_id"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    selected, used = [], set()
    for i, (d, score) in enumerate(eligible):
        family = find(i)
        if config.collapse_families and family in used:
            rejected.append({"id": d["id"], "reason": "dependent_copy"})
            continue
        if len(selected) >= config.top_k:
            continue
        used.add(family)
        selected.append((d, score, family))
    provenance_count = len({family for _, _, family in selected})
    trace = [
        {
            "id": d["id"],
            "source_url": d.get("source_url", ""),
            "host": urlparse(d.get("source_url", "")).hostname,
            "retrieval_score": score,
            "family_component": family,
            "source_rank": d.get("source_rank", 5),
            "is_primary_declared": d.get("is_primary", False),
        }
        for d, score, family in selected
    ]
    reasons = []
    if not selected:
        reasons.append("no_compatible_retrieved_evidence")
        probabilities = np.array([0.0, 0.0, 1.0])
    else:
        scores = scorer(claim, [d["text"] for d, _, _ in selected])
        if len(scores) != len(selected):
            raise ValueError("NLI rows do not align with retrieved evidence")
        # Validate each row and remap to Supported/Refuted/NEI.
        mapped = np.array([direct_probabilities([row]) for row in scores])
        weights = np.array(
            [1 / d.get("source_rank", 5) if config.weight_sources else 1.0 for d, _, _ in selected]
        )
        probabilities = np.average(mapped, axis=0, weights=weights)
        for row, nli, weight in zip(trace, scores, weights, strict=True):
            row["nli_probabilities"] = list(nli)
            row["aggregation_weight"] = float(weight)
    if provenance_count < config.minimum_families:
        reasons.append("too_few_provenance_components")
    if config.require_primary and not any(d.get("is_primary", False) for d, _, _ in selected):
        reasons.append("missing_declared_primary_source")
    order = np.sort(probabilities)
    if probabilities.max() < config.minimum_confidence:
        reasons.append("low_heuristic_confidence")
    if order[-1] - order[-2] < config.minimum_margin:
        reasons.append("ambiguous_stance")
    label = TARGET_LABELS[int(probabilities.argmax())]
    if label == "Not Enough Evidence":
        reasons.append("neutral_dominant")
    return {
        "status": "abstain" if reasons else "machine_candidate",
        "candidate_label": None if reasons else label,
        "stance_scores": probabilities.tolist(),
        "abstention_reasons": reasons,
        "selected_evidence": trace,
        "rejected_evidence": rejected,
        "provenance_components": provenance_count,
        "config": asdict(config),
        "qualification": (
            "heuristic scores; declared provenance; no gold verdict or source authentication"
        ),
    }
