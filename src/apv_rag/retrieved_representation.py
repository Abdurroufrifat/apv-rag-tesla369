"""Construct expanded features from audited retrieved document positions."""

from apv_rag.evidence_representation import expanded_features


def audit_features(claim, documents, audit):
    positions = audit["document_ids"]
    if len(set(positions)) != len(positions) or any(
        not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(documents)
        for i in positions
    ):
        raise ValueError("invalid retrieved document positions")
    selected = [documents[i] for i in positions]
    return expanded_features(
        claim,
        [d["text"] for d in selected],
        [d["source_url"] for d in selected],
        audit["nli_probabilities"],
    )
