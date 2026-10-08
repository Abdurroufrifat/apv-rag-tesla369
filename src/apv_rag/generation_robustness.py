"""Synthetic exact-copy and source-order controls for frozen generation contexts."""
import hashlib

from apv_rag.generative_rag import LABELS
from apv_rag.integrated_gate import collapse_context

CONDITIONS = ("baseline", "copies_raw", "copies_collapsed", "reverse_order")


def context_variants(evidence):
    if not evidence:
        raise ValueError("Empty evidence context")
    baseline = [dict(e) for e in evidence]
    first = baseline[0]
    copies = [dict(first, id=f"synthetic_copy_{i + 1}_of_{first['id']}",
                   family_id=first.get("family_id", first["id"])) for i in range(3)]
    repeated = baseline + copies
    if len({str(e["id"]) for e in repeated}) != len(repeated):
        raise ValueError("Copy IDs collide with original sources")
    collapsed = collapse_context(repeated)
    if collapsed != baseline:
        raise ValueError("Original context contains dependent copies; baseline would change")
    return {"baseline": baseline, "copies_raw": repeated, "copies_collapsed": collapsed,
            "reverse_order": list(reversed(baseline))}


def explanation_ids(rows, cohort, count=60):
    ids = {str(r["claim_id"]) for r in rows}
    if count <= 0 or len(ids) < count:
        raise ValueError("Insufficient unique claims for explanation selection")
    return sorted(ids, key=lambda i: hashlib.sha256(
        f"APV-generation-robustness-v1:{cohort}:{i}".encode()).hexdigest())[:count]


def question(claim, evidence, verdict=None):
    context = "\n".join(f"[{r['id']}] {r['text']}" for r in evidence)
    prefix = f"Evidence: {context}\nClaim: {claim}\n"
    if verdict is None:
        return prefix + "Reply only Supported, Refuted, or Not Enough Evidence."
    if verdict not in LABELS:
        raise ValueError("Invalid verdict for explanation")
    return (prefix + f"Verdict: {verdict}\nExplain why this evidence supports, contradicts, or cannot establish "
            "the claim. Use only the supplied evidence and keep the explanation short.")
