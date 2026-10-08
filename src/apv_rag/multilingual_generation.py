"""Label-independent explanation cohort and multilingual supplied-evidence prompts."""
import hashlib

from apv_rag.generative_rag import LABELS

LANGUAGES = {"en": "English", "es": "Spanish", "fr": "French", "id": "Indonesian",
             "ja": "Japanese", "zh": "Chinese"}


def explanation_ids(rows, count=60):
    ids = {str(r["id"]) for r in rows}
    if count <= 0 or len(ids) < count:
        raise ValueError("Insufficient unique claim IDs for the frozen explanation cohort")
    return sorted(ids, key=lambda i: hashlib.sha256(f"APV-multilingual-generation-v1:{i}".encode()).hexdigest())[:count]


def verdict_question(claim, evidence):
    return (f"Evidence: [premise] {evidence}\nClaim: {claim}\n"
            "Reply only Supported, Refuted, or Not Enough Evidence.")


def explanation_question(claim, evidence, verdict, language):
    if verdict not in LABELS or language not in LANGUAGES:
        raise ValueError("Unknown verdict or explanation language")
    return (f"Evidence: [premise] {evidence}\nClaim: {claim}\nVerdict: {verdict}\n"
            f"Explain in {LANGUAGES[language]} why this evidence supports, contradicts, or cannot "
            "establish the claim. Use only the supplied evidence and keep the explanation short.")
