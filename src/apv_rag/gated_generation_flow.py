"""Apply an experimental context gate before requesting any generation."""
from apv_rag.generative_rag import LABELS
from apv_rag.integrated_gate import apply_gate, collapse_context
from apv_rag.numeric_integrity_v2 import numeric_provenance_v2


def execute_generation(shown_claim, evidence, generate, gate_probability=None, threshold=.5):
    """None selects the no-gate control; generate returns answer/prompt/token count."""
    evidence = collapse_context(evidence)
    row = {
        "evidence": evidence, "gate_probability": gate_probability, "threshold": threshold,
        "generation_requests": [], "stages": ["collapse_context", "gate"], "reasons": [],
        "verdict_prompt": None, "explanation_prompt": None,
        "verdict_prompt_tokens": None, "explanation_prompt_tokens": None,
        "generated_verdict": None, "generated_explanation": None,
        "raw_candidate_label": None, "candidate_label": None, "numeric_provenance": None,
    }
    if not evidence:
        row["reasons"].append("no_evidence")
        return row
    if gate_probability is not None and apply_gate(True, [], gate_probability, threshold) is None:
        row["reasons"].append("gate_below_threshold")
        return row
    context = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)
    question = f"Evidence: {context}\nClaim: {shown_claim}\nReply only Supported, Refuted, or Not Enough Evidence."
    row["generation_requests"].append("verdict")
    row["stages"].append("verdict")
    response = generate("verdict", question)
    verdict = response["answer"]
    row.update(generated_verdict=verdict, verdict_prompt=response["prompt"], verdict_prompt_tokens=response["prompt_tokens"])
    if verdict not in LABELS:
        row["reasons"].append("invalid_verdict")
        return row
    row["raw_candidate_label"] = verdict
    question = (
        f"Evidence: {context}\nClaim: {shown_claim}\nVerdict: {verdict}\n"
        "Explain why this evidence supports, contradicts, or cannot establish "
        "the claim. Use only the supplied evidence and keep the explanation short."
    )
    row["generation_requests"].append("explanation")
    row["stages"].append("explanation")
    response = generate("explanation", question)
    explanation = response["answer"]
    row.update(generated_explanation=explanation, explanation_prompt=response["prompt"],
               explanation_prompt_tokens=response["prompt_tokens"])
    if not explanation:
        row["reasons"].append("empty_explanation")
    row["stages"].append("numeric_check")
    row["numeric_provenance"] = numeric_provenance_v2(explanation or "", shown_claim, evidence)
    if row["numeric_provenance"]["absent_from_inputs"]:
        row["reasons"].append("numeric_value_absent")
    if not row["reasons"]:
        row["candidate_label"] = verdict
    return row
