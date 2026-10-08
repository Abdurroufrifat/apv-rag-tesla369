"""Separate constrained verdict generation from free-form explanations."""

from apv_rag.generative_rag import LABELS


def allowed_next_tokens(sequences, prefix, eos_token_id):
    """Allow only continuations of complete label token sequences."""
    prefix = list(prefix)
    allowed = set()
    for sequence in sequences:
        sequence = list(sequence)
        if sequence[: len(prefix)] == prefix:
            allowed.add(sequence[len(prefix)] if len(prefix) < len(sequence) else eos_token_id)
    if not allowed:
        raise ValueError("Decoder prefix is outside allowed verdict sequences")
    return sorted(allowed)


def verdict_prompt(claim, evidence):
    passages = "\n".join(f"[{d['id']}] {d['text']}" for d in evidence)
    return (
        f"Evidence: {passages or '(none)'}\nClaim: {claim}\n"
        "Does the evidence support the claim, contradict it, or provide insufficient information? "
        "Answer with Supported, Refuted, or Not Enough Evidence."
    )


def generate_answer(model, tokenizer, torch, claim, evidence):
    prompt = verdict_prompt(claim, evidence)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    if inputs.input_ids.shape[1] > 512:
        raise ValueError("Verdict prompt exceeds input budget")
    sequences = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]

    def allowed(batch_id, tokens):
        # T5 decoder starts with its configured decoder-start token.
        prefix = tokens.tolist()
        if prefix[0] != model.config.decoder_start_token_id:
            raise ValueError("Unexpected decoder start token")
        return allowed_next_tokens(sequences, prefix[1:], tokenizer.eos_token_id)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max(map(len, sequences)) + 1,
            prefix_allowed_tokens_fn=allowed,
        )
    label = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if label not in LABELS:
        raise ValueError("Constrained verdict decoder returned an invalid label")
    passages = " ".join(f"[{d['id']}] {d['text']}" for d in evidence)
    explanation_prompt = (
        f"Evidence: {passages}\n"
        f"Claim: {claim}\nVerdict: {label}\n"
        "Explain this verdict in one sentence using only the evidence. "
        "Cite the relevant evidence ID in square brackets. "
        "If information is insufficient, explain what evidence is missing."
    )
    inputs = tokenizer(explanation_prompt, return_tensors="pt", truncation=False)
    if inputs.input_ids.shape[1] > 512:
        raise ValueError("Explanation prompt exceeds input budget")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=96, do_sample=False, num_beams=1)
    explanation = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    return {
        "verdict_prompt": prompt,
        "explanation_prompt": explanation_prompt,
        "generated_verdict": label,
        "generated_explanation": explanation,
        "assembled_answer": f"{label} | {explanation}",
    }
