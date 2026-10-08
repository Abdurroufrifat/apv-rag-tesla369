"""Track number provenance across both supplied inputs; not factual entailment."""

import re


def numeric_provenance(explanation, claim, evidence):
    def numbers(text):
        text = re.sub(r"\[[^\]]*\]", "", text)
        return set(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)", text))

    used = numbers(explanation)
    source = numbers(" ".join(d["text"] for d in evidence))
    claim_values = numbers(claim)
    return {
        "evidence_values": sorted(used & source),
        "claim_only_values": sorted((used & claim_values) - source),
        "absent_from_inputs": sorted(used - source - claim_values),
    }
