"""Conservative lexical checks; these do not establish semantic support."""

import re


def new_numeric_values(explanation, evidence):
    def numbers(text):
        text = re.sub(r"\[[^\]]*\]", "", text)
        return set(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)", text))

    return sorted(numbers(explanation) - numbers(" ".join(d["text"] for d in evidence)))
