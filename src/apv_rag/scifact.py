"""Target label mapping and document-connected SciFact evaluation groups."""

import re
import unicodedata


def normalized_claim(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def target_label(record):
    labels = {r["label"] for rationales in record["evidence"].values() for r in rationales}
    if not labels:
        return "Not Enough Evidence"
    if not labels <= {"SUPPORT", "CONTRADICT"}:
        raise ValueError("Unknown SciFact label")
    if len(labels) > 1:
        return None
    return {"SUPPORT": "Supported", "CONTRADICT": "Refuted"}[next(iter(labels))]


def connected_groups(records):
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, record in enumerate(records):
        for doc in record["cited_doc_ids"]:
            if doc in owners:
                parent[find(i)] = find(owners[doc])
            owners[doc] = i
    return [find(i) for i in range(len(records))]
