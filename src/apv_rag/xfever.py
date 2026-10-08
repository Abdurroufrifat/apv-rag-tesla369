"""Validate aligned multilingual XFEVER claim-evidence evaluation rows."""

LABEL_MAP = {
    "SUPPORTS": "Supported",
    "REFUTES": "Refuted",
    "NOT ENOUGH INFO": "Not Enough Evidence",
}


def validate_parallel(sets):
    reference = sets["en/test.6h.jsonl"]
    expected = [(r["id"], r["label"]) for r in reference]
    if not reference:
        raise ValueError("Empty reference set")
    for name, rows in sets.items():
        if [(r["id"], r["label"]) for r in rows] != expected:
            raise ValueError(f"Parallel alignment mismatch: {name}")
        for r in rows:
            if r["label"] not in LABEL_MAP:
                raise ValueError("Unknown target label")
            if not all(
                isinstance(r[k], str) and r[k].strip() for k in ("claim", "evidence", "page")
            ):
                raise ValueError("Missing paired text or page")
    return len(reference)
