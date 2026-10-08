"""Optional excerpt-pool binding for the fixed multilingual controller.

This checks declared benchmark pool membership, not natural-language identity,
publisher provenance, evidence relevance, or factual correctness.
"""

from __future__ import annotations

from collections.abc import Mapping

from apv_rag.fresh_pipeline import POLICIES
from apv_rag.gated_generation_flow import execute_generation
from apv_rag.integrated_gate import collapse_context
from apv_rag.multilingual_pipeline import execute


def build_source_index(corpora: Mapping) -> dict[str, dict[str, str]]:
    """Index pinned (ID, exact excerpt) pairs separately for each input file."""
    result = {}
    for file, entries in corpora.items():
        if not isinstance(file, str) or not isinstance(entries, list):
            raise ValueError("Expected separately named excerpt pools")
        rows = {}
        for row in entries:
            if (not isinstance(row, Mapping) or not isinstance(row.get("id"), str)
                    or not isinstance(row.get("text"), str) or not row["text"].strip()
                    or row["id"] in rows):
                raise ValueError("Duplicate or invalid excerpt-pool member")
            rows[row["id"]] = row["text"]
        result[file] = rows
    return result


def admit_context(source: Mapping, prepared: Mapping, corpora: Mapping, *, indexed=False) -> bool:
    """Bind prepared passages to the source record's declared language pool."""
    try:
        file = source["file"]
        if (not isinstance(file, str) or source["language"] != file.split("/")[0]
                or source["claim"] != prepared["claim"]
                or not isinstance(prepared["shown_claim"], str)
                or not prepared["shown_claim"].strip()
                or prepared["evidence"] != collapse_context(prepared["retrieved_evidence"])):
            return False
        pools = corpora if indexed else build_source_index(corpora)
        pool = pools[file]
        raw = source["retrieved_evidence"]
        shown = prepared["retrieved_evidence"]
        if len(raw) != len(shown) or not shown or len({d["id"] for d in shown}) != len(shown):
            return False
        for original, passage in zip(raw, shown, strict=True):
            if (original["id"] != passage["id"]
                    or original["score"] != passage["score"]
                    or pool.get(original["id"]) != original["text"]
                    or not isinstance(passage["text"], str)
                    or not passage["text"]
                    or not original["text"].startswith(passage["text"])):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def execute_language_bound(source, prepared, corpora, english, multilingual, heads,
                           generate, *, indexed=False):
    """Reject an unbound pool before using NLI scores or calling a generator."""
    admitted = admit_context(source, prepared, corpora, indexed=indexed)
    if admitted:
        rows = execute(prepared, english, multilingual, heads, generate)
    else:
        rows = []
        for policy in POLICIES:
            row = execute_generation(prepared["shown_claim"], [], generate)
            row["reasons"] = ["source_language_pool_mismatch"]
            row["policy"] = policy
            rows.append(row)
    for row in rows:
        row["source_language_pool_bound"] = admitted
    return rows
