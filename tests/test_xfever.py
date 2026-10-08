import pytest

from apv_rag.xfever import validate_parallel


def row(identifier=1):
    return dict(id=identifier, label="SUPPORTS", claim="claim", evidence="evidence", page="page")


def test_preserves_repeated_claim_rows_and_requires_alignment():
    assert (
        validate_parallel({"en/test.6h.jsonl": [row(), row()], "es/test.6h.jsonl": [row(), row()]})
        == 2
    )
    with pytest.raises(ValueError):
        validate_parallel({"en/test.6h.jsonl": [row()], "es/test.6h.jsonl": [row(2)]})
    with pytest.raises(ValueError):
        validate_parallel({"en/test.6h.jsonl": [row() | {"evidence": ""}]})
