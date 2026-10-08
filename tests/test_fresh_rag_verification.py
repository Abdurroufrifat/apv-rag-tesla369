from hashlib import sha256
import pytest
from verify_fresh_climate_rag import verify_file_hashes


def test_received_hash_check_rejects_modified_predictions(tmp_path):
    file=tmp_path/'predictions.json';file.write_bytes(b'[]')
    expected={'predictions.json':sha256(b'[]').hexdigest()}
    verify_file_hashes(tmp_path,expected)
    file.write_bytes(b'[{}]')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        verify_file_hashes(tmp_path,expected)
