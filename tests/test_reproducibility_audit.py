"""Prevent receipt omissions and corrupt files from being reported as verified."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def auditor():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_project_reproducibility.py'
    assert path.exists(), 'The reproducibility auditor has not been implemented'
    return importlib.import_module('scripts.audit_project_reproducibility')


def test_available_file_is_verified_without_rewriting_it(tmp_path):
    data = b'Original result bytes\r\n'
    artifact = tmp_path / 'predictions.json'
    artifact.write_bytes(data)
    receipt = tmp_path / 'output_manifest.json'
    receipt.write_text(json.dumps({'predictions.json': hashlib.sha256(data).hexdigest()}))
    report = auditor().audit_receipt(receipt, tmp_path)
    assert report['files'][0]['status'] == 'verified'
    assert artifact.read_bytes() == data


def test_omitted_file_is_missing_not_verified(tmp_path):
    receipt = tmp_path / 'output_manifest.json'
    receipt.write_text(json.dumps({'cache.npy': 'a' * 64}))
    report = auditor().audit_receipt(receipt, tmp_path)
    assert report['files'][0]['status'] == 'missing'
    assert not (tmp_path / 'cache.npy').exists()


def test_changed_file_is_a_checksum_mismatch(tmp_path):
    (tmp_path / 'result.json').write_bytes(b'changed')
    receipt = tmp_path / 'audit_manifest.json'
    receipt.write_text(json.dumps({'files': {'result.json': hashlib.sha256(b'original').hexdigest()}}))
    report = auditor().audit_receipt(receipt, tmp_path)
    assert report['files'][0]['status'] == 'checksum_mismatch'


@pytest.mark.parametrize('name', ['../outside.json', '/absolute.json', 'C:/private.json', r'..\outside.json'])
def test_receipt_cannot_read_outside_its_folder(tmp_path, name):
    receipt = tmp_path / 'output_manifest.json'
    receipt.write_text(json.dumps({name: 'a' * 64}))
    report = auditor().audit_receipt(receipt, tmp_path)
    assert report['files'][0]['status'] == 'invalid_path'


def test_invalid_digest_is_reported(tmp_path):
    receipt = tmp_path / 'output_manifest.json'
    receipt.write_text(json.dumps({'result.json': 'not a digest'}))
    report = auditor().audit_receipt(receipt, tmp_path)
    assert report['files'][0]['status'] == 'invalid_digest'
