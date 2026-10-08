"""Restoring aliases must preserve source bytes and refuse corrupt data."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def restorer():
    path = Path(__file__).resolve().parents[1] / 'scripts/restore_received_exports.py'
    assert path.exists(), 'The checked export recovery command is not implemented'
    return importlib.import_module('scripts.restore_received_exports')


def example(root, name='input_manifest.json', content=b'{"original":true}\r\n'):
    target = root / 'artifacts/example_received' / name
    source = root / 'artifacts/original_run' / name
    source.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = target.parent / 'output_manifest.json'
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    manifest.write_text(json.dumps({**old, name: digest}))
    return target, source, {'target': target.relative_to(root).as_posix(),
        'matching_original_paths': [source.relative_to(root).as_posix()], 'sha256': digest}


def test_restore_preserves_original_bytes_and_receipt(tmp_path):
    target, source, entry = example(tmp_path)
    receipt = (target.parent / 'output_manifest.json').read_bytes()
    original = source.read_bytes()
    report = restorer().restore_exports(tmp_path, [entry])
    assert report['copied'] == 1
    assert target.read_bytes() == source.read_bytes() == original
    assert (target.parent / 'output_manifest.json').read_bytes() == receipt


def test_corrupt_source_prevents_any_copies(tmp_path):
    first, _, a = example(tmp_path)
    second, source, b = example(tmp_path, 'features.npy', b'expected array bytes')
    source.write_bytes(b'corrupt array bytes')
    with pytest.raises(ValueError):
        restorer().restore_exports(tmp_path, [a, b])
    assert not first.exists() and not second.exists()


def test_existing_wrong_target_is_not_overwritten(tmp_path):
    target, _, entry = example(tmp_path)
    target.write_bytes(b'keep this wrong file for investigation')
    with pytest.raises(ValueError):
        restorer().restore_exports(tmp_path, [entry])
    assert target.read_bytes() == b'keep this wrong file for investigation'


def test_map_cannot_override_expected_receipt_hash(tmp_path):
    target, _, entry = example(tmp_path)
    entry['sha256'] = 'a' * 64
    with pytest.raises(ValueError):
        restorer().restore_exports(tmp_path, [entry])
    assert not target.exists()
