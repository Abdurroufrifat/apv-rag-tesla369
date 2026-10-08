import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from run_clean_confirmation import digest
from run_clean_multilingual_confirmation import check_receipt, find_environment, invoke_fresh


def test_receipt_checks_actual_bytes_and_refuses_changed_result(tmp_path):
    result = tmp_path / 'comparison.json'
    result.write_text('{"status":"reproduced"}')
    (tmp_path / 'audit_manifest.json').write_text(json.dumps({'files': {'comparison.json': digest(result)}}))
    check_receipt(tmp_path)
    result.write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match='receipt'):
        check_receipt(tmp_path)


@pytest.mark.parametrize('name', ['../comparison.json', '/comparison.json', 'C:/comparison.json', 'a\\b.json'])
def test_receipt_refuses_escaping_paths(tmp_path, name):
    (tmp_path / 'audit_manifest.json').write_text(json.dumps({'files': {name: '0' * 64}}))
    with pytest.raises(ValueError):
        check_receipt(tmp_path)


def test_missing_environment_gives_actionable_error(tmp_path):
    with pytest.raises(ValueError, match='clean English'):
        find_environment(tmp_path)


def test_environment_lookup_uses_completed_run_with_valid_receipts(tmp_path):
    run = tmp_path / 'artifacts/clean_confirmation_runs/run_001'
    child = run / 'environment/Scripts/python.exe'
    child.parent.mkdir(parents=True)
    child.write_bytes(b'test executable, never run')
    (run / 'comparison.json').write_text(json.dumps({'status': 'reproduced', 'difference_count': 0,
         'neural_inference_rerun': True, 'claims': 300, 'policy_records': 1200,
         'prior_feature_or_answer_cache_imported': False}))
    (run / 'setup_status.json').write_text('{"status":"completed"}')
    (run / 'audit_manifest.json').write_text(json.dumps({'files': {
        n: digest(run / n) for n in ('comparison.json', 'setup_status.json')}}))
    assert find_environment(tmp_path) == run


def test_fresh_invocation_preserves_original_export_hook_and_arguments(tmp_path):
    called = []
    def original_export(folder):
        called.append(folder)
    producer = SimpleNamespace(export=original_export)
    def inference_main():
        assert sys.argv[1:] == ['--output', str(tmp_path)]
        (tmp_path / 'predictions.json').write_text('[]')
        producer.export(tmp_path)
    producer.main = inference_main
    original_arguments = sys.argv
    invoke_fresh(producer, tmp_path)
    assert (tmp_path / 'predictions.json').is_file()
    assert not called
    assert producer.export is original_export
    assert sys.argv is original_arguments
    with pytest.raises(ValueError):
        invoke_fresh(producer, tmp_path)


def test_inference_failure_restores_original_hook(tmp_path):
    def original_export(folder):
        pass
    def fail():
        raise RuntimeError('inference interrupted')
    producer = SimpleNamespace(export=original_export, main=fail)
    original_arguments = sys.argv
    with pytest.raises(RuntimeError, match='interrupted'):
        invoke_fresh(producer, tmp_path)
    assert producer.export is original_export
    assert sys.argv is original_arguments
