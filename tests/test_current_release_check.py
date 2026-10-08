from pathlib import Path
import subprocess

from check_current_release import execute_checks


def test_all_checks_use_same_interpreter_and_project_paths(tmp_path):
    calls = []
    def fake(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, 'passed\n', '')
    records = execute_checks(tmp_path, tmp_path / 'logs', [('test', ('-m', 'pytest', '-q'))], runner=fake)
    assert records[0]['status'] == 'passed'
    assert calls[0][1]['cwd'] == tmp_path
    assert str(tmp_path / 'scripts') in calls[0][1]['env']['PYTHONPATH']
    assert (tmp_path / 'logs/test.log').read_text() == 'passed\n'


def test_failure_is_recorded_and_remaining_checks_run(tmp_path):
    def fake(command, **kwargs):
        return subprocess.CompletedProcess(command, 1 if command[-1] == 'bad' else 0, '', 'detail')
    records = execute_checks(tmp_path, tmp_path / 'logs',
                             [('first', ('bad',)), ('second', ('good',))], runner=fake)
    assert [row['status'] for row in records] == ['failed', 'passed']


def test_timeout_is_a_failure(tmp_path):
    def fake(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    records = execute_checks(tmp_path, tmp_path / 'logs', [('slow', ('slow',))], runner=fake)
    assert records[0]['status'] == 'failed'
    assert 'TimeoutExpired' in (tmp_path / 'logs/slow.log').read_text()
