"""Reinstall the recorded Windows environment and rerun 300 frozen English claims."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'artifacts/fever_pipeline_received_v1/fever_pipeline_v1'
TOLERANCE = 1e-5  # Declared before the rerun; strings, labels and IDs stay exact.
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def require_empty(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError('Existing files refused; choose a new run directory. Originals are preserved.')


def pinned_requirements(records):
    pins = {}
    for item in records:
        name, value = item['name'], item['version']
        if re.sub(r'[-_.]+', '-', name).lower() == 'apv-rag':
            continue
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or not re.fullmatch(r'[A-Za-z0-9_.+!-]+', value):
            raise ValueError('Invalid package name/version in installed environment')
        key = re.sub(r'[-_.]+', '-', name).lower()
        if key in pins:
            raise ValueError('Duplicate distribution')
        pins[key] = f'{name}=={value}'
    return '\n'.join(pins[k] for k in sorted(pins)) + '\n'


def compare_values(actual, expected, path='$'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return [path + ': keys differ']
        return [error for k, value in expected.items()
                for error in compare_values(actual[k], value, path + '.' + str(k))]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [path + ': list length/type differs']
        return [error for i, value in enumerate(expected)
                for error in compare_values(actual[i], value, f'{path}[{i}]')]
    if isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or
                not math.isfinite(actual) or not math.isfinite(expected) or
                abs(actual - expected) > TOLERANCE):
            return [path + ': numeric value differs or is invalid']
        return []
    return [] if type(actual) is type(expected) and actual == expected else [path + ': value differs']


def preflight():
    from run_fever_pipeline import load_inputs
    from verify_fever_pipeline import audit
    from apv_rag.nli_comparison import _model_files
    source, manifest, ref, claims = load_inputs()
    audit(BASELINE)
    for name, subdir in [('generator', 'qwen2.5-1.5b-instruct'),
                         ('nli', 'nli-deberta-v3-small'), ('embedding', 'all-MiniLM-L6-v2')]:
        expected = ref['generator']['files'] if name == 'generator' else ref['feature_models'][name]
        if _model_files(ROOT / 'models' / subdir) != expected:
            raise ValueError('Missing or changed local model: ' + subdir)
    index = ROOT / 'data/processed/fever/heldout_v1/wiki.sqlite'
    if not index.is_file():
        raise ValueError('Original Wikipedia SQLite index is missing')
    return source, manifest, ref, claims, index


def command(arguments, out, name):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'scripts')])
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONOPTIMIZE', None)
    with (out / (name + '.log')).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(arguments, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        if process.wait() != 0:
            raise ValueError(name + ' failed; see its log. No success is recorded.')


def worker(out):
    if Path(sys.prefix).resolve() != (out / 'environment').resolve() or sys.prefix == sys.base_prefix:
        raise ValueError('Neural rerun requires the isolated environment created for this run')
    cfg = (out / 'environment/pyvenv.cfg').read_text(encoding='utf-8').lower()
    if 'include-system-site-packages = false' not in cfg:
        raise ValueError('System-site packages must be disabled')
    source, manifest, ref, claims, index = preflight()
    if {k: importlib.metadata.version(k) for k in ref['packages']} != ref['packages']:
        raise ValueError('Reinstalled neural package versions differ from frozen reference')
    from run_fever_pipeline import run_stage
    from verify_fever_pipeline import verify_stage_data
    from apv_rag.fever_corpus import EXPECTED_ARCHIVE_SHA256, EXPECTED_PAGES
    import sqlite3
    with sqlite3.connect(f'file:{index.resolve().as_posix()}?mode=ro', uri=True) as db:
        metadata = dict(db.execute('SELECT key,value FROM metadata'))
    identity = load(BASELINE / 'input_manifest.json')
    if (metadata != identity['index_metadata'] or metadata['archive_sha256'] != EXPECTED_ARCHIVE_SHA256 or
            metadata['source_rows'] != str(EXPECTED_PAGES)):
        raise ValueError('Original Wikipedia index metadata differs')
    stage = out / 'confirmation'
    require_empty(stage)
    prior_receipt = digest(BASELINE / 'output_manifest.json')
    # The original stage computes retrieval, tokenization, features and generation.
    # Its fresh SQLite database gets seeds={} and contains no old answers.
    rows = run_stage('confirmation', claims['confirmation'], source, out, index, ref, identity)
    if len(rows) != 1200:
        raise ValueError('Expected 300 claims and 1200 policy records')
    verify_stage_data(stage, claims['confirmation'], ref['gate_models'])
    errors = []
    for name in ('prepared_contexts.json', 'feature_cache.json', 'responses_used.json', 'predictions.json'):
        errors += compare_values(load(stage / name), load(BASELINE / 'confirmation' / name), name)
    def jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    errors += compare_values(jsonl(stage / 'retrieved_contexts.jsonl'),
                             jsonl(BASELINE / 'confirmation/retrieved_contexts.jsonl'), 'retrieval')
    if digest(BASELINE / 'output_manifest.json') != prior_receipt:
        raise ValueError('Original receipt changed during run')
    # Recheck originals, source inputs and local weights after inference.
    preflight()
    result = {'stage': 'clean_english_confirmation_v1',
              'status': 'reproduced' if not errors else 'rerun_differences',
              'claims': 300, 'policy_records': 1200, 'neural_inference_rerun': True,
              'prior_feature_or_answer_cache_imported': False,
              'absolute_numeric_tolerance': TOLERANCE,
              'difference_count': len(errors), 'differences': errors,
              'baseline_receipt_sha256': prior_receipt,
              'confirmation_fit_performed': False, 'development_training_rerun': False,
              'multilingual_neural_reproduction_verified': False, 'full_charter_complete': False,
              'python': sys.version, 'platform': platform.platform(),
              'index_metadata': metadata,
              'scope': 'Fresh inference for the original 300 English confirmation claims, four frozen policies. '
                       'No calibration fitting, new accuracy experiment, historical authentication or explanation-truth claim.'}
    save(out / 'comparison.json', result)
    print('Clean English comparison: ' + result['status'], flush=True)
    if errors:
        raise ValueError('Fresh outputs differ; differences preserved, tolerance and originals unchanged')


def export(out):
    target = ROOT / 'clean_confirmation_outputs.zip'
    paths = [p for p in out.rglob('*') if p.is_file() and
             not {'environment', 'wheelhouse', '__pycache__'} & set(p.relative_to(out).parts)]
    save(out / 'audit_manifest.json', {'code_sha256': {str(Path(__file__).relative_to(ROOT)): digest(__file__)},
                                      'files': {p.relative_to(out).as_posix(): digest(p) for p in paths}})
    temporary = target.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths + [out / 'audit_manifest.json']:
            z.write(p, 'clean_confirmation_v1/' + p.relative_to(out).as_posix())
    temporary.replace(target)
    print('Upload: ' + str(target), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        out = args.worker.resolve()
        if not out.is_relative_to(ROOT / 'artifacts/clean_confirmation_runs'):
            parser.error('Worker output outside the clean reproduction run directory')
        worker(out)
        return
    if platform.system() != 'Windows':
        parser.error('Run this installer on Windows using the existing project .venv interpreter')
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        parser.error('Use .\\.venv\\Scripts\\python.exe so the package snapshot comes from this project')
    _, _, ref, _, _ = preflight()
    if {k: importlib.metadata.version(k) for k in ref['packages']} != ref['packages']:
        raise ValueError('Existing neural versions differ; do not alter the frozen reference')
    if args.preflight:
        print('Original sources, receipts, weights and index present. No installation or neural rerun performed.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'artifacts/clean_confirmation_runs' / ('run_' + stamp)
    out.mkdir(parents=True)
    packages = json.loads(subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'],
                                                 text=True, encoding='utf-8'))
    save(out / 'source_packages.json', packages)
    lock = out / 'requirements.txt'
    lock.write_text(pinned_requirements(packages), encoding='utf-8')
    save(out / 'setup_status.json', {'status': 'started', 'source_python': sys.version,
         'source_interpreter': sys.executable, 'neural_inference_rerun': False,
         'full_charter_complete': False})
    try:
        child = out / 'environment/Scripts/python.exe'
        command([sys.executable, '-m', 'venv', str(out / 'environment')], out, 'create_environment')
        wheels = out / 'wheelhouse'
        command([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                 '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                 '-r', str(lock), '-d', str(wheels)], out, 'download_wheels')
        save(out / 'wheel_receipt.json', {p.name: digest(p) for p in sorted(wheels.iterdir()) if p.is_file()})
        command([str(child), '-m', 'pip', 'install', '--no-index', '--no-deps',
                 '--find-links', str(wheels), '-r', str(lock)], out, 'install_packages')
        command([str(child), '-m', 'pip', 'check'], out, 'dependency_check')
        installed = json.loads(subprocess.check_output([str(child), '-m', 'pip', 'list', '--format=json'],
                                                       text=True, encoding='utf-8'))
        save(out / 'installed_packages.json', installed)
        if pinned_requirements(installed) != lock.read_text(encoding='utf-8'):
            raise ValueError('Isolated installed package inventory differs from source snapshot')
        command([str(child), str(Path(__file__)), '--worker', str(out)], out, 'neural_rerun')
        save(out / 'setup_status.json', {'status': 'completed', 'neural_inference_rerun': True,
                                        'full_charter_complete': False})
    except Exception as exc:
        save(out / 'failure.json', {'error': str(exc), 'status': 'failed', 'full_charter_complete': False})
        export(out)
        raise
    export(out)


if __name__ == '__main__':
    main()
