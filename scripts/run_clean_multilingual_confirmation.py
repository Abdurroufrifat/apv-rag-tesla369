"""Rerun frozen multilingual confirmation using the verified clean English environment."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_clean_confirmation import (compare_values, command, digest, load,
                                    pinned_requirements, require_empty, save, TOLERANCE)

BASELINE = ROOT / 'artifacts/multilingual_confirmation_received_v1/multilingual_confirmation_v1'
COMPARE_FILES = ('prepared_contexts.json', 'feature_cache.json', 'multilingual_feature_cache.json',
                 'responses_used.json', 'predictions.json', 'direct_predictions.json',
                 'correctness_predictions.json', 'summary.json')


def check_receipt(folder):
    files = load(folder / 'audit_manifest.json')['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty or invalid receipt')
    for name, expected in files.items():
        pure = PurePosixPath(name)
        if (pure.is_absolute() or '..' in pure.parts or ':' in name or '\\' in name or
                not (folder / name).resolve().is_relative_to(folder.resolve())):
            raise ValueError('Unsafe receipt path')
        if not (folder / name).is_file() or digest(folder / name) != expected:
            raise ValueError('Clean English receipt mismatch: ' + name)


def find_environment(root):
    parent = root / 'artifacts/clean_confirmation_runs'
    for out in sorted(parent.glob('run_*'), reverse=True):
        if not (out / 'setup_status.json').exists() or load(out / 'setup_status.json')['status'] != 'completed':
            continue
        check_receipt(out)
        result = load(out / 'comparison.json')
        if (result.get('status') != 'reproduced' or result.get('difference_count') != 0 or
                result.get('claims') != 300 or result.get('policy_records') != 1200 or
                result.get('neural_inference_rerun') is not True or
                result.get('prior_feature_or_answer_cache_imported') is not False):
            raise ValueError('Completed clean English run is not a successful reproduction')
        if not (out / 'environment/Scripts/python.exe').is_file():
            raise ValueError('Keep the isolated environment from the successful clean English run')
        return out
    raise ValueError('No completed clean English environment found; retain its original run directory')


def invoke_fresh(producer, inference):
    """Use the unchanged inference main; suppress its shared root ZIP destination only."""
    require_empty(inference)
    previous_args, previous_export = sys.argv, producer.export
    def local_export(folder):
        if Path(folder).resolve() != inference.resolve():
            raise ValueError('Unexpected export destination')
        print('Fresh multilingual inference saved in its own run directory.', flush=True)
    try:
        producer.export = local_export
        sys.argv = ['run_multilingual_confirmation.py', '--output', str(inference)]
        producer.main()
    finally:
        producer.export = previous_export
        sys.argv = previous_args


def worker(out):
    context = load(out / 'run_context.json')
    clean_run = (ROOT / context['clean_environment_run']).resolve()
    if not clean_run.is_relative_to(ROOT / 'artifacts/clean_confirmation_runs'):
        raise ValueError('Unexpected source environment directory')
    if Path(sys.prefix).resolve() != (clean_run / 'environment').resolve() or sys.prefix == sys.base_prefix:
        raise ValueError('Use the selected isolated environment for neural inference')
    config = (clean_run / 'environment/pyvenv.cfg').read_text(encoding='utf-8').lower()
    if 'include-system-site-packages = false' not in config:
        raise ValueError('System-site packages must be disabled')
    check_receipt(clean_run)
    if digest(clean_run / 'audit_manifest.json') != context['source_clean_audit_sha256']:
        raise ValueError('Selected clean environment receipt changed after launch')
    installed = json.loads(subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'],
                                                  text=True, encoding='utf-8'))
    save(out / 'installed_packages.json', installed)
    if pinned_requirements(installed) != (clean_run / 'requirements.txt').read_text(encoding='utf-8'):
        raise ValueError('Isolated environment package versions changed')
    source_code = load(clean_run / 'audit_manifest.json')['code_sha256']
    if {n.replace('\\', '/') for n in source_code} != {'scripts/run_clean_confirmation.py'}:
        raise ValueError('Unexpected clean English producer record')
    if any(digest(ROOT / n.replace('\\', '/')) != h for n, h in source_code.items()):
        raise ValueError('Clean English producer bytes changed')
    import run_multilingual_confirmation as original
    from apv_rag.nli_comparison import _model_files
    manifest, rows, ref, fits = original.load_inputs()
    if {n: importlib.metadata.version(n) for n in ref['packages']} != ref['packages']:
        raise ValueError('Pinned multilingual package versions differ')
    original.audit(BASELINE, manifest, rows, ref, fits)
    baseline_digest = digest(BASELINE / 'output_manifest.json')
    producer_identity = original.identity(manifest, ref)
    # Validate the original four local model assets before accepting a fresh run.
    for name, subdir in [('generator', 'qwen2.5-1.5b-instruct'), ('nli', 'nli-deberta-v3-small'),
                         ('embedding', 'all-MiniLM-L6-v2'), ('multilingual', 'mdeberta-multilingual-nli')]:
        expected = (ref['generator']['files'] if name == 'generator' else
                    ref['multilingual_nli']['files'] if name == 'multilingual' else ref['feature_models'][name])
        if _model_files(ROOT / 'models' / subdir) != expected:
            raise ValueError('Missing or changed local model: ' + subdir)
    inference = out / 'inference'
    require_empty(inference)
    invoke_fresh(original, inference)
    original.audit(inference, manifest, rows, ref, fits)
    # Recheck original source, calibrator and producer identity after the run.
    checked_manifest, checked_rows, checked_ref, checked_fits = original.load_inputs()
    if (original.identity(checked_manifest, checked_ref) != producer_identity or checked_fits != fits or
            checked_rows != rows or digest(BASELINE / 'output_manifest.json') != baseline_digest):
        raise ValueError('Original source or producer changed during neural inference')
    original.audit(BASELINE, checked_manifest, checked_rows, checked_ref, checked_fits)
    differences = [error for name in COMPARE_FILES
                   for error in compare_values(load(inference / name), load(BASELINE / name), name)]
    result = {'stage': 'clean_multilingual_confirmation_v1',
              'status': 'reproduced' if not differences else 'rerun_differences',
              'queries': len(rows), 'underlying_claim_ids': len({r['claim_id'] for r in rows}),
              'policy_records': len(load(inference / 'predictions.json')),
              'distinct_responses': len(load(inference / 'responses_used.json')),
              'neural_inference_rerun': True, 'prior_feature_or_answer_cache_imported': False,
              'confirmation_fitting': False, 'environment_reinstalled_this_step': False,
              'clean_environment_run': context['clean_environment_run'],
              'absolute_numeric_tolerance': TOLERANCE, 'difference_count': len(differences),
              'differences': differences, 'baseline_receipt_sha256': baseline_digest,
              'python': sys.version, 'platform': platform.platform(), 'full_charter_complete': False,
              'scope': 'Frozen closed-pool multilingual confirmation inference only; 100 IDs across six variants. '
                       'No new independent claims, calibration fitting, full-page retrieval, attribution authentication '
                       'or semantic explanation truth validation.'}
    save(out / 'comparison.json', result)
    print('Clean multilingual comparison: ' + result['status'], flush=True)
    if differences:
        raise ValueError('Fresh multilingual outputs differ; originals and comparison tolerance are preserved')


def export(out):
    paths = [p for p in out.rglob('*') if p.is_file()]
    save(out / 'audit_manifest.json', {'code_sha256': {
        n: digest(ROOT / n) for n in ('scripts/run_clean_multilingual_confirmation.py',
                                     'scripts/run_clean_confirmation.py')},
        'files': {p.relative_to(out).as_posix(): digest(p) for p in paths}})
    target = ROOT / 'clean_multilingual_confirmation_outputs.zip'
    temporary = target.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths + [out / 'audit_manifest.json']:
            z.write(p, 'clean_multilingual_confirmation_v1/' + p.relative_to(out).as_posix())
    temporary.replace(target)
    print('Upload: ' + str(target), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        out = args.worker.resolve()
        if not out.is_relative_to(ROOT / 'artifacts/clean_multilingual_confirmation_runs'):
            parser.error('Worker output outside multilingual reproduction runs')
        worker(out)
        return
    if platform.system() != 'Windows':
        parser.error('Run on Windows from your existing project .venv')
    clean_run = find_environment(ROOT)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'artifacts/clean_multilingual_confirmation_runs' / ('run_' + stamp)
    out.mkdir(parents=True)
    save(out / 'run_context.json', {'clean_environment_run': clean_run.relative_to(ROOT).as_posix(),
         'source_clean_audit_sha256': digest(clean_run / 'audit_manifest.json'),
         'source_packages': load(clean_run / 'installed_packages.json'),
         'environment_reinstalled_this_step': False})
    child = clean_run / 'environment/Scripts/python.exe'
    try:
        command([str(child), '-m', 'pip', 'check'], out, 'dependency_check')
        command([str(child), str(Path(__file__)), '--worker', str(out)], out, 'neural_rerun')
        save(out / 'status.json', {'status': 'completed', 'full_charter_complete': False})
    except Exception as exc:
        save(out / 'failure.json', {'status': 'failed', 'error': str(exc), 'full_charter_complete': False})
        export(out)
        raise
    export(out)


if __name__ == '__main__':
    main()
