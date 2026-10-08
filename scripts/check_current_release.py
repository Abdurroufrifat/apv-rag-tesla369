"""Run current non-neural export checks and package logs in one Windows command."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    ('tests', ('-m', 'pytest', '-q')),
    ('multilingual_confirmation', ('scripts/verify_multilingual_confirmation.py', '--verify')),
    ('source_origin', ('scripts/audit_source_origin.py', '--verify')),
    ('evidence_display', ('scripts/audit_evidence_display.py', '--verify')),
    ('guard_tradeoffs', ('scripts/analyze_guard_tradeoffs.py', '--verify')),
    ('latest_stage_evidence', ('scripts/verify_current_stage_evidence.py',)),
    ('component_ablation', ('scripts/analyze_frozen_components.py', '--verify')),
    ('project_status', ('scripts/consolidate_project_status.py',)),
)


def execute_checks(root, out, checks=CHECKS, *, runner=subprocess.run):
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'src'), str(root / 'scripts'),
                                       env.get('PYTHONPATH', '')])
    env['PYTHONIOENCODING'] = 'utf-8'
    records = []
    for name, arguments in checks:
        command = [sys.executable, *arguments]
        try:
            result = runner(command, cwd=root, env=env, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=300)
            output = result.stdout + result.stderr
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            code, output = 1, f'{type(exc).__name__}: {exc}\n'
        (out / f'{name}.log').write_text(output, encoding='utf-8')
        record = {'name': name, 'arguments': list(arguments), 'exit_code': code,
                  'status': 'passed' if code == 0 else 'failed'}
        if name == 'tests':
            match = re.search(r'(\d+) passed', output)
            record['tests_passed'] = int(match.group(1)) if match else None
        records.append(record)
        print(f"{name}: {record['status']}", flush=True)
    return records


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New empty result directory; defaults to a timestamped run')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.output or ROOT / 'artifacts/current_release_checks' / f'run_{stamp}'
    if out.exists() and any(out.iterdir()):
        parser.error('Choose a new empty output directory to preserve prior logs')
    out.mkdir(parents=True, exist_ok=True)
    records = execute_checks(ROOT, out)
    versions = {}
    for package in ('pytest', 'numpy', 'scikit-learn', 'torch', 'transformers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    requirements = ROOT / 'artifacts/project_status_consolidated/requirements.json'
    if requirements.exists():
        (out / 'research_requirements.json').write_bytes(requirements.read_bytes())
    summary = {'stage': 'current_release_checks_v1', 'utc': stamp,
               'status': 'current_export_checks_passed' if all(r['exit_code'] == 0 for r in records) else 'failed_checks',
               'checks': records, 'environment': {'platform': platform.platform(),
                    'python': sys.version, 'interpreter': sys.executable, 'packages': versions},
               'full_charter_complete': False, 'full_project_clean_reproduction_verified': False,
               'scoped_clean_confirmation_export_checks_passed': any(
                   r['name'] == 'latest_stage_evidence' and r['exit_code'] == 0 for r in records),
               'historical_attribution_authenticated': False, 'semantic_explanation_truth_verified': False,
               'qualification': 'Tests and current saved-output replays only. Preserves mixed results and '
                    'open research requirements; no neural inference, manuscript or GitHub push.'}
    write_json(out / 'summary.json', summary)
    lines = ['# Current export verification', '', f"Status: {summary['status']}.", '',
             '| Check | Result |', '|---|---|']
    lines += [f"| {r['name']} | {r['status']} |" for r in records]
    lines += ['', summary['qualification'], '',
              'Passing these checks does not complete the original research charter. Historical attribution '
              'authentication, semantic explanation truth, the remaining source/date/family/trained-robustness '
              'experiments and broader retrieval transfer remain unresolved. A clean neural installation and '
              'rerun are not performed by this checker. Received scoped English and multilingual clean '
              'confirmation exports are checked by latest_stage_evidence. Research requirements are copied '
              'into this export when available.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    code_files = ['scripts/check_current_release.py'] + [args[0] for _, args in CHECKS if args[0] != '-m']
    manifest = {'code_sha256': {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in code_files},
                'files': {p.name: sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
    write_json(out / 'audit_manifest.json', manifest)
    archive = ROOT / 'current_release_checks_outputs.zip'
    temporary = archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            z.write(path, f'current_release_check/{path.name}')
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise ValueError('Result ZIP integrity check failed')
    os.replace(temporary, archive)
    print(f"Current checks: {summary['status']}. Export: {archive}")
    print('Full research charter remains unfinished; no neural inference performed.')
    return 0 if summary['status'] == 'current_export_checks_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
