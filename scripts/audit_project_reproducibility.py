"""Inventory exports and run non-neural checks without downloading or fitting models."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


COMMANDS = (
    ('tests', ('-m', 'pytest', '-q')),
    ('semantic_sufficiency', ('scripts/verify_semantic_sufficiency.py',)),
    ('scifact_retrieval', ('scripts/verify_sentence_rag.py',)),
    ('climate_retrieval', ('scripts/verify_climate_rag.py',)),
    ('climate_supplied', ('scripts/verify_climate_supplied.py',)),
    ('live_controller', ('scripts/verify_gated_generation.py',)),
    ('multilingual_generation', ('scripts/verify_xfever_generation.py',)),
    ('copy_order_generation', ('scripts/verify_generation_robustness.py',)),
    ('project_status', ('scripts/consolidate_project_status.py',)),
)
CORE_EXPORTS = {
    'semantic_sufficiency_received', 'sentence_rag_received', 'climate_rag_received',
    'climate_supplied_received', 'integrated_gate_received', 'gated_generation_received',
    'xfever_received', 'xfever_multilingual_received', 'xfever_generation_received',
    'generation_robustness_received',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


def audit_receipt(receipt, root):
    report = {'receipt': str(receipt.relative_to(root)).replace('\\', '/'),
              'receipt_sha256': digest(receipt), 'files': []}
    try:
        data = json.loads(receipt.read_text(encoding='utf-8'))
        items = data['files'] if receipt.name == 'audit_manifest.json' else data
        if not isinstance(items, dict) or not items:
            raise ValueError('A nonempty file-to-digest mapping is required')
    except (ValueError, KeyError, TypeError) as exc:
        report['schema_error'] = str(exc)
        return report
    for name, expected in sorted(items.items()):
        entry = {'name': name, 'expected_sha256': expected}
        pure = PurePosixPath(name)
        path = receipt.parent / name
        safe = (not pure.is_absolute() and '..' not in pure.parts and ':' not in name and '\\' not in name
                and path.resolve().is_relative_to(receipt.parent.resolve()))
        if not safe:
            entry['status'] = 'invalid_path'
        elif not isinstance(expected, str) or re.fullmatch(r'[0-9a-fA-F]{64}', expected) is None:
            entry['status'] = 'invalid_digest'
        elif not path.is_file():
            entry['status'] = 'missing'
        else:
            entry['actual_sha256'] = digest(path)
            entry['status'] = 'verified' if entry['actual_sha256'] == expected.lower() else 'checksum_mismatch'
        report['files'].append(entry)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='artifacts/reproducibility_audit_v1')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (root / args.out).resolve()
    if not out.is_relative_to(root / 'artifacts'):
        parser.error('Output must be inside the project artifacts directory')
    if out == root / 'artifacts' or out.exists() and any(out.iterdir()) and not (out / 'audit_inventory.json').exists():
        parser.error('Choose a new audit folder or the existing audit folder; do not overwrite an experiment')
    out.mkdir(parents=True, exist_ok=True)
    if sys.flags.optimize:
        parser.error('Run without -O: the frozen replay scripts use assertions')
    packages = ('pytest', 'PyYAML', 'jsonschema', 'psutil', 'numpy', 'scikit-learn', 'matplotlib',
                'torch', 'transformers', 'sentence-transformers')
    installed = {}
    for name in packages:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    runtime = {'python': platform.python_version(), 'platform': platform.platform(),
               'interpreter': sys.executable, 'packages': installed, 'neural_inference_run': False,
               'model_weights_checked': False, 'sqlite_caches_checked': False,
               'windows_execution_verified_here': platform.system() == 'Windows'}
    syntax = []
    for directory in ('src', 'scripts', 'tests'):
        for path in sorted((root / directory).rglob('*.py')):
            try:
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            except (SyntaxError, UnicodeError) as exc:
                syntax.append({'path': str(path.relative_to(root)), 'error': str(exc)})
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join([str(root / 'src'), str(root / 'scripts'), environment.get('PYTHONPATH', '')])
    environment['PYTHONUTF8'] = '1'
    environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    environment.pop('PYTHONOPTIMIZE', None)
    command_results = []
    for name, arguments in COMMANDS:
        print(f'Checking {name}...', flush=True)
        try:
            result = subprocess.run([sys.executable, *arguments], cwd=root, env=environment,
                                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
            log = result.stdout + result.stderr
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            log, code = str(exc), -1
        (out / f'{name}.log').write_text(log, encoding='utf-8')
        command_results.append({'name': name, 'arguments': list(arguments), 'returncode': code,
                                'log': f'{name}.log', 'status': 'passed' if code == 0 else 'failed'})
    receipts = []
    for path in sorted((root / 'artifacts').rglob('*manifest.json')):
        if out in path.parents or path.name not in ('output_manifest.json', 'audit_manifest.json'):
            continue
        receipts.append(audit_receipt(path, root))
    counts = Counter(entry['status'] for receipt in receipts for entry in receipt['files'])
    missing = [{'receipt': r['receipt'], 'path': str(PurePosixPath(r['receipt']).parent / e['name'])}
               for r in receipts for e in r['files'] if e['status'] == 'missing']
    core_missing = [r for r in receipts if PurePosixPath(r['receipt']).parent.name in CORE_EXPORTS
                    and (r.get('schema_error') or any(e['status'] != 'verified' for e in r['files']))]
    present_core = {PurePosixPath(r['receipt']).parent.name for r in receipts}
    absent_core = sorted(CORE_EXPORTS - present_core)
    profiles = []
    for name in ('gated_generation_received', 'semantic_sufficiency_received', 'xfever_generation_received', 'generation_robustness_received'):
        source = root / 'artifacts' / name / 'input_manifest.json'
        data = json.loads(source.read_text(encoding='utf-8'))
        recorded = data['packages']
        profiles.append({'source': str(source.relative_to(root)), 'sha256': digest(source), 'recorded_packages': recorded,
                         'current_package_differences': {k: {'recorded': v, 'current': installed.get(k)}
                            for k, v in recorded.items() if installed.get(k) != v}})
    corrupt = sum(n for status, n in counts.items() if status not in ('verified', 'missing'))
    failed = [r['name'] for r in command_results if r['status'] == 'failed']
    blocking = bool(corrupt or failed or syntax or core_missing or absent_core or any(r.get('schema_error') for r in receipts))
    inventory = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'runtime': runtime,
                 'receipts': receipts, 'file_status_counts': dict(counts), 'missing_older_files': missing,
                 'core_export_problems': [r['receipt'] for r in core_missing], 'absent_core_exports': absent_core,
                 'commands': command_results, 'syntax_errors': syntax, 'recorded_environments': profiles,
                 'status': 'failed_checks' if blocking else ('checked_with_export_gaps' if missing else 'scoped_checks_passed'),
                 'full_project_complete': False, 'clean_environment_reproduction_verified': False}
    write_json(out / 'audit_inventory.json', inventory)
    write_json(out / 'recorded_environments.json', profiles)
    write_json(out / 'missing_export_files.json', missing)
    # These are recorded top-level versions, not a transitive dependency lock or tested installer.
    for name in ('gated_generation_received', 'semantic_sufficiency_received'):
        profile = next(p for p in profiles if Path(p['source']).parent.name == name)
        content = '# Recorded versions from a frozen run receipt; not a clean-environment guarantee.\n'
        content += '\n'.join(f'{k}=={v}' for k, v in sorted(profile['recorded_packages'].items())) + '\n'
        (out / f'{name}_packages.txt').write_text(content, encoding='utf-8')
    lines = ['# Current reproducibility audit', '',
        f"Audit status: {inventory['status']}. This reports file availability and executable checks; it does not close the project charter.", '',
        f"{len(receipts)} receipts inspected: {counts['verified']} file references verified, {counts['missing']} missing, {corrupt} invalid or mismatched. Missing files are listed individually in missing_export_files.json. No expected hashes or original experiment inputs were changed to make a check pass.", '',
        '| Check | Result | Log |', '|---|---|---|']
    for result in command_results:
        lines.append(f"| {result['name']} | {result['status']} | {result['log']} |")
    lines += ['', 'The recent replay chain checks cached semantic features, serialized gate probabilities, original BM25 document IDs/scores and sentence selections, source labels, controller decisions, multilingual aligned outputs, copy/order responses and statistics. These are export replays, not neural inference or training reproduction. Tests use synthetic cases and do not measure model efficacy.', '',
        'Older received folders contain selected result exports. Their receipts also name input identities, audits and arrays that are absent from this package. Keep the originals on the Windows computer and retain their recorded checksums. Missing files are not verified, and the package is not a complete archive for reproducing all historical experiments. The current core exports are checked separately from those gaps.', '',
        f"Current audit Python: {runtime['python']}; platform: {platform.system()}. Windows execution verified here: {runtime['windows_execution_verified_here']}. The recorded package profiles preserve upstream declarations and differences from this audit environment. They are not complete dependency locks; no clean neural environment was built. Model weights, SQLite caches, tokenizer counts and neural execution were not checked. No downloads or model fitting are performed by this command.", '',
        'The setup now installs the editable package before direct scripts. Current Windows instructions use the existing D:\\apv-rag-tesla369 folder and its interpreter explicitly. Existing environments and caches should be retained.', '',
        'Fresh retrieval and feature integration has been checked on saved cohorts, and the optional snapshot check has been replayed across four policies. These results reuse prior answers and do not authenticate publishers. Explanation truth, full multilingual retrieval and gate transfer, and correctness calibration for the integrated generator remain open. Numeric checks and NLI diagnostics cannot establish factual explanations. Tesla claims remain machine_candidate or abstain with no historical gold verdict. No manuscript, GitHub push or scope reduction is included.', '',
        'Next research work requires an evaluation design for the remaining claims that does not fit or select a policy on the same labels used to judge it. Historical export recovery can use original Windows caches without a neural rerun. The missing older files in this consolidated copy are listed rather than treated as verified.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    source_paths = ('scripts/audit_project_reproducibility.py', 'scripts/setup_windows.cmd', 'START_HERE.md',
                    'README.md', 'requirements-local.txt', 'requirements-colab.txt', 'pyproject.toml')
    write_json(out / 'audit_manifest.json', {'audit_code_and_setup_sha256': {p: digest(root / p) for p in source_paths},
        'files': {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'audit_manifest.json'}})
    print(f"Audit finished: {inventory['status']}; {counts['verified']} hashes match, {counts['missing']} missing references; {len(failed)} failed commands.")
    print(out)
    return int(blocking)


if __name__ == '__main__':
    raise SystemExit(main())
