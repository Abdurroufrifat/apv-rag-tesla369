"""Copy checksum-matching original files into incomplete received-export folders."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact_path(root, name):
    pure = PurePosixPath(name)
    path = root / name
    if pure.is_absolute() or '..' in pure.parts or ':' in name or '\\' in name or not path.resolve().is_relative_to(root / 'artifacts'):
        raise ValueError(f'Invalid artifact path: {name}')
    return path


def restore_exports(root, entries, plan_only=False):
    root = root.resolve()
    plans, seen = [], set()
    for entry in entries:
        target = artifact_path(root, entry['target'])
        if target in seen or not target.parent.name.endswith('_received'):
            raise ValueError(f'Invalid or duplicate received-export target: {target}')
        seen.add(target)
        receipt = json.loads((target.parent / 'output_manifest.json').read_text(encoding='utf-8'))
        expected = receipt.get(target.name)
        if expected != entry['sha256']:
            raise ValueError(f'Recovery map differs from original receipt: {target}')
        if target.exists():
            if not target.is_file() or digest(target) != expected:
                raise ValueError(f'Existing target differs; refusing overwrite: {target}')
            plans.append({'target': target, 'source': target, 'expected': expected, 'status': 'already_present'})
            continue
        source = None
        for name in entry['matching_original_paths']:
            candidate = artifact_path(root, name)
            if candidate.is_file() and digest(candidate) == expected:
                source = candidate
                break
        if source is None:
            raise ValueError(f'No checksum-matching original file for: {target}')
        plans.append({'target': target, 'source': source, 'expected': expected, 'status': 'planned'})
    # Check every source and target before copying any file.
    copied = 0
    if not plan_only:
        for plan in plans:
            if plan['status'] == 'already_present':
                continue
            target, source = plan['target'], plan['source']
            fd, name = tempfile.mkstemp(prefix='restore-', suffix='.tmp', dir=target.parent)
            os.close(fd)
            temporary = Path(name)
            try:
                shutil.copyfile(source, temporary)
                if digest(temporary) != plan['expected']:
                    raise ValueError(f'Source changed during copy: {source}')
                # Publish the copied bytes without overwriting a target created meanwhile.
                os.link(temporary, target)
                plan['status'] = 'copied'
                copied += 1
            finally:
                temporary.unlink(missing_ok=True)
    return {'copied': copied, 'already_present': sum(p['status'] == 'already_present' for p in plans),
            'plan_only': plan_only, 'files': [{'target': str(p['target'].relative_to(root)).replace('\\', '/'),
                'source': str(p['source'].relative_to(root)).replace('\\', '/'), 'sha256': p['expected'],
                'status': p['status']} for p in plans]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    mapping = json.loads((root / 'artifacts/reproducibility_audit_windows_verification/alias_recovery_map.json').read_text(encoding='utf-8'))
    if mapping['unmatched_files']:
        raise ValueError('Unmatched files remain in the recovery map')
    report = restore_exports(root, mapping['matching_original_files'], args.plan_only)
    out = root / 'artifacts/export_recovery_v1'
    out.mkdir(exist_ok=True)
    (out / 'recovery_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f"Copied: {report['copied']}; already present: {report['already_present']}; plan only: {args.plan_only}")
    print('Original files and expected checksums were preserved. Run audit_project_reproducibility.py next.')


if __name__ == '__main__':
    main()
