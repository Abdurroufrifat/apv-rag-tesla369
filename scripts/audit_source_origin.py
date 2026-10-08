"""Collect fixed institutional pages once, or verify saved captures offline."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.source_origin import (SourcePolicy, capture_source, check_binding,
                                   digest, guarded_verify_claim)

POLICY_PATH = ROOT / 'config/source_origin_v1.json'
OUTPUT = ROOT / 'artifacts/source_origin_audit_v1'
CODE = ('src/apv_rag/source_origin.py', 'src/apv_rag/source_pipeline.py',
        'scripts/audit_source_origin.py', 'config/source_origin_v1.json',
        'data/pilot/tesla_phase1c_source_register_v0_1.csv')


def policies():
    data = json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    result = []
    for row in data['sources']:
        policy = SourcePolicy(row['key'], row['url'], tuple(row['anchors']),
                              row['evidence_role'], tuple(row['redirect_urls']))
        policy.validate()
        result.append(policy)
    if len({p.key for p in result}) != len(result):
        raise ValueError('Duplicate source policy identifiers')
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def code_hashes():
    return {name: digest((ROOT / name).read_bytes()) for name in CODE}


def diagnostics(out, attempts):
    rows = []
    def forbidden(*args):
        raise AssertionError('Unbound evidence reached model scoring')
    for policy in policies():
        attempt = next(a for a in attempts if a['source_key'] == policy.key)
        if attempt['status'] != 'captured':
            rows.append({'source_key': policy.key, 'capture_bound': False,
                         'status': 'inaccessible', 'diagnostic_cases': []})
            continue
        receipt = json.loads((out / policy.key / 'receipt.json').read_text(encoding='utf-8'))
        payload = (out / policy.key / 'page.html').read_bytes()
        if receipt['transport']['method'] != 'urllib_verified_https':
            raise ValueError('Synthetic transport cannot count as a live institutional capture')
        document = {'id': policy.key, 'source_key': policy.key, 'source_url': policy.url,
                    'language': 'en', 'text': policy.anchors[-1],
                    'evidence_role': policy.evidence_role}
        if not check_binding(document, policy, receipt, payload)['bound']:
            raise ValueError(f'Original capture no longer binds: {policy.key}')
        controls = [
            ('whitespace_normalization', dict(document, text='\n' + document['text'] + '\t'), receipt, payload, True),
            ('fabricated_excerpt', dict(document, text='APV fabricated quote never present 369987654'), receipt, payload, False),
            ('declared_url_spoof', dict(document, source_url='https://spoof.example/tesla'), receipt, payload, False),
            ('wrong_page_identifier', dict(document, source_url=policy.url + '/other'), receipt, payload, False),
            ('http_downgrade', dict(document, source_url=policy.url.replace('https:', 'http:')), receipt, payload, False),
            ('changed_payload', document, receipt, payload + b'changed', False),
            ('historical_role_escalation', dict(document, evidence_role='authenticated_historical_quote'), receipt, payload, False),
        ]
        altered = copy.deepcopy(receipt)
        altered['transport']['certificate_verification'] = False
        controls.append(('disabled_certificate_verification', document, altered, payload, False))
        observed = []
        for name, doc, rec, body, expected in controls:
            bound = check_binding(doc, policy, rec, body)['bound']
            if bound != expected:
                raise ValueError(f'Control failed: {policy.key}/{name}')
            if not bound:
                result = guarded_verify_claim('Tesla motor', 'en', [doc], forbidden,
                                              {policy.key: (policy, rec, body)})
                if result['status'] != 'abstain':
                    raise ValueError('Unbound context did not abstain')
            observed.append({'case': name, 'expected_bound': expected, 'observed_bound': bound,
                             'unbound_model_calls': 0 if not bound else None})
        missing = guarded_verify_claim('Tesla motor', 'en', [document], forbidden, {})
        if missing['status'] != 'abstain':
            raise ValueError('Missing capture did not abstain')
        observed.append({'case': 'missing_capture', 'expected_bound': False,
                         'observed_bound': False, 'unbound_model_calls': 0})
        rows.append({'source_key': policy.key, 'capture_bound': True,
                     'status': 'metadata_or_navigation_only', 'diagnostic_cases': observed})
    return rows


def capture(out):
    if (out / 'audit_manifest.json').exists():
        raise ValueError('Frozen capture already exists. Verify it; choose a separate output to recollect.')
    attempts = []
    for policy in policies():
        try:
            receipt, payload = capture_source(policy)
            write_json(out / policy.key / 'receipt.json', receipt)
            (out / policy.key / 'page.html').write_bytes(payload)
            attempts.append({'source_key': policy.key, 'status': 'captured', 'bytes': len(payload)})
        except Exception as exc:
            attempts.append({'source_key': policy.key, 'status': 'inaccessible',
                             'error_type': type(exc).__name__, 'error': str(exc)})
    write_json(out / 'attempts.json', attempts)
    rows = diagnostics(out, attempts)
    summary = {'stage': 'source_origin_v1', 'attempted_sources': len(attempts),
               'captured_sources': sum(r['capture_bound'] for r in rows),
               'inaccessible_sources': [r['source_key'] for r in rows if not r['capture_bound']],
               'diagnostics': rows,
               'scope': 'Fixed institutional metadata/navigation captures; trusted collector, unsigned receipts. '
                        'No historical quote, author, primary-source status, publication-date truth, '
                        'OCR, embedded-resource content, model accuracy or provenance benefit authenticated.',
               'collection_utc': datetime.now(timezone.utc).isoformat()}
    write_json(out / 'summary.json', summary)
    lines = ['# Source origin and capture binding', '', summary['scope'], '',
             f"Attempted pages: {len(attempts)}. Captured and replay-bound: {summary['captured_sources']}. "
             f"Inaccessible: {len(summary['inaccessible_sources'])}.", '',
             '| Source | Result |', '|---|---|']
    for row in rows:
        lines.append(f"| {row['source_key']} | {row['status']} |")
    lines += ['', 'For each successful capture, the saved metadata excerpt and whitespace control bind. '
              'Eight altered/missing-source controls are refused before model scoring; no neural inference runs. '
              'These are deterministic admission checks, not independently sampled attacks or accuracy evidence.', '',
              'The Politika source register remains unchanged. A page catalogue capture would not authenticate '
              'the privately saved newspaper image, its transcript or the Tesla quotation. Inaccessible sources '
              'are recorded as inaccessible, never as authenticated. HTML text extraction cannot establish CSS '
              'visibility or detect a compromised legitimate publisher. Receipts cannot withstand replacement '
              'of the entire archive and manifest by a malicious collector.', '',
              'The bounded origin-binding step is complete; the original historical authentication requirement '
              'remains open. No new human labels, manuscript or GitHub push are involved.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    files = {p.relative_to(out).as_posix(): digest(p.read_bytes())
             for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'audit_manifest.json'}
    write_json(out / 'audit_manifest.json', {'protocol': 'source_origin_v1',
                                           'code_sha256': code_hashes(), 'files': files})
    return verify(out)


def verify(out=OUTPUT):
    receipt = json.loads((out / 'audit_manifest.json').read_text(encoding='utf-8'))
    if receipt['code_sha256'] != code_hashes():
        raise ValueError('Source origin code/policy/register changed')
    observed_files = {p.relative_to(out).as_posix() for p in out.rglob('*')
                      if p.is_file() and p.name != 'audit_manifest.json'}
    if observed_files != set(receipt['files']):
        raise ValueError('Capture archive file inventory changed')
    for name, expected in receipt['files'].items():
        if digest((out / name).read_bytes()) != expected:
            raise ValueError(f'Capture audit file changed: {name}')
    attempts = json.loads((out / 'attempts.json').read_text(encoding='utf-8'))
    expected_keys = {p.key for p in policies()}
    if len(attempts) != len(expected_keys) or {a['source_key'] for a in attempts} != expected_keys:
        raise ValueError('Capture attempts do not match fixed policy')
    if any(a['status'] not in ('captured', 'inaccessible') for a in attempts):
        raise ValueError('Unknown capture outcome')
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    rows = diagnostics(out, attempts)
    if summary['diagnostics'] != rows:
        raise ValueError('Capture diagnostics do not replay')
    if (summary['attempted_sources'] != len(attempts) or
        summary['captured_sources'] != sum(r['capture_bound'] for r in rows) or
        summary['inaccessible_sources'] != [r['source_key'] for r in rows if not r['capture_bound']]):
        raise ValueError('Capture summary counts do not replay')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true', help='Collect a new fixed-source audit')
    parser.add_argument('--verify', action='store_true', help='Replay shipped audit offline (default)')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.capture and args.verify:
        parser.error('Choose capture or offline verify')
    summary = capture(args.output) if args.capture else verify(args.output)
    print(f"Source origin audit verified: {summary['captured_sources']}/{summary['attempted_sources']} "
          'institutional pages captured; no historical verdict assigned.')
    print('Neural model calls: 0. Historical authentication remains open.')


if __name__ == '__main__':
    main()
