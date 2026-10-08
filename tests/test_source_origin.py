import copy
import io
import json
from pathlib import Path
import sys
import urllib.request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from apv_rag.source_origin import (
    CaptureError, OriginRedirectHandler, SourcePolicy, capture_source,
    check_binding, guarded_verify_claim,
)
from apv_rag.source_pipeline import PipelineConfig


POLICY = SourcePolicy('museum', 'https://museum.example/tesla',
                      ('Tesla catalogue', 'Museum collection'), 'catalogue_metadata')
HTML = b'<html><h1>Tesla catalogue</h1><p>Museum collection electric motor.</p><script>fabricated quote</script></html>'


class Response(io.BytesIO):
    status = 200
    headers = {'Content-Type': 'text/html; charset=utf-8'}

    def geturl(self):
        return POLICY.url


def captured(body=HTML):
    return capture_source(POLICY, opener=lambda request, timeout: Response(body))


def document(**changes):
    return {'id': 'a', 'text': 'Museum collection electric motor.', 'language': 'en',
            'source_key': 'museum', 'source_url': POLICY.url,
            'evidence_role': 'catalogue_metadata', **changes}


def test_genuine_metadata_and_whitespace_bind_but_script_text_does_not():
    receipt, payload = captured()
    assert check_binding(document(text='Museum\n collection electric motor.'), POLICY, receipt, payload)['bound']
    assert not check_binding(document(text='fabricated quote'), POLICY, receipt, payload)['bound']
    assert not check_binding(document(text=''), POLICY, receipt, payload)['bound']


@pytest.mark.parametrize('changes', [
    {'source_url': 'https://museum.example.evil/tesla'},
    {'source_url': 'http://museum.example/tesla'},
    {'source_url': 'https://museum.example@evil.example/tesla'},
    {'source_url': 'https://museum.example/other-page'},
    {'text': 'Tesla said the secret is three six nine.'},
    {'evidence_role': 'authenticated_historical_quote'},
])
def test_invalid_evidence_never_reaches_scorer(changes):
    receipt, payload = captured()
    def forbidden(*args):
        pytest.fail('Unbound evidence reached model scoring')
    result = guarded_verify_claim('Tesla motor', 'en', [document(**changes)], forbidden,
                                  {'museum': (POLICY, receipt, payload)})
    assert result['status'] == 'abstain'
    assert result['candidate_label'] is None
    assert result['selected_evidence'] == []
    assert result['origin_rejected_evidence']


def test_changed_payload_and_receipt_cannot_replay():
    receipt, payload = captured()
    assert not check_binding(document(), POLICY, receipt, payload + b'changed')['bound']
    altered = copy.deepcopy(receipt)
    altered['final_url'] = 'https://evil.example/tesla'
    assert not check_binding(document(), POLICY, altered, payload)['bound']
    altered = copy.deepcopy(receipt)
    altered['transport']['certificate_verification'] = False
    assert not check_binding(document(), POLICY, altered, payload)['bound']


@pytest.mark.parametrize('url', ['http://museum.example/tesla',
    'https://museum.example:8443/tesla', 'https://user@museum.example/tesla',
    'https://museum.example/tesla#quote'])
def test_invalid_policy_refused_before_network(url):
    with pytest.raises(CaptureError):
        capture_source(SourcePolicy('x', url, ('Tesla',), 'catalogue_metadata'),
                       opener=lambda *a, **kw: pytest.fail('Network used'))


@pytest.mark.parametrize('url', ['https://evil.example/tesla',
    'https://museum.example/different', 'http://museum.example/tesla'])
def test_redirect_rejected_before_follow(url):
    handler = OriginRedirectHandler(POLICY)
    with pytest.raises(CaptureError):
        handler.redirect_request(urllib.request.Request(POLICY.url), None, 302,
                                 'Found', {}, url)


@pytest.mark.parametrize('body', [b'<h1>Access denied</h1>', b'', b'<script>Tesla catalogue Museum collection</script>'])
def test_inaccessible_or_unidentified_content_is_not_a_capture(body):
    with pytest.raises(CaptureError):
        captured(body)


def test_payload_limits_enforced():
    with pytest.raises(CaptureError):
        capture_source(POLICY, max_bytes=20, opener=lambda *a, **kw: Response(HTML))


def test_genuine_context_reaches_existing_pipeline_without_gold_verdict():
    receipt, payload = captured()
    calls = []
    def scorer(claim, texts):
        calls.extend(texts)
        return [[.05, .9, .05] for _ in texts]
    result = guarded_verify_claim('electric motor', 'en', [document()], scorer,
        {'museum': (POLICY, receipt, payload)}, PipelineConfig(minimum_families=1))
    assert calls == ['Museum collection electric motor.']
    assert result['status'] == 'machine_candidate'
    assert 'historical attribution' in result['origin_qualification']


def test_missing_capture_is_rejected():
    result = guarded_verify_claim('motor', 'en', [document()], lambda *a: pytest.fail('Scored'), {})
    assert result['origin_rejected_evidence'][0]['reason'] == 'missing_capture'


def test_mixed_pool_only_scores_bound_documents():
    receipt, payload = captured()
    calls = []
    def scorer(claim, texts):
        calls.extend(texts)
        return [[.05, .9, .05] for _ in texts]
    result = guarded_verify_claim('electric motor', 'en',
        [document(), document(id='bad', text='fabricated quote')], scorer,
        {'museum': (POLICY, receipt, payload)}, PipelineConfig(minimum_families=1))
    assert calls == ['Museum collection electric motor.']
    assert [row['id'] for row in result['selected_evidence']] == ['a']
    assert result['origin_rejected_evidence'][0]['id'] == 'bad'


def test_saved_live_audit_replays_and_detects_changed_file(tmp_path):
    import shutil
    from audit_source_origin import OUTPUT, verify
    summary = verify()
    assert summary['attempted_sources'] == 3
    shutil.copytree(OUTPUT, tmp_path / 'audit')
    (tmp_path / 'audit' / 'summary.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='file changed'):
        verify(tmp_path / 'audit')


def test_saved_audit_rejects_false_counts_even_with_recomputed_file_hash(tmp_path):
    import shutil
    from audit_source_origin import OUTPUT, digest, verify
    shutil.copytree(OUTPUT, tmp_path / 'audit')
    path = tmp_path / 'audit' / 'summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    summary['captured_sources'] += 1
    path.write_text(json.dumps(summary), encoding='utf-8')
    manifest_path = tmp_path / 'audit' / 'audit_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files']['summary.json'] = digest(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='counts'):
        verify(tmp_path / 'audit')
