"""Fixed-origin HTTPS collection and capture-bound evidence admission.

Unsigned research receipts assume a trusted collector. They are neither TLS
notarization nor authentication of historical statements, authors or dates.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
import re
import ssl
import urllib.request
from urllib.parse import urlsplit

from apv_rag.source_pipeline import verify_claim


class CaptureError(ValueError):
    pass


def digest(data):
    return sha256(data).hexdigest()


@dataclass(frozen=True)
class SourcePolicy:
    key: str
    url: str
    anchors: tuple[str, ...]
    evidence_role: str
    redirect_urls: tuple[str, ...] = ()

    def validate(self):
        if not self.key or not self.anchors or any(not a.strip() for a in self.anchors):
            raise CaptureError('Nonempty source key and content anchors required')
        if self.evidence_role not in ('catalogue_metadata', 'institutional_navigation'):
            raise CaptureError('This protocol supports metadata/navigation only')
        host = safe_url(self.url)
        for url in self.redirect_urls:
            if safe_url(url) != host:
                raise CaptureError('Redirect policy must preserve the exact HTTPS host')

    @property
    def fingerprint(self):
        return digest(json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode())

    @property
    def allowed_urls(self):
        return (self.url, *self.redirect_urls)


def safe_url(url):
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == 'https' and parsed.hostname and
                 parsed.username is None and parsed.password is None and
                 parsed.port in (None, 443) and not parsed.fragment and
                 not any(ord(c) < 33 for c in url) and '\\' not in url)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise CaptureError('Only credential-free HTTPS URLs on port 443 are allowed')
    return parsed.hostname


class OriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, policy):
        self.policy = policy
        self.chain = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url(newurl)
        if newurl not in self.policy.allowed_urls or len(self.chain) >= 5:
            raise CaptureError('Redirect is outside the fixed source policy')
        self.chain.append({'from': req.full_url, 'to': newurl, 'status': code})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def normalize(text):
    return ' '.join(text.split())


class VisibleText(HTMLParser):
    hidden = {'script', 'style', 'noscript', 'template'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.suppressed = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.hidden:
            self.suppressed += 1

    def handle_endtag(self, tag):
        if tag in self.hidden and self.suppressed:
            self.suppressed -= 1

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)


def visible_text(payload, content_type):
    if content_type.split(';')[0].strip().lower() != 'text/html':
        raise CaptureError('Expected HTML page, not a substituted binary or JSON response')
    match = re.search(r'charset\s*=\s*["\x27]?([\w-]+)', content_type, re.I)
    charset = match.group(1) if match else 'utf-8'
    try:
        parser = VisibleText()
        parser.feed(payload.decode(charset, errors='strict'))
        parser.close()
        return normalize(' '.join(parser.parts))
    except (UnicodeError, LookupError) as exc:
        raise CaptureError('Page encoding could not be decoded strictly') from exc


def capture_source(policy, *, timeout=15, max_bytes=2_000_000, opener=None):
    """Collect one fixed page. ``opener`` is a synthetic unit-test transport hook.

    TLS fields record client verification configuration, not a stored certificate
    chain or independently verifiable proof of the server exchange.
    """
    policy.validate()
    if timeout <= 0 or max_bytes < 1:
        raise CaptureError('Positive timeout and byte limit required')
    context = ssl.create_default_context()
    redirects = OriginRedirectHandler(policy)
    client = opener or urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context), redirects).open
    request = urllib.request.Request(policy.url, headers={
        'User-Agent': 'APV-RAG-Source-Capture/1.0', 'Accept': 'text/html',
        'Accept-Encoding': 'identity'})
    with client(request, timeout=timeout) as response:
        final_url = response.geturl()
        if response.status != 200 or final_url not in policy.allowed_urls:
            raise CaptureError('Successful response on an allowed URL required')
        content_type = response.headers.get('Content-Type', '')
        payload = response.read(max_bytes + 1)
    if not payload or len(payload) > max_bytes:
        raise CaptureError('Empty or oversized capture')
    text = visible_text(payload, content_type)
    if any(normalize(a).casefold() not in text.casefold() for a in policy.anchors):
        raise CaptureError('Source identity anchors missing from page text')
    receipt = {
        'protocol': 'source_origin_v1', 'source_key': policy.key,
        'requested_url': policy.url, 'final_url': final_url,
        'redirect_chain': redirects.chain, 'policy_sha256': policy.fingerprint,
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'status': 200, 'content_type': content_type, 'payload_bytes': len(payload),
        'payload_sha256': digest(payload), 'normalized_text_sha256': digest(text.encode()),
        'evidence_role': policy.evidence_role,
        'transport': {'certificate_verification': context.verify_mode == ssl.CERT_REQUIRED,
                      'hostname_verification': context.check_hostname,
                      'method': 'synthetic_test_transport' if opener else 'urllib_verified_https'},
    }
    return receipt, payload


def check_binding(document, policy, receipt, payload):
    """Replay bytes and policy; no network, truth judgment or OCR is performed."""
    try:
        policy.validate()
        if (receipt['protocol'] != 'source_origin_v1' or
            receipt['source_key'] != policy.key or document['source_key'] != policy.key or
            document['source_url'] != policy.url or receipt['requested_url'] != policy.url or
            receipt['final_url'] not in policy.allowed_urls or
            receipt['policy_sha256'] != policy.fingerprint or receipt['status'] != 200 or
            receipt['transport']['certificate_verification'] is not True or
            receipt['transport']['hostname_verification'] is not True or
            receipt['evidence_role'] != policy.evidence_role or
            document.get('evidence_role') != policy.evidence_role):
            raise CaptureError('Source, role, policy or transport binding mismatch')
        current = policy.url
        chain = receipt['redirect_chain']
        if len(chain) > 5:
            raise CaptureError('Redirect chain too long')
        for hop in chain:
            if (hop['from'] != current or hop['to'] not in policy.allowed_urls or
                hop['status'] not in (301, 302, 303, 307, 308)):
                raise CaptureError('Invalid recorded redirect chain')
            current = hop['to']
        if current != receipt['final_url']:
            raise CaptureError('Final URL does not match recorded redirects')
        if (not payload or len(payload) != receipt['payload_bytes'] or
            digest(payload) != receipt['payload_sha256']):
            raise CaptureError('Captured page bytes changed')
        text = visible_text(payload, receipt['content_type'])
        if digest(text.encode()) != receipt['normalized_text_sha256']:
            raise CaptureError('Normalized text binding mismatch')
        if any(normalize(a).casefold() not in text.casefold() for a in policy.anchors):
            raise CaptureError('Missing source identity anchors')
        excerpt = normalize(document['text'])
        if not excerpt or excerpt not in text:
            raise CaptureError('Evidence excerpt not present in captured page text')
        return {'id': document['id'], 'bound': True, 'source_key': policy.key,
                'payload_sha256': receipt['payload_sha256'], 'role': policy.evidence_role}
    except (CaptureError, KeyError, TypeError, AttributeError) as exc:
        return {'id': document.get('id'), 'bound': False, 'reason': str(exc)}


def guarded_verify_claim(claim, claim_language, documents, scorer, captures, config=None):
    admitted, rejected, bindings = [], [], []
    for document in documents:
        capture = captures.get(document.get('source_key'))
        result = (check_binding(document, *capture) if capture else
                  {'id': document.get('id'), 'bound': False, 'reason': 'missing_capture'})
        if result['bound']:
            admitted.append(document)
            bindings.append(result)
        else:
            rejected.append(result)
    result = verify_claim(claim, claim_language, admitted, scorer, config)
    if rejected and not admitted:
        result['abstention_reasons'].append('no_origin_bound_evidence')
    result.update(origin_bindings=bindings, origin_rejected_evidence=rejected,
                  origin_qualification='Trusted-collector HTTPS page/metadata binding only; '
                  'no authenticated historical attribution, author, primary status or gold verdict. '
                  'Unsigned receipts cannot independently attest past TLS exchanges.')
    return result
