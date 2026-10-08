# Fixed institutional source-origin protocol

This stage adds an admission layer before the existing provenance/NLI pipeline.
It implements a bounded part of the original source-authentication requirement.
It does not create historical gold labels or authenticate a Tesla quotation.

## Fixed collection design

The three URLs in `config/source_origin_v1.json` come from the existing Tesla
source register and evidence seed: the National Library of Serbia Politika issue
catalogue, the Library of Congress Tesla guide, and the Nikola Tesla Museum life
and work page. Only catalogue metadata and institutional navigation are allowed.
Each source requires fixed identifying text anchors. The policy permits no
redirect URLs; any redirect outside that exact policy is refused before follow.
The code supports explicitly listed redirects on the same HTTPS host for future
separate protocols. This protocol was not widened to accommodate failed requests.

Collection uses the Python default certificate trust store, hostname verification,
HTTPS on port 443, a 15-second request timeout and a 2 MB response limit. It rejects
credentials, fragments, malformed URLs, foreign redirects, non-200 responses,
unexpected media types, empty/oversized responses and missing page anchors.
Raw HTML and unsigned collection receipts retain the URL, timestamp, redirect
chain, transport configuration, policy fingerprint and SHA-256 of original bytes.
Collection does not use web-search snippets as captured source evidence.

The text extractor removes script, style, noscript and template blocks and
normalizes whitespace. It does not resolve iframe content, OCR images, execute
JavaScript, determine CSS visibility, translate text or paraphrase excerpts.
The declared source URL must match its configured source, the raw bytes must
match their receipt, and the evidence role must match the allowed metadata or
navigation role. An admitted excerpt must occur literally in extracted text,
allowing whitespace normalization. OCR edits and paraphrases will be rejected;
this deliberate strictness must not be reported as semantic verification.

`guarded_verify_claim` passes admitted documents to the unchanged `verify_claim`
pipeline. Missing or mismatched documents cannot reach the NLI scorer. Mixed pools
can score the admitted subset, subject to the original family/confidence rules.
Outputs remain `machine_candidate` or `abstain`. The guard's origin qualification
does not upgrade declared source ranks, primary flags or family independence.

## Verification and use

Extract the update into the existing `D:\apv-rag-tesla369` directory. Replay the
shipped audit without network access or model inference:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\audit_source_origin.py --verify
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
.\.venv\Scripts\python.exe -m pytest -q
```

The offline verifier checks exact file inventory, producer/policy/register hashes,
capture bytes, original excerpt binding and deterministic altered-context cases.
Eight rejected cases per page cover fabricated excerpts, spoofed source URLs,
wrong page identifiers, HTTP downgrade, changed payloads, historical-role
escalation, disabled certificate verification and missing capture. A benign
whitespace control passes. These cases are constructed admission tests, not
independent attacks, historical accuracy measurements or benchmark gold labels.

For a genuinely separate collection, use `--capture --output NEW_DIRECTORY`.
The shipped frozen audit cannot be overwritten through `--capture`. Network
failures are saved as inaccessible and must not count as authenticated pages.
The optional injected opener in the library function is a synthetic test hook;
such captures cannot count as real pages in the live audit.

## Trust boundary and remaining scope

The collector and its original frozen archive are trusted. Receipts are unsigned
research logs, not cryptographic TLS notarization. Hashes detect changes relative
to the stored manifest; they cannot detect coordinated replacement of captures,
receipts and manifest by a malicious operator. Default TLS verifies the server
exchange during collection; a replayed JSON assertion cannot independently prove
that past exchange. Content anchors identify the expected page, not its truth.
A compromised legitimate publisher, fabricated source statement or incorrect
catalogue date can pass an origin check. No publisher-honesty claim is made.

The Politika catalogue contains issue metadata, not the decisive newspaper text.
The private page-2 image and its original source-register hash remain unchanged.
Capturing the catalogue does not connect those image bytes to the library's
image service or validate a transcript. The museum page includes an external
timeline; that embedded resource is not implicitly captured. Linked guide
resources are also outside the saved HTML. Historical attribution authentication,
semantic explanation truth and general provenance efficacy remain open.

The implementation and capture diagnostics are completed. No new human review,
neural inference, manuscript or GitHub push is part of this step.
