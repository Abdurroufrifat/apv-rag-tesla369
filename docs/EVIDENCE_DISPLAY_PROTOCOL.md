# Exact cited-excerpt evidence displays

This bounded change addresses missing citations and generated unsupported
references in the existing English evidence/explanation flow. It adds a
deterministic evidence display rather than rewriting generated rationales or
treating NLI scores as factual truth. No new inference or human labels are used.

The renderer accepts the canonical claim, existing candidate, retrieved passages,
collapsed context and pinned benchmark corpus. Gold labels, free-form explanations
and model confidence scores are excluded. It reuses the existing claim-specific
sentence selection and nonempty prefix binding in `source_snapshot_guard`, then
requires the collapsed context to equal the original family-collapse result.
Every retrieved passage must bind. Upstream abstention is never reversed.

An admitted candidate gets a structured JSON evidence packet. Each item has an
explicit citation document ID, literal excerpt, original selected-sentence
indices, excerpt SHA-256 and canonical full-document SHA-256. The displayed
candidate is still a machine candidate. Each packet states that the relationship
between evidence and verdict is unverified. No free-form rationale enters the
packet. Evidence text can itself contain source quotations, citations or URLs;
those remain literal source text and are not authenticated external references.

The audit uses all four saved policies on the 600 original SciFact/climate claim
contexts and 360 stress contexts: 3,840 policy records, 960 contexts. It uses the
existing raw-corpus checksums, saved-output receipts and prior verifier input
bindings. It does not select successful records or tune a threshold. Source
predictions, caches and reported benchmark results remain unchanged. Counts of
displayed candidates and withheld displays are admission diagnostics, not accuracy
or explanation-quality measurements. Shared policy records are not independent.

Run the shipped audit offline from the existing Windows project directory:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\audit_evidence_display.py --verify
```

The verifier rechecks source/code/protocol hashes and reconstructs every packet,
summary count and report. A citation ID alone is insufficient; modified excerpt
text, indices, document hashes or candidate status fail replay. `--build` creates
a separate audit and refuses to overwrite an existing frozen receipt.

Limits remain explicit. Exact snapshot text is not publisher authentication.
Passages may be clipped or stitched in the model's original selected order and
can be incomplete or irrelevant. A donor context can pass if it happens to match
the existing claim-specific selection rule. OCR edits and paraphrases do not
match the strict rule. Showing a cited excerpt does not prove that it entails a
Supported candidate, contradicts a Refuted candidate, or justifies evidence
absence for Not Enough Evidence. This packet does not supply a certified
natural-language explanation. The prior 840-explanation/2,520-pair diagnostic is
unchanged and has not become a truth checker. Semantic explanation truth remains
open. No manuscript, GitHub push or new human review is part of this stage.
