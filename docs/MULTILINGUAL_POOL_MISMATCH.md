# Language-pool mismatch admission check

This optional wrapper checks whether each prepared excerpt is a prefix of its
source record and whether the complete source excerpt's ID and text occur in
that record's pinned XFEVER file-specific pool. The file prefix must match the
declared language. The existing multilingual controller is called only if the
check passes; its features, model scores, thresholds and prompts are unchanged.

The audit reads the verified clean multilingual confirmation's 600 prepared
contexts and the pinned model-only inputs and six excerpt pools. It binds all
600 originals. For each underlying claim ID, it substitutes the prepared
evidence from the next language variant in a fixed cycle, leaving the target
claim unchanged. All 600 substitutions fail the target pool check, and the
wrapper emits four abstentions per refused context with zero generation calls.
Saved rows, input/code/output hashes and a complete deterministic replay live
in `artifacts/multilingual_pool_mismatch_v1`.

Run after extracting the update into `D:\apv-rag-tesla369`:

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\audit_multilingual_pool_mismatch.py --verify
```

The result measures only exact identity relative to the closed benchmark pools.
It is not a language detector, publisher check, proof of claim relevance, or
accuracy measurement. An identical excerpt present in two pools can pass both.
The pools contain benchmark-selected target material. The original 2,400
policy outputs and confirmation results remain unchanged; the wrapper is
separate and has no new neural inference. No label is read for the audit, and
no manuscript or GitHub push is part of this step.
