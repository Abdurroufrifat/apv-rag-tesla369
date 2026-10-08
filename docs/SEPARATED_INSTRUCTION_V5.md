# Separated instruction interface

Diagnostics showed all five exact instruction probes passed and chat token sequences matched. The combined verdict/explanation/citation request failed. This separate smoke interface uses the same model and direct chat-template tensors, a short verdict-only query, then a separate explanation query. It does not reinterpret earlier malformed outputs.

The application records the supplied passage ID as context provenance, not as a model-selected or semantically verified citation. The interface is restricted to one passage per synthetic fixture; source selection in multi-document RAG remains open. Empty evidence bypasses generation and abstains. Reject unknown verdicts, empty explanations and numbers absent from evidence. All fixture verdicts and structural/numeric checks must pass for interface readiness; explanation entailment is still unverified. These adaptively reused fixtures do not establish research performance. No full benchmark is authorized by the readiness flag alone.

Reuse existing Qwen model, CPU float32, four threads, seed 369 and greedy generation with 128 new tokens per call. Preserve prompts and generated fields. No new download, no manuscript.

Run python scripts\smoke_test_separated_instruction.py and export artifacts/separated_instruction_smoke_v5/*.json.
