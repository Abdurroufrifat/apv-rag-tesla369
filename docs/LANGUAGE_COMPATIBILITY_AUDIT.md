# Controlled language compatibility audit

Completed locally using the verified SciFact evidence traces. The run covers five scenarios over 300 claims each: matching English tags, mismatched evidence tags, mismatched claim tag, mixed tags, and no evidence. No new NLI inference is required.

Results are in `artifacts/language_compatibility_audit`. All incompatible-only and empty-evidence cases abstained without calling the scorer. No incompatible-tag document reached the selected evidence list. The matching-tag replay reproduced the earlier seven candidates.

The text was not translated. Serbian `sr` tags were deliberately assigned to unchanged English text as metadata controls. This tests the filtering contract only. It cannot establish multilingual understanding, language detection, or performance on Serbian claims. Removing evidence changes the averaged stance scores: the mixed control emitted thirteen candidates, which is not a measured accuracy improvement.

A genuine multilingual benchmark, including the planned XFEVER track, is still required. The missing-evidence scenario is distinct from naturally unavailable archival primary sources. These test results must not close those research requirements. Manuscript writing remains paused until explicit user approval.
