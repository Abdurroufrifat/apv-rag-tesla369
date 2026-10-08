# Independent bilingual review instructions: T369-004

> **Archived:** Do not use or distribute this guide. The project now follows the
> machine-only protocol in `docs/MACHINE_ONLY_PROTOCOL.md`.

## Your task

Independently assess whether a Serbian newspaper interview materially supports the
following English attribution:

> Nikola Tesla said: The present is theirs; the future, for which I really worked,
> is mine.

Do not consult another reviewer, a completed annotation, a fact-check verdict, or a
summary of this project's conclusion before freezing your form.

## Source to inspect

- Repository: National Library of Serbia digital archive
- Newspaper: *Politika*
- Date: 27 April 1927
- Article: `Посета г. Николи Тесли`
- Printed page: 2
- Column: 1, the leftmost column
- Stable identifier: `URN:NB:RS:SD_2F6F6602455A67B1B521D786232CBF4A-1927-04-27`
- URL: <https://digitalna.nb.rs/view/URN:NB:RS:SD_2F6F6602455A67B1B521D786232CBF4A-1927-04-27>

Inspect the original scan in the archive viewer. The page and column are finding
aids, not a supplied transcription.

## Complete the CSV

Open the CSV included in your packet and complete every blank field.

1. Use the assigned anonymous `reviewer_code`; do not enter your name or email.
2. Confirm your Serbian-English qualification and independent work.
3. Transcribe the decisive Serbian paragraph exactly as printed. Preserve words,
   punctuation, and apparent historical spelling; mark unreadable characters in
   square brackets and explain them in `limitations`.
4. Write a literal English translation that follows the Serbian wording closely.
5. Write a separate normalized English rendering that reads naturally without
   adding ideas absent from the source.
6. Assign `translation_equivalence`:
   - `exact`: the supplied English claim matches both wording and meaning closely;
   - `materially_equivalent`: wording differs, but the central attributed meaning is
     supported;
   - `non_equivalent`: the source does not support the central meaning;
   - `uncertain`: scan or language ambiguity prevents a dependable decision.
7. Assign an attribution `verdict`:
   - `authenticated`: sufficient evidence supports the attribution or its explicitly
     identified materially equivalent translation;
   - `misattributed`: sufficient evidence assigns it to a different origin or speaker;
   - `contradicted`: sufficient direct evidence conflicts with the attribution;
   - `insufficient`: the inspected evidence cannot justify a definitive label.
8. Use `sufficient` for the first three verdicts and `insufficient` only for the
   `insufficient` verdict.
9. Give confidence from `0` to `1`, identify decisive evidence, and write a rationale
   of at least 80 characters.
10. Set `frozen=true` only after the record is complete and add a UTC timestamp.

## Important distinction

An English translation can be materially authentic without being Tesla's verbatim
English wording. State clearly whether your finding concerns the Serbian meaning,
a literal translation, or the familiar normalized English sentence.

Automated translation may be consulted only as an aid and must be disclosed in
`limitations`. Your transcription, translation, and label must reflect your own
bilingual judgment.
