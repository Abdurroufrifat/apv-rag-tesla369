# Phase 1C coordinator guide: blinded bilingual review

> **Archived:** Human review is no longer part of this project. This file is
> retained only as an audit record of the superseded protocol.

## Purpose

Phase 1C converts the `T369-004` source lead into defensible human annotation.
The research lead acts as coordinator. Two Serbian-English bilingual reviewers
work independently, and neither may see the Phase 1B machine verdict or the
other review before both files are frozen.

This phase does **not** create a gold label automatically. Gold status is allowed
only after two completed reviews, an agreement calculation, and documented
adjudication of every disagreement.

## Roles

- **Coordinator:** creates and distributes separate blinded packets, receives the
  frozen forms, checks hashes, and keeps the reviewers separated.
- **Reviewer A:** independently inspects the original Serbian scan and completes
  packet A.
- **Reviewer B:** independently inspects the same source and completes packet B
  without seeing A's work.
- **Adjudicator:** compares the two frozen records afterward. The adjudicator must
  not resolve their own disputed annotation alone.

Reviewer names and contact information must not be placed in the dataset. Assign
anonymous codes such as `RA-7K2M` and `RB-9P4Q` and keep any identity key outside
the repository.

An AI system, machine translation system, or chatbot may assist with navigation
or produce a candidate translation, but it cannot count as a human annotator,
bilingual reviewer, or independent agreement observation.

## Private source capture

The coordinator's screenshot remains at:

```text
data/raw/T369-004/politika_1927-04-27_p2_col1_target.png
```

It is intentionally ignored by Git. The repository records only its metadata and
SHA-256 fingerprint in
`data/pilot/tesla_phase1c_source_register_v0_1.csv`. Do not add the scan with
`git add -f`, publish it, or place it inside a reviewer ZIP.

## Create the two packets

From the project root on Windows:

```powershell
.\.venv\Scripts\python.exe scripts\validate_phase1c.py --require-private-capture
.\.venv\Scripts\python.exe scripts\create_phase1c_packets.py
```

The second command creates two separate ZIP files under
`artifacts/phase1c_blinded/`. The whole `artifacts/` directory is ignored by Git.

Send packet A only to Reviewer A and packet B only to Reviewer B. Do not send the
whole repository, the Phase 1B pilot report, the source register, or any provisional
verdict table to either reviewer.

## Required reviewer qualifications

For this record, both reviewers must be able to read Serbian Cyrillic and translate
Serbian into English. Each reviewer must personally inspect the archive scan rather
than approve a supplied translation. Record the qualification category, not a CV or
personal details, in the review form.

Acceptable qualification evidence can include native or near-native Serbian plus
working English, formal Serbian-English translation experience, or a relevant
language/historical-research background. The paper must state the operational
qualification rule used.

## Freeze procedure

Each reviewer must:

1. complete every field;
2. set `independent_review_confirmed` to `true`;
3. set `frozen` to `true` only when finished;
4. provide a timezone-aware UTC timestamp such as `2026-09-05T08:30:00Z`;
5. save the CSV without changing the header; and
6. return only their completed CSV.

After receiving both files, store them in separate private folders and compute a
SHA-256 hash for each. Do not edit a frozen file. If a correction is necessary,
retain the original and create a new version with an audit note.

## What is and is not established now

The original issue, article, page, column, and evidence-capture hash are documented.
The exact Serbian transcription, literal English translation, normalized English
rendering, equivalence decision, and final attribution label are still pending human
review. Until then, no paper table or model-training file may call this record gold.
