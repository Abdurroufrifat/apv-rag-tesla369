# Phase 1C installation and validation

> **Archived:** The machine-only protocol adopted on 2026-09-08 retires this
> workflow. Do not generate or distribute reviewer packets. Continue with
> `docs/MACHINE_ONLY_PROTOCOL.md` and `scripts/validate_phase2.py`.

## Correct destination

Extract the Phase 1C update ZIP directly into:

```text
D:\apv-rag-tesla369
```

Do not extract it into `docs`, `data`, `scripts`, or a second nested
`apv-rag-tesla369` folder. Allow Windows to merge folders and replace the three
updated documentation files when prompted. The ZIP does not contain or overwrite
the private newspaper screenshot.

## Validate from VS Code

Open a PowerShell terminal whose prompt is:

```text
PS D:\apv-rag-tesla369>
```

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\validate_phase1c.py --require-private-capture
.\.venv\Scripts\python.exe -m pytest -q
```

The first command must report that private capture integrity is verified. The
second command must pass all tests.

## Generate reviewer packets

Only after validation passes, run:

```powershell
.\.venv\Scripts\python.exe scripts\create_phase1c_packets.py
```

This creates:

```text
artifacts\phase1c_blinded\Reviewer_A_T369-004_blinded.zip
artifacts\phase1c_blinded\Reviewer_B_T369-004_blinded.zip
```

These files are private working artifacts and are ignored by Git. Send each ZIP only
to its assigned bilingual reviewer.
