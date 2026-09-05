# Phase 1B update: exact Windows installation steps

This update belongs inside your existing project folder:

`D:\apv-rag-tesla369`

Do not create `D:\apv-rag-tesla369\apv-rag-tesla369`. The ZIP contains paths such
as `data`, `docs`, `schemas`, `scripts`, `src`, and `tests`; those folders must merge
with the same folders already visible in your screenshot.

## 1. Download the update

Download `APV_RAG_Phase1B_Update_v1.zip`. Windows normally saves it here:

`C:\Users\YOUR_WINDOWS_NAME\Downloads\APV_RAG_Phase1B_Update_v1.zip`

You do not need to move the ZIP before extracting it.

## 2. Extract into the correct folder

1. Open **File Explorer**.
2. Open **Downloads**.
3. Right-click `APV_RAG_Phase1B_Update_v1.zip`.
4. Select **Extract All**.
5. In the destination box, type exactly:

   `D:\apv-rag-tesla369`

6. Select **Extract**.
7. If Windows asks to merge folders, approve the merge.
8. If Windows opens **Replace or Skip Files**, select **Replace the files in the
   destination**. The update intentionally refreshes `README.md`, `data\README.md`,
   and `docs\DECISION_LOG.md`; your screenshot showed the repository was clean
   before this update.

This update adds new files and updates project documentation. It does not replace
your frozen Phase 1A claim or search-plan CSV files.

## 3. Confirm the folder structure

Open:

`D:\apv-rag-tesla369`

Confirm these exact files exist:

```text
D:\apv-rag-tesla369\data\pilot\tesla_phase1b_evidence_v0_1.csv
D:\apv-rag-tesla369\data\pilot\tesla_phase1b_search_log_v0_1.csv
D:\apv-rag-tesla369\data\pilot\tesla_phase1b_verdicts_v0_1.csv
D:\apv-rag-tesla369\data\pilot\tesla_phase1b_provenance_edges_v0_1.csv
D:\apv-rag-tesla369\data\pilot\phase1b_manifest_v0_1.sha256
D:\apv-rag-tesla369\docs\PHASE_1B_PILOT_REPORT.md
D:\apv-rag-tesla369\scripts\validate_phase1b.py
D:\apv-rag-tesla369\src\apv_rag\phase1b.py
D:\apv-rag-tesla369\tests\test_phase1b.py
```

If you instead see a second project folder nested inside the first, stop and move
the inner folder's contents up one level before running commands.

## 4. Open the correct folder in VS Code

1. Open VS Code.
2. Select **File -> Open Folder**.
3. Select `D:\apv-rag-tesla369`.
4. Select **Select Folder**.
5. Select **Terminal -> New Terminal**.

The prompt must be:

```powershell
PS D:\apv-rag-tesla369>
```

If it is different, run:

```powershell
Set-Location D:\apv-rag-tesla369
```

## 5. Validate Phase 1B

Run these commands one at a time:

```powershell
.\.venv\Scripts\python.exe scripts\validate_phase1b.py
.\.venv\Scripts\python.exe -m pytest -q
```

Expected validator summary:

```text
Phase 1B validation passed.
Pilot claims: 5
Evidence records: 20 across 10 provenance families
Executed searches: 35
Provenance edges: 11
Provisional outcomes: 1 authenticated; 4 insufficient
Scientific status: machine-assisted pre-annotations only; no gold labels assigned.
Human review required for every claim; bilingual review required for T369-004.
```

The full suite should report `13 passed`.

## 6. Read the results in this order

1. `docs\PHASE_1B_PILOT_REPORT.md`
2. `docs\PHASE_1B_HUMAN_REVIEW.md`
3. `data\pilot\tesla_phase1b_verdicts_v0_1.csv`
4. `data\pilot\tesla_phase1b_provenance_edges_v0_1.csv`

Do not edit the frozen file:

`data\pilot\tesla_search_plan_v0_1.csv`

Do not call the provisional verdicts gold labels. The next research action is two
independent human annotations, with bilingual review for `T369-004`.

## 7. Commit and push

Only after both validation commands pass, run:

```powershell
git status --short
git add README.md docs data\pilot schemas scripts\validate_phase1b.py src\apv_rag\phase1b.py tests\test_phase1b.py
git commit -m "Add Phase 1B archival verification pilot"
git push
git status
```

The final `git status` should say that your branch is up to date and the working
tree is clean. Send a screenshot of the two validation results and final status
before we begin human annotation.
