# Phase 1A update: exact Windows installation steps

This update is designed for the existing project at:

`D:\apv-rag-tesla369`

Do not make a second project folder. The ZIP contains only new Phase 1A files,
arranged so they merge into the existing project.

## 1. Download and extract the update

1. Download `APV_RAG_Phase1A_Update_v1.zip` to the normal Windows Downloads
   folder.
2. In File Explorer, right-click the ZIP and select **Extract All**.
3. For the destination, enter exactly:

   `D:\apv-rag-tesla369`

4. Select **Extract**. If Windows asks whether to merge folders, approve the
   merge. The update does not replace the original starter records.

After extraction, confirm that this file exists:

`D:\apv-rag-tesla369\docs\PHASE_1A_SEARCH_PROTOCOL.md`

## 2. Open the correct folder in VS Code

1. Open VS Code.
2. Select **File -> Open Folder**.
3. Choose exactly `D:\apv-rag-tesla369` and select **Select Folder**.
4. Select **Terminal -> New Terminal**.
5. Check that the prompt starts with:

   `PS D:\apv-rag-tesla369>`

If it does not, run:

```powershell
Set-Location D:\apv-rag-tesla369
```

## 3. Validate the update

Run these commands one at a time:

```powershell
.\.venv\Scripts\python.exe scripts\validate_phase1a.py
.\.venv\Scripts\python.exe -m pytest -q
```

The first command should report 20 canonical claim candidates, 16 evidence
seeds, and 20 frozen search plans. The full test suite should report 7 passed.

## 4. Review, but do not relabel, the records

Open these files in VS Code:

- `docs\GAP_AUDIT_2026-09-05.md`
- `docs\PHASE_1A_SEARCH_PROTOCOL.md`
- `docs\ANNOTATION_GUIDE_V0_1.md`
- `data\pilot\tesla_claims_pilot_v0_1.csv`
- `data\pilot\tesla_evidence_seed_v0_1.csv`
- `data\pilot\tesla_search_plan_v0_1.csv`

All 20 records are deliberately marked as candidates with `insufficient`
evidence. Do not convert them into gold labels yet. Phase 1B will execute the
frozen searches and perform human evidence review before any label is changed.

## 5. Commit and push the Phase 1A update

Run these commands one at a time:

```powershell
git status --short
git add data\pilot docs schemas\evidence_record.schema.json src\apv_rag\evidence.py scripts\validate_phase1a.py tests\test_phase1a.py
git commit -m "Add Phase 1A claim and provenance protocol"
git push
git status
```

The final command should report `nothing to commit, working tree clean`.

## What comes next

Phase 1B will verify the first five high-risk Tesla 3-6-9 attribution claims in
primary archives and record every query, archive, result, and negative search.
