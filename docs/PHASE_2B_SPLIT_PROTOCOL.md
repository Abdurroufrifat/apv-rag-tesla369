# Phase 2B: leakage-safe AVeriTeC partition

## Purpose

The official AVeriTeC development set remains untouched and evaluation-only.
Model development uses a new partition derived only from the official training
split.

## Group construction

Two training records are placed in the same connected component when they share:

1. the same normalized claim text; or
2. the same normalized fact-checking article URL.

The transitive closure is used. For example, if records A and B share a claim and
B and C share an article, A, B, and C remain in one group. This prevents exact
claim duplication and shared-article evidence from crossing the internal split.

The single known empty training claim is excluded and recorded by upstream index.
No original label or source file is changed.

## Assignment

Connected groups are assigned with seed `369`. A deterministic 5,000-candidate
group-level search selects the partition closest to the 80/20 size and four-label
targets. Assignment occurs at group level, never record level.

## Generated files

The command writes index files and a manifest below:

```text
data/processed/averitec/phase2b_split_v0_1/
```

The index files refer to positions in the immutable official `train.json`. They do
not duplicate or rewrite the upstream dataset.

## Windows commands

From `D:\apv-rag-tesla369` run:

```powershell
.\.venv\Scripts\python.exe scripts\build_averitec_splits.py
.\.venv\Scripts\python.exe scripts\validate_averitec_splits.py
.\.venv\Scripts\python.exe -m pytest -q
```

Do not open, reorder, or edit the generated JSON files.
