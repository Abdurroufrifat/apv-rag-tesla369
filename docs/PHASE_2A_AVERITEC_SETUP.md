# Phase 2A: official AVeriTeC setup

## Scientific purpose

AVeriTeC is the primary quantitative benchmark. Its published annotations supply
the ground truth, so this project does not create human labels. The upstream
repository is pinned to commit `7c62d1ec8df3fb560d6efe2b85fa191135636f81`.

Official source: <https://github.com/MichSchli/AVeriTeC>

License: Creative Commons Attribution-NonCommercial 4.0. Use the files for this
noncommercial research project, cite the dataset paper, and do not repackage the
dataset in this repository.

## Windows commands

Open VS Code at `D:\apv-rag-tesla369`. In its PowerShell terminal run:

```powershell
.\.venv\Scripts\python.exe scripts\download_averitec.py
.\.venv\Scripts\python.exe scripts\validate_averitec.py
```

The downloader writes only to:

```text
D:\apv-rag-tesla369\data\external\averitec\official_7c62d1e
```

It downloads `train.json` and `dev.json` atomically, checks their SHA-256 hashes,
validates record counts and labels, and writes `dataset_manifest.json`. A valid
existing file is reused; a mismatched file is never silently overwritten.

The pinned upstream snapshot contains one empty training claim and repeated exact
claim texts (69 extra train occurrences and 9 extra development occurrences).
The files remain immutable. These known issues are recorded in the manifest and
will be handled by a frozen grouping/exclusion policy in Phase 2B.

## Frozen expectations

| Split | Records | SHA-256 |
|---|---:|---|
| train | 3,068 | `ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c` |
| dev | 500 | `499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300` |

The official repository does not include a labeled test file in its `data`
directory. We therefore preserve `dev.json` as evaluation-only and do not tune on
it. Phase 2B will derive a grouped validation partition from `train.json` using a
frozen script before any model fitting.
