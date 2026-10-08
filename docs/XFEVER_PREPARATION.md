# XFEVER multilingual input preparation

Download the official data archive only; pretrained checkpoint archives are not needed for this preparation stage. The published archive is approximately 403 MB. The downloader resumes partial downloads when the server supports byte ranges and otherwise restarts safely. It checks the release MD5 and records SHA-256, then inventories file names without extracting paths or opening label records.

```powershell
cd D:\apv-rag-tesla369
.\.venv\Scripts\python.exe scripts\prepare_xfever.py
```

After completion, provide `download_manifest.json` and `archive_inventory.json` from `data\external\xfever\zenodo_8206962`. Do not upload the large archive. Keep it locally for the evaluation runner. A network interruption can be resumed by rerunning the same command. If checksum validation fails after a complete download, do not edit the expected checksum to force acceptance.

Source: https://zenodo.org/records/8206962
Published data.tar.gz MD5: 8521ac9572a50a24b35b8a65e3e8abbb
Project: https://github.com/nii-yamagishilab/xfever
Paper: https://aclanthology.org/2023.rocling-1.1/

XFEVER includes translated claim/evidence texts. Translation origin and split must be recorded in the evaluation design. The current local English NLI model is not a multilingual model; applying it to foreign-language inputs would be a transfer stress test, not a claim of multilingual capability. A multilingual model comparison requires a separately pinned model. Do not tune on target evaluation labels. Neither this download nor earlier language-tag checks completes multilingual evaluation. Manuscript work remains paused.
