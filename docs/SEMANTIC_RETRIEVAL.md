# Semantic retrieval comparison

Uses the existing pinned DeBERTa NLI model offline. Retrieval-only experiment on 46 previously observed claims. Rank top10 BM25 candidates by maximum contradiction/entailment probability; retain3. Labels are used only for final evidence article recall. NLI scores do not authenticate sources. No Qwen generation, model download or training.

Extract into D:\apv-rag-tesla369. Run .\.venv\Scripts\python.exe scripts\run_semantic_retrieval.py. Interrupted inference resumes under matching input/model hashes. Send semantic_retrieval_outputs.zip from the project root.

Local verification:46claim-only preflight inputs passed; Python compilation passed. Neural inference untested here because model weights are unavailable. Last full suite360passed before this runner was added.
