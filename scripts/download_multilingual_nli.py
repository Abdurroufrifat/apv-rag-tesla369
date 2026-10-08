"""Download a pinned multilingual control model and record file integrity."""
import json
from pathlib import Path

from apv_rag.nli_comparison import _model_files
from apv_rag.splits import write_json_atomic

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
REVISION = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"


def main():
    from huggingface_hub import snapshot_download

    root = Path(__file__).resolve().parents[1]
    destination = root / "models/mdeberta-multilingual-nli"
    snapshot_download(
        repo_id=MODEL_ID, revision=REVISION, local_dir=destination,
        allow_patterns=["config.json", "model.safetensors", "spm.model",
                        "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "added_tokens.json"],
    )
    mapping = json.loads((destination / "config.json").read_text())["id2label"]
    if {str(v).lower() for v in mapping.values()} != {
        "contradiction", "entailment", "neutral"
    }:
        raise ValueError("Unexpected model labels")
    write_json_atomic(root / "models/multilingual_nli_manifest.json", {
        "model_id": MODEL_ID, "revision": REVISION,
        "files": _model_files(destination),
    })
    print("Pinned multilingual model downloaded and checksummed.")


if __name__ == "__main__":
    main()
