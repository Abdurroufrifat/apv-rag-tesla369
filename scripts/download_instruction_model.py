"""Download the pinned Qwen2.5-1.5B-Instruct generative baseline."""

from pathlib import Path

from apv_rag.nli_comparison import _model_files
from apv_rag.splits import write_json_atomic

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def main():
    from huggingface_hub import snapshot_download

    root = Path(__file__).resolve().parents[1]
    destination = root / "models/qwen2.5-1.5b-instruct"
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    write_json_atomic(
        root / "models/instruction_model_manifest.json",
        {"model_id": MODEL_ID, "revision": REVISION, "files": _model_files(destination)},
    )
    print("Pinned generative model downloaded and checksummed.")


if __name__ == "__main__":
    main()
