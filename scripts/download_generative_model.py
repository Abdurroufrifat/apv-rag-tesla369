"""Download the pinned FLAN-T5-base generative baseline."""

from pathlib import Path

from apv_rag.nli_comparison import _model_files
from apv_rag.splits import write_json_atomic

MODEL_ID = "google/flan-t5-base"
REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"


def main():
    from huggingface_hub import snapshot_download

    root = Path(__file__).resolve().parents[1]
    destination = root / "models/flan-t5-base"
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    write_json_atomic(
        root / "models/generative_model_manifest.json",
        {"model_id": MODEL_ID, "revision": REVISION, "files": _model_files(destination)},
    )
    print("Pinned generative model downloaded and checksummed.")


if __name__ == "__main__":
    main()
