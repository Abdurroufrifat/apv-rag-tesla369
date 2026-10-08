"""Extract only the official parallel XFEVER evaluation subset files."""

import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

LANGUAGES = ("es", "fr", "id", "ja", "zh")
EXPECTED_ARCHIVE_SHA256 = "8b7948894c8724d9a52e86e18fe0e369c5d58b12ddba853e836b8d80611a4895"


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/external/xfever/zenodo_8206962"
    archive = source / "data.tar.gz"
    if sha256(archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive differs from verified download receipt")
    output = source / "evaluation_inputs_v1"
    bundle = source / "XFEVER_evaluation_inputs.zip"
    if output.exists() or bundle.exists():
        raise FileExistsError("Evaluation inputs already exist; refusing overwrite")
    names = ["data/en/test.6h.jsonl"]
    names += [
        f"data/{language}/test.6h{suffix}.jsonl"
        for language in LANGUAGES
        for suffix in ("", ".human")
    ]
    contents = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if member.name not in names:
                continue
            if not member.isfile() or member.size > 2 * 1024 * 1024:
                raise ValueError("Unexpected evaluation member")
            if member.name in contents:
                raise ValueError("Duplicate archive member")
            contents[member.name] = handle.extractfile(member).read()
    if set(contents) != set(names):
        raise ValueError("Expected parallel test files absent")
    # Inspect shape only. Do not choose claims or models based on labels or scores.
    metadata = {}
    for name, data in contents.items():
        records = [json.loads(line) for line in data.splitlines()]
        if not records or not all(isinstance(r, dict) for r in records):
            raise ValueError("Invalid evaluation JSONL")
        metadata[name] = {
            "records": len(records),
            "sha256": hashlib.sha256(data).hexdigest(),
            "first_record_fields": {k: type(v).__name__ for k, v in records[0].items()},
        }
    output.mkdir()
    for name, data in contents.items():
        destination = output / name.removeprefix("data/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    manifest = {
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "files": metadata,
        "status": "evaluation inputs prepared; no model predictions or scores",
        "selection": "entire official test.6h English, machine and human translation subsets",
        "training_files_read": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(bundle) as handle:
        if handle.testzip() is not None:
            raise ValueError("Evaluation bundle failed integrity check")
    print(bundle)
    print("Prepared files only; multilingual evaluation has not run")


if __name__ == "__main__":
    main()
