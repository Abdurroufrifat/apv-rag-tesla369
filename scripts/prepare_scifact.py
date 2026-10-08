"""Download SciFact and seal its untouched evaluation inputs for transfer work."""

import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/external/scifact/sealed_v1"
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if sha256(output / name) != digest:
                raise ValueError(f"SciFact input checksum mismatch: {name}")
        print("Existing SciFact package integrity verified")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError("Unexpected archive size")
    selected = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for name in ("corpus.jsonl", "claims_dev.jsonl"):
            members = [m for m in archive.getmembers() if m.isfile() and m.name == f"data/{name}"]
            if len(members) != 1:
                raise ValueError(f"Unexpected archive member: {name}")
            selected[name] = archive.extractfile(members[0]).read()
    # Do not inspect development labels or compute evaluation scores here.
    corpus = [json.loads(line) for line in selected["corpus.jsonl"].splitlines()]
    if len({r["doc_id"] for r in corpus}) != len(corpus):
        raise ValueError("Duplicate document identifiers")
    if not all(isinstance(r["abstract"], list) for r in corpus):
        raise ValueError("Invalid corpus schema")
    output.mkdir(parents=True)
    (output / "source_archive.tar.gz").write_bytes(payload)
    for name, contents in selected.items():
        (output / name).write_bytes(contents)
    write_json_atomic(
        output / "manifest.json",
        {
            "source_url": URL,
            "pinning": "first-download byte hashes; upstream latest URL is mutable",
            "evaluation_status": "not evaluated; labels unopened by preparation script",
            "corpus_documents": len(corpus),
            "files": {
                name: sha256(output / name)
                for name in ("source_archive.tar.gz", "corpus.jsonl", "claims_dev.jsonl")
            },
        },
    )
    print("SciFact downloaded and sealed; evaluation has not run")
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
