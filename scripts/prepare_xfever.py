"""Resume and verify the official XFEVER archive; inventory without extracting."""

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/8206962/files/data.tar.gz?download=1"
EXPECTED_MD5 = "8521ac9572a50a24b35b8a65e3e8abbb"


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/external/xfever/zenodo_8206962"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "data.tar.gz"
    partial = output / "data.tar.gz.part"
    if not archive.exists():
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        request = urllib.request.Request(URL, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            if start and status == 206:
                if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("Unexpected resume response")
                mode = "ab"
            elif status == 200:
                mode = "wb"
                start = 0
            else:
                raise ValueError(f"Unexpected download status: {status}")
            total = start
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    total += len(chunk)
                    if total > 600 * 1024 * 1024:
                        raise ValueError("Archive exceeds expected size range")
                    if total // (20 * 1024 * 1024) != (total - len(chunk)) // (20 * 1024 * 1024):
                        print(f"Downloaded {total / 1024 / 1024:.0f} MiB", flush=True)
        if digest(partial, "md5") != EXPECTED_MD5:
            raise ValueError("Official archive MD5 mismatch; partial file retained for inspection")
        partial.replace(archive)
    if digest(archive, "md5") != EXPECTED_MD5:
        raise ValueError("Existing archive differs from published checksum")
    inventory = []
    with tarfile.open(archive, "r:gz") as handle:
        for item in handle:
            if item.isfile():
                inventory.append({"path": item.name, "bytes": item.size})
    (output / "archive_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    manifest = {
        "source_url": URL,
        "official_record": "https://zenodo.org/records/8206962",
        "archive_bytes": archive.stat().st_size,
        "published_md5": EXPECTED_MD5,
        "actual_md5": digest(archive, "md5"),
        "archive_sha256": digest(archive, "sha256"),
        "inventory_sha256": digest(output / "archive_inventory.json", "sha256"),
        "evaluation_status": "not evaluated; archive inventoried without reading label records",
        "license_status": "dataset terms must be checked separately from repository code license",
    }
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("XFEVER archive checksum verified; evaluation has not run")
    print(output / "download_manifest.json")


if __name__ == "__main__":
    main()
