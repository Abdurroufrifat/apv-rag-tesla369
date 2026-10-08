"""Verify the latest bounded project evidence before regenerating status."""
import json
from pathlib import Path

from apv_rag.splits import sha256


def clean_confirmation(root: Path, stage: str) -> dict:
    if stage not in {"clean_confirmation", "clean_multilingual_confirmation"}:
        raise ValueError("Unknown clean confirmation stage")
    verified = root / "artifacts" / f"{stage}_verification_v1"
    received = root / "artifacts" / f"{stage}_received_v1"
    manifest = json.loads((verified / "audit_manifest.json").read_text(encoding="utf-8"))
    if sha256(received / "audit_manifest.json") != manifest["received_audit_manifest_sha256"]:
        raise ValueError("Clean received manifest changed")
    for name, expected in manifest["files"].items():
        if sha256(verified / name) != expected:
            raise ValueError("Clean verification receipt changed")
    original = json.loads((received / "audit_manifest.json").read_text(encoding="utf-8"))
    for name, expected in original["files"].items():
        if sha256(received / name) != expected:
            raise ValueError(f"Clean export changed: {name}")
    result = json.loads((verified / "verification.json").read_text(encoding="utf-8"))
    if result["verification"] != "passed" or result["difference_count"] != 0:
        raise ValueError("Clean confirmation comparison did not pass")
    return result


def current_evidence(root: Path) -> dict:
    from apv_rag.capture_timing_ablation import verify as verify_timing
    from audit_multilingual_pool_mismatch import verify as verify_pool
    # Both audit modules use this same existing project root in their CLI.
    import audit_multilingual_pool_mismatch as pool
    if pool.ROOT.resolve() != root.resolve():
        raise ValueError("Pool audit root differs")
    return {"english_clean_confirmation": clean_confirmation(root, "clean_confirmation"),
            "multilingual_clean_confirmation": clean_confirmation(root, "clean_multilingual_confirmation"),
            "capture_timing_ablation": verify_timing(root),
            "multilingual_pool_mismatch": verify_pool()}
