import json

import pytest

from apv_rag.current_stage_evidence import clean_confirmation
from apv_rag.splits import sha256


def test_scoped_receipt_accepts_unchanged_export_and_refuses_changed_file(tmp_path):
    verified = tmp_path / "artifacts/clean_confirmation_verification_v1"
    received = tmp_path / "artifacts/clean_confirmation_received_v1"
    verified.mkdir(parents=True)
    received.mkdir(parents=True)
    (received / "output.json").write_text("original", encoding="utf-8")
    (received / "audit_manifest.json").write_text(json.dumps({
        "files": {"output.json": sha256(received / "output.json")}}), encoding="utf-8")
    (verified / "verification.json").write_text(json.dumps({
        "verification": "passed", "difference_count": 0}), encoding="utf-8")
    (verified / "audit_manifest.json").write_text(json.dumps({
        "files": {"verification.json": sha256(verified / "verification.json")},
        "received_audit_manifest_sha256": sha256(received / "audit_manifest.json")}), encoding="utf-8")
    assert clean_confirmation(tmp_path, "clean_confirmation")["verification"] == "passed"
    (received / "output.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Clean export changed"):
        clean_confirmation(tmp_path, "clean_confirmation")


def test_release_command_includes_latest_evidence_and_component_analysis():
    from check_current_release import CHECKS
    assert ("latest_stage_evidence", ("scripts/verify_current_stage_evidence.py",)) in CHECKS
    assert ("component_ablation", ("scripts/analyze_frozen_components.py", "--verify")) in CHECKS
