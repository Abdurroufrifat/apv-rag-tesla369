"""Validation helpers for the machine-only benchmark protocol."""

from __future__ import annotations

from typing import Any


class ProtocolError(ValueError):
    """Raised when benchmark settings violate the scientific boundary."""


def validate_machine_only_protocol(config: dict[str, Any]) -> None:
    """Validate the binding machine-only benchmark configuration."""

    if config.get("protocol_version") != "0.3":
        raise ProtocolError("protocol_version must be 0.3")
    if config.get("mode") != "machine_only":
        raise ProtocolError("mode must be machine_only")

    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        raise ProtocolError("datasets must be a mapping")

    tesla = datasets.get("tesla", {})
    if tesla.get("role") != "unlabeled_stress_test":
        raise ProtocolError("Tesla must remain an unlabeled stress test")
    if tesla.get("labels") != "prohibited":
        raise ProtocolError("Tesla labels must be prohibited")

    for name in ("primary", "transfer", "multilingual"):
        dataset = datasets.get(name, {})
        if dataset.get("labels") != "upstream_frozen":
            raise ProtocolError(f"{name} labels must be upstream_frozen")

    controls = config.get("leakage_controls", {})
    required_true = (
        "split_before_augmentation",
        "group_by_claim",
        "group_by_provenance_family",
        "fit_calibration_on_validation_only",
        "tesla_excluded_from_training",
        "tesla_excluded_from_model_selection",
        "tesla_excluded_from_headline_metrics",
    )
    false_controls = [name for name in required_true if controls.get(name) is not True]
    if false_controls:
        raise ProtocolError(f"required leakage controls are disabled: {false_controls}")

    seeds = config.get("random_seeds")
    if not isinstance(seeds, list) or len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ProtocolError("at least five distinct random seeds are required")

    baselines = set(config.get("baselines", []))
    required_baselines = {"standard_rag", "apv_rag_without_provenance", "apv_rag_full"}
    if not required_baselines.issubset(baselines):
        raise ProtocolError("required baseline or provenance ablation is missing")
