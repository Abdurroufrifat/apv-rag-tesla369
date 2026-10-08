from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from apv_rag.benchmark_protocol import ProtocolError, validate_machine_only_protocol

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return yaml.safe_load((ROOT / "config" / "benchmark.yaml").read_text(encoding="utf-8"))


def test_frozen_protocol_is_valid():
    validate_machine_only_protocol(load_config())


def test_tesla_labels_are_rejected():
    config = deepcopy(load_config())
    config["datasets"]["tesla"]["labels"] = "machine_generated"
    with pytest.raises(ProtocolError, match="labels must be prohibited"):
        validate_machine_only_protocol(config)


def test_tesla_training_is_rejected():
    config = deepcopy(load_config())
    config["leakage_controls"]["tesla_excluded_from_training"] = False
    with pytest.raises(ProtocolError, match="leakage controls"):
        validate_machine_only_protocol(config)


def test_provenance_ablation_is_required():
    config = deepcopy(load_config())
    config["baselines"].remove("apv_rag_without_provenance")
    with pytest.raises(ProtocolError, match="provenance ablation"):
        validate_machine_only_protocol(config)
