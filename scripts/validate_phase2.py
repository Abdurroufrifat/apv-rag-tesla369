#!/usr/bin/env python3
"""Validate the Phase 2 machine-only benchmark boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apv_rag.benchmark_protocol import ProtocolError, validate_machine_only_protocol  # noqa: E402


def main() -> int:
    path = PROJECT_ROOT / "config" / "benchmark.yaml"
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_machine_only_protocol(config)
    except (OSError, ProtocolError, yaml.YAMLError) as exc:
        print(f"Phase 2 validation failed: {exc}")
        return 1

    print("Phase 2 machine-only protocol validation passed.")
    print("Human review required: no")
    print("Benchmark labels: upstream frozen labels only")
    print("Tesla track: unlabeled archival stress test")
    print("Tesla permitted outputs: machine_candidate or abstain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
