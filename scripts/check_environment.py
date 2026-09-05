#!/usr/bin/env python3
"""Print a concise, non-sensitive compute report for reproducibility."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil


def gib(value: int) -> str:
    return f"{value / (1024**3):.1f} GiB"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def gpu_summary() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "No NVIDIA GPU detected by nvidia-smi (CPU/local-light workflow is OK)."
    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi was found but could not be queried: {type(exc).__name__}"
    return result.stdout.strip() or "nvidia-smi returned no GPU information."


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    disk = shutil.disk_usage(project_root)
    memory = psutil.virtual_memory()

    print("APV-RAG environment report")
    print(f"Project: {project_root}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()} [{sys.executable}]")
    print(f"Logical CPU cores: {psutil.cpu_count(logical=True)}")
    print(f"Physical CPU cores: {psutil.cpu_count(logical=False) or 'unknown'}")
    print(f"RAM total / available: {gib(memory.total)} / {gib(memory.available)}")
    print(f"Disk free: {gib(disk.free)}")
    print(f"GPU: {gpu_summary()}")
    print("Key packages:")
    for package in ("pytest", "PyYAML", "jsonschema", "psutil"):
        print(f"  {package}: {package_version(package)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

