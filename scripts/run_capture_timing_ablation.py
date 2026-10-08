"""Run or verify the exploratory Wayback capture-timing ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apv_rag.capture_timing_ablation import run, verify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not arguments.verify:
        print("Fitting one fixed capture-timing candidate on 2,458 training records...", flush=True)
        print(run(root), flush=True)
    result = verify(root)
    print("Capture-timing ablation verified:", result["baseline"]["macro_f1"],
          "->", result["augmented"]["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
