"""Run the CPU-only, offline local-model excerpt comparison."""

import argparse
from pathlib import Path

from apv_rag.nli_comparison import run_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run_comparison(Path(__file__).resolve().parents[1],
                   batch_size=args.batch_size, threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
