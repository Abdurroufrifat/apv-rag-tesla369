"""Run fixed-aggregator own-excerpt diagnostics on internal validation."""

import argparse
from pathlib import Path

from apv_rag.own_excerpt import run_own_excerpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run_own_excerpt(Path(__file__).resolve().parents[1], args.batch_size, args.threads)
