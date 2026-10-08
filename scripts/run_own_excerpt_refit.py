"""Train own-excerpt classifier on the original training split only."""

import argparse
from pathlib import Path

from apv_rag.own_excerpt_refit import run_refit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run_refit(Path(__file__).resolve().parents[1], args.batch_size, args.threads)
