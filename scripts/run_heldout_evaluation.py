"""Run the frozen, offline held-out excerpt evaluation."""

import argparse
from pathlib import Path

from apv_rag.heldout import run_heldout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run_heldout(Path(__file__).resolve().parents[1], args.batch_size, args.threads)
