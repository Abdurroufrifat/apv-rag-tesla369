"""Run the frozen Phase 2G selective-prediction analysis."""

from pathlib import Path

from apv_rag.abstention_model import run_phase2g

if __name__ == "__main__":
    directory = run_phase2g(Path(__file__).resolve().parents[1])
    print("Phase 2G calibrated-abstention analysis passed.")
    print(f"Run folder: {directory}")
    print("Official development records used: 0")
