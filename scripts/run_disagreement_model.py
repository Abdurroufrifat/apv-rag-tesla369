"""Run the frozen Phase 2E evidence-disagreement ablation."""

from pathlib import Path

from apv_rag.disagreement_model import run_phase2e

if __name__ == "__main__":
    directory = run_phase2e(Path(__file__).resolve().parents[1])
    print("Phase 2E evidence-disagreement modeling passed.")
    print(f"Run folder: {directory}")
    print("Official development records used: 0")
