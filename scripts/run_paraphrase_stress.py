"""Run the frozen Phase 2J claim-paraphrase experiment."""

from pathlib import Path

from apv_rag.paraphrase_stress import run_phase2j

if __name__ == "__main__":
    directory = run_phase2j(Path(__file__).resolve().parents[1])
    print("Phase 2J claim-paraphrase experiment passed.")
    print(f"Run folder: {directory}")
    print("Official development records used: 0")
