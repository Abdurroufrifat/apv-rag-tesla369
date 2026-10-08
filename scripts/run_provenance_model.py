"""Run the frozen Phase 2F provenance-family ablation."""

from pathlib import Path

from apv_rag.provenance_model import run_phase2f

if __name__ == "__main__":
    directory = run_phase2f(Path(__file__).resolve().parents[1])
    print("Phase 2F provenance-family modeling passed.")
    print(f"Run folder: {directory}")
    print("Official development records used: 0")
