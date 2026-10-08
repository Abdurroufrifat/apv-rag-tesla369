"""Run the frozen Phase 2H repetition stress experiment."""

from pathlib import Path

from apv_rag.repetition_stress import run_phase2h

if __name__ == "__main__":
    directory = run_phase2h(Path(__file__).resolve().parents[1])
    print("Phase 2H repetition-induced belief-shift experiment passed.")
    print(f"Run folder: {directory}")
    print("Official development records used: 0")
