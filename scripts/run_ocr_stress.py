"""Run the frozen Phase 2I OCR-corruption experiment."""

from pathlib import Path

from apv_rag.ocr_stress import run_phase2i

if __name__ == "__main__":
    directory = run_phase2i(Path(__file__).resolve().parents[1])
    print("Phase 2I OCR-corruption experiment passed.")
    print(f"Run folder: {directory}")
    print("Official development records used: 0")
