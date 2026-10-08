"""Run the frozen Phase 2K synthetic citation-insertion experiment."""

from pathlib import Path

from apv_rag.citation_insertion_stress import run_phase2k


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    run_dir = run_phase2k(root)
    print("Phase 2K fabricated-citation insertion experiment passed.")
    print(f"Run folder: {run_dir}")
    print("Official development records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
