"""Check the local comparison outputs and recompute reported metrics."""

from pathlib import Path

from apv_rag.nli_comparison import validate_comparison


def main():
    output = Path(__file__).resolve().parents[1] / "artifacts/retrieval_nli_comparison"
    errors = validate_comparison(output)
    if errors:
        print("Retrieval/NLI comparison validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Retrieval/NLI comparison validation passed.")
    print("Official development records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
