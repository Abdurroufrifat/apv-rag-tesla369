"""Verify latest source receipts and exploratory outputs without model calls."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apv_rag.current_stage_evidence import current_evidence  # noqa: E402

if __name__ == "__main__":
    result = current_evidence(ROOT)
    print("Latest evidence verified:", ", ".join(result))
    print("Scoped received clean-confirmation exports; full charter remains open.")
