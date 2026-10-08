"""Check saved decision results without running neural inference or training."""

import json
from pathlib import Path

from sklearn.metrics import f1_score

from apv_rag.decision_rules import decision_indices
from apv_rag.evidence_baseline import LABELS
from apv_rag.splits import sha256


def main():
    directory = Path(__file__).resolve().parents[1] / "artifacts/cached_decision_comparison"
    manifest = json.loads((directory / "decision_manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f"output checksum mismatch: {name}")
    summary = json.loads((directory / "decision_summary.json").read_text())
    rows = json.loads((directory / "decision_predictions.json").read_text())
    if summary["official_dev_records_used"] != 0 or summary["probabilities_changed"]:
        raise ValueError("invalid experiment boundaries")
    for method, result in summary["methods"].items():
        for run in result["results"]:
            subset = [r for r in rows if r["method"] == method and r["seed"] == run["seed"]]
            positions = decision_indices(
                [r["probabilities"] for r in subset], summary["priors"], result["selected_exponent"]
            )
            if [LABELS[i] for i in positions] != [r["selected_label"] for r in subset]:
                raise ValueError("saved verdicts disagree with selected decision rule")
            macro = f1_score(
                [r["true_label"] for r in subset],
                [r["selected_label"] for r in subset],
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
            if macro != run["selected_macro_f1"]:
                raise ValueError("saved decision metrics do not match predictions")
    print("Cached decision comparison validation passed.")
    print("Official development records used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
