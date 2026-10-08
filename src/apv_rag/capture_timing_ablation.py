"""Exploratory AVeriTeC archive-capture timing ablation on the observed internal split.

Wayback timestamps describe a snapshot, never a document's publication date.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS
from apv_rag.evidence_baseline import LABELS, load_split_records
from apv_rag.metrics import classification_metrics
from apv_rag.provenance_model import build_phase2f_pipeline
from apv_rag.splits import sha256, write_json_atomic

FEATURES = (
    "claim_date_present", "archive_capture_fraction", "post_claim_capture_fraction",
    "median_capture_lag_years", "median_absolute_capture_lag_years",
)
_WAYBACK = re.compile(r"^https?://web\.archive\.org/web/(\d{8})(?:\d{0,6})(?:[a-z_]{0,3})?/", re.I)


def _claim_date(value: object):
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%d-%m-%Y").date()
    except ValueError:
        return None


def _capture_date(answer: Mapping):
    for field in ("cached_source_url", "source_url"):
        url = answer.get(field)
        match = _WAYBACK.match(url) if isinstance(url, str) else None
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError:
                continue
    return None


def capture_features(record: Mapping) -> np.ndarray:
    """Measure snapshot availability and timing without touching benchmark labels."""
    claim = _claim_date(record.get("claim_date"))
    answers = [answer for question in record.get("questions") or []
               for answer in question.get("answers") or []]
    captures = [date for answer in answers if (date := _capture_date(answer)) is not None]
    lags = [(date - claim).days / 365.25 for date in captures] if claim else []
    return np.asarray([
        float(claim is not None),
        len(captures) / len(answers) if answers else 0.0,
        sum(lag > 0 for lag in lags) / len(lags) if lags else 0.0,
        float(np.median(lags)) if lags else 0.0,
        float(np.median(np.abs(lags))) if lags else 0.0,
    ], dtype=float)


class CaptureTimingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, records, y=None):
        return self

    def transform(self, records: Sequence[Mapping]):
        return sparse.csr_matrix(np.vstack([capture_features(row) for row in records]))


def build_timing_pipeline(config: Mapping) -> Pipeline:
    """Keep Phase 2F's fixed C=1.0 classifier and prepend five capture-only features."""
    baseline = build_phase2f_pipeline(config, "phase2e", 1.0)
    phase2e = baseline.named_steps["features"].transformer_list[0][1]
    timing = Pipeline([("extract", CaptureTimingTransformer()),
                       ("scale", StandardScaler(with_mean=False))])
    return Pipeline([
        ("features", FeatureUnion([("phase2e", phase2e), ("capture_timing", timing)])),
        ("classifier", baseline.named_steps["classifier"]),
    ])


def _predictions(model, records, indices):
    raw = model.predict_proba(records)
    columns = [list(model.classes_).index(label) for label in LABELS]
    probabilities = raw[:, columns]
    return [{
        "upstream_index": index,
        "true_label": str(record["label"]),
        "predicted_label": LABELS[int(np.argmax(row))],
        "probabilities": {label: float(value) for label, value in zip(LABELS, row, strict=True)},
    } for index, record, row in zip(indices, records, probabilities, strict=True)]


def _score(rows):
    return classification_metrics(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows],
        np.asarray([[row["probabilities"][label] for label in LABELS] for row in rows]),
        LABELS, ece_bins=15, coverages=[0.5, 0.8, 1.0],
    )


def run(root: Path) -> Path:
    root = root.resolve()
    source = root / "data/external/averitec" / DATASET_DIR_NAME / "train.json"
    if sha256(source) != SPLITS["train"].sha256:
        raise ValueError("pinned AVeriTeC training source mismatch")
    split = root / "data/processed/averitec/phase2b_split_v0_1"
    cfg_path = root / "config/provenance_model.yaml"
    frozen = root / "artifacts/phase2f_provenance_model/run_20260915T085322Z"
    paths = [source, split / "split_manifest.json", split / "train_indices.json",
             split / "validation_indices.json", cfg_path,
             frozen / "validation_predictions.jsonl", frozen / "run_manifest.json"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(paths[-1].read_text(encoding="utf-8"))
    if manifest.get("official_dev_records_used") != 0 or manifest.get("selected_representation") != "phase2e" or manifest.get("selected_c") != 1.0:
        raise ValueError("frozen Phase 2F baseline identity mismatch")
    for name, spec in manifest["outputs"].items():
        if sha256(frozen / name) != spec["sha256"]:
            raise ValueError(f"frozen Phase 2F output mismatch: {name}")
    split_manifest = json.loads(paths[1].read_text(encoding="utf-8"))
    if split_manifest.get("source", {}).get("path", "").replace("\\", "/") != str(source.relative_to(root)).replace("\\", "/"):
        raise ValueError("split source path mismatch")
    indices_train = json.loads(paths[2].read_text(encoding="utf-8"))
    indices_val = json.loads(paths[3].read_text(encoding="utf-8"))
    if set(indices_train) & set(indices_val):
        raise ValueError("train and validation overlap")
    all_records = json.loads(source.read_text(encoding="utf-8"))
    train = load_split_records(all_records, indices_train)
    validation = load_split_records(all_records, indices_val)
    baseline = [json.loads(line) for line in paths[5].read_text(encoding="utf-8").splitlines()]
    if len(baseline) != len(validation) or any(
        row["upstream_index"] != index or row["true_label"] != record["label"]
        for row, index, record in zip(baseline, indices_val, validation, strict=True)
    ):
        raise ValueError("frozen Phase 2F predictions do not align")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if int(config["random_seed"]) != 369 or int(config["calibration_cv"]) != 5:
        raise ValueError("Phase 2F training configuration changed")
    destination = root / "artifacts/capture_timing_ablation_v1"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite {destination}")
    model = build_timing_pipeline(config)
    model.fit(train, [record["label"] for record in train])
    augmented = _predictions(model, validation, indices_val)
    baseline_metrics = _score(baseline)
    augmented_metrics = _score(augmented)
    timing = np.vstack([capture_features(record) for record in validation])
    summary = {
        "protocol": "exploratory-capture-timing-v1",
        "population": "previously observed AVeriTeC internal validation; no official dev inference",
        "counts": {"training": len(train), "validation": len(validation),
                   "dated_validation": int((timing[:, 0] > 0).sum()),
                   "validation_with_capture": int((timing[:, 1] > 0).sum())},
        "features": FEATURES,
        "baseline": {"representation": "frozen Phase 2F phase2e C=1.0",
                     "macro_f1": baseline_metrics["macro_f1"],
                     "balanced_accuracy": baseline_metrics["balanced_accuracy"]},
        "augmented": {"representation": "phase2e + capture timing C=1.0",
                      "macro_f1": augmented_metrics["macro_f1"],
                      "balanced_accuracy": augmented_metrics["balanced_accuracy"]},
        "paired": {"changed_decisions": sum(a["predicted_label"] != b["predicted_label"] for a, b in zip(augmented, baseline, strict=True)),
                   "baseline_only_correct": sum(a["predicted_label"] != a["true_label"] and b["predicted_label"] == b["true_label"] for a, b in zip(augmented, baseline, strict=True)),
                   "augmented_only_correct": sum(a["predicted_label"] == a["true_label"] and b["predicted_label"] != b["true_label"] for a, b in zip(augmented, baseline, strict=True))},
        "limitations": ["Capture time is not publication time or source authentication.",
                        "Gold benchmark answer excerpts are supplied; retrieval and Tesla history are not evaluated.",
                        "Internal validation was repeatedly observed in earlier project stages; these results cannot independently confirm improvement."],
    }
    destination.mkdir(parents=True)
    output = destination / "validation_predictions.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in augmented), encoding="utf-8")
    write_json_atomic(destination / "summary.json", summary)
    inputs = {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in paths}
    outputs = {name: sha256(destination / name) for name in ("summary.json", output.name)}
    write_json_atomic(destination / "receipt.json", {"inputs": inputs, "outputs": outputs,
        "code": {str(path.relative_to(root)): sha256(path) for path in (
            root / "src/apv_rag/capture_timing_ablation.py",
            root / "src/apv_rag/provenance_model.py",
            root / "src/apv_rag/disagreement_model.py")}})
    return destination


def verify(root: Path) -> dict:
    root = root.resolve()
    directory = root / "artifacts/capture_timing_ablation_v1"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    for group in ("inputs", "outputs", "code"):
        for relative, expected in receipt[group].items():
            path = directory / relative if group == "outputs" else root / relative
            if sha256(path) != expected:
                raise ValueError(f"{group} SHA-256 mismatch: {relative}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (directory / "validation_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    indices = json.loads((root / "data/processed/averitec/phase2b_split_v0_1/validation_indices.json").read_text(encoding="utf-8"))
    if len(predictions) != len(indices) or any(row["upstream_index"] != index for row, index in zip(predictions, indices, strict=True)):
        raise ValueError("prediction index mismatch")
    score = _score(predictions)
    if abs(score["macro_f1"] - summary["augmented"]["macro_f1"]) > 1e-12:
        raise ValueError("prediction metric mismatch")
    return summary
