"""Imbalance-aware word, character, and hybrid evidence models for Phase 2D."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS
from apv_rag.evidence_baseline import (
    LABELS,
    assert_prediction_alignment,
    load_split_records,
    render_claim_plus_evidence,
)
from apv_rag.metrics import classification_metrics
from apv_rag.splits import sha256, write_json_atomic

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _vectorizer(config: Mapping[str, Any], analyzer: str) -> TfidfVectorizer:
    settings = config[f"{analyzer}_tfidf"]
    return TfidfVectorizer(
        analyzer="word" if analyzer == "word" else "char_wb",
        ngram_range=tuple(int(value) for value in settings["ngram_range"]),
        min_df=int(settings["min_df"]),
        max_features=int(settings["max_features"]),
        sublinear_tf=True,
    )


def build_phase2d_pipeline(
    config: Mapping[str, Any], representation: str, c_value: float
) -> Pipeline:
    """Build a calibrated, class-balanced model for one text representation."""

    if representation == "word":
        transformers = [("word", _vectorizer(config, "word"))]
    elif representation == "char":
        transformers = [("char", _vectorizer(config, "char"))]
    elif representation == "hybrid":
        transformers = [
            ("word", _vectorizer(config, "word")),
            ("char", _vectorizer(config, "char")),
        ]
    else:
        raise ValueError(f"unknown representation: {representation}")
    seed = int(config["random_seed"])
    classifier = CalibratedClassifierCV(
        estimator=LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=2500,
            random_state=seed,
        ),
        method="sigmoid",
        cv=StratifiedKFold(
            n_splits=int(config["calibration_cv"]),
            shuffle=True,
            random_state=seed,
        ),
    )
    return Pipeline([("features", FeatureUnion(transformers)), ("classifier", classifier)])


def select_phase2d_setting(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen Macro-F1 and conflict-class selection rule."""

    if not candidates:
        raise ValueError("no Phase 2D candidates are available")
    conflict = LABELS[3]
    order = {"word": 0, "char": 1, "hybrid": 2}
    return max(
        candidates,
        key=lambda row: (
            float(row["metrics"]["macro_f1"]),
            float(row["metrics"]["per_class"][conflict]["f1"]),
            float(row["metrics"]["balanced_accuracy"]),
            -float(row["c"]),
            order.get(str(row["representation"]), -1),
        ),
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _fit_candidate(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    representation: str,
    c_value: float,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_text = [render_claim_plus_evidence(row) for row in train_records]
    validation_text = [render_claim_plus_evidence(row) for row in validation_records]
    y_train = [str(row["label"]) for row in train_records]
    y_true = [str(row["label"]) for row in validation_records]
    if (set(y_train) | set(y_true)) - set(LABELS):
        raise ValueError("Phase 2D data contain an unknown label")
    if min(Counter(y_train).values()) < int(config["calibration_cv"]):
        raise ValueError("each training class needs at least calibration_cv records")
    model = build_phase2d_pipeline(config, representation, c_value)
    model.fit(train_text, y_train)
    raw = model.predict_proba(validation_text)
    positions = {label: index for index, label in enumerate(model.classes_)}
    probabilities = np.column_stack([raw[:, positions[label]] for label in LABELS])
    predictions = [LABELS[index] for index in probabilities.argmax(axis=1)]
    metrics = classification_metrics(
        y_true,
        predictions,
        probabilities,
        LABELS,
        ece_bins=int(config["ece_bins"]),
        coverages=[float(value) for value in config["target_coverages"]],
    )
    details = [
        {
            "true_label": truth,
            "predicted_label": predicted,
            "probabilities": {
                label: float(probabilities[row, column])
                for column, label in enumerate(LABELS)
            },
            "confidence": float(probabilities[row].max()),
        }
        for row, (truth, predicted) in enumerate(zip(y_true, predictions, strict=True))
    ]
    return {
        "representation": representation,
        "c": float(c_value),
        "metrics": metrics,
    }, details


def _save_figures(
    run_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    best_by_representation = {}
    for row in candidates:
        name = str(row["representation"])
        previous = best_by_representation.get(name)
        if previous is None or row["metrics"]["macro_f1"] > previous["metrics"]["macro_f1"]:
            best_by_representation[name] = row
    names = [name for name in ("word", "char", "hybrid") if name in best_by_representation]
    scores = [best_by_representation[name]["metrics"]["macro_f1"] for name in names]
    comparison_path = run_dir / "representation_comparison.png"
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    bars = axis.bar(names, scores, color=["#4C78A8", "#F58518", "#54A24B"][: len(names)])
    axis.set_ylabel("Validation Macro-F1")
    axis.set_ylim(0, max(0.6, max(scores) + 0.08))
    axis.set_title("Phase 2D evidence representations")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=200)
    plt.close(fig)

    matrix = confusion_matrix(
        [row["true_label"] for row in predictions],
        [row["predicted_label"] for row in predictions],
        labels=list(LABELS),
    )
    matrix_path = run_dir / "confusion_matrix.png"
    short_labels = ["Supported", "Refuted", "NEI", "Conflict"]
    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(4), short_labels, rotation=25, ha="right")
    axis.set_yticks(range(4), short_labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Selected Phase 2D model")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=200)
    plt.close(fig)
    return comparison_path, matrix_path


def run_phase2d(
    project_root: Path,
    run_id: str | None = None,
    *,
    expected_source_sha256: str = SPLITS["train"].sha256,
) -> Path:
    """Fit Phase 2D candidates using only the frozen Phase 2B split."""

    root = Path(project_root).resolve()
    config_path = root / "config" / "imbalance_baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = root / "data" / "external" / "averitec" / DATASET_DIR_NAME / "train.json"
    if sha256(source_path) != expected_source_sha256:
        raise ValueError("train.json SHA-256 does not match the expected source")
    split_dir = root / "data" / "processed" / "averitec" / "phase2b_split_v0_1"
    split_manifest_path = split_dir / "split_manifest.json"
    split_manifest = _load_json(split_manifest_path)
    relative_source = str(source_path.relative_to(root)).replace("\\", "/")
    manifest_source = str(split_manifest.get("source", {}).get("path", "")).replace(
        "\\", "/"
    )
    if manifest_source != relative_source:
        raise ValueError("split manifest source does not match train.json")
    records = _load_json(source_path)
    train_indices = _load_json(split_dir / "train_indices.json")
    validation_indices = _load_json(split_dir / "validation_indices.json")
    train_records = load_split_records(records, train_indices)
    validation_records = load_split_records(records, validation_indices)
    if set(train_indices) & set(validation_indices):
        raise ValueError("training and validation indices overlap")

    identifier = run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = root / str(config["output_directory"]) / identifier
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    candidates = []
    predictions_by_key = {}
    for representation in config["representations"]:
        for c_value in config["regularization_c"]:
            result, details = _fit_candidate(
                train_records,
                validation_records,
                str(representation),
                float(c_value),
                config,
            )
            candidates.append(result)
            predictions_by_key[(str(representation), float(c_value))] = details
    selected = select_phase2d_setting(candidates)
    predictions = predictions_by_key[(selected["representation"], float(selected["c"]))]
    for index, row in zip(validation_indices, predictions, strict=True):
        row["upstream_index"] = index
    assert_prediction_alignment(validation_indices, predictions)

    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "validation_predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    selection_path = run_dir / "model_selection.json"
    _write_jsonl_atomic(prediction_path, predictions)
    write_json_atomic(metrics_path, selected["metrics"])
    write_json_atomic(
        selection_path,
        {
            "selection_rule": "macro_f1, conflict_f1, balanced_accuracy, lower_c",
            "selected_representation": selected["representation"],
            "selected_c": selected["c"],
            "candidates": candidates,
        },
    )
    comparison_path, matrix_path = _save_figures(run_dir, candidates, predictions)
    output_paths = [prediction_path, metrics_path, selection_path, comparison_path, matrix_path]
    manifest = {
        "protocol_version": str(config["protocol_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "run_id": identifier,
        "random_seed": int(config["random_seed"]),
        "labels": list(LABELS),
        "selected_representation": selected["representation"],
        "selected_c": selected["c"],
        "class_weight": "balanced",
        "official_dev_records_used": 0,
        "counts": {"train": len(train_indices), "validation": len(validation_indices)},
        "inputs": {
            "train.json": {"path": relative_source, "sha256": sha256(source_path)},
            "split_manifest.json": {
                "path": str(split_manifest_path.relative_to(root)),
                "sha256": sha256(split_manifest_path),
            },
            "imbalance_baseline.yaml": {
                "path": str(config_path.relative_to(root)),
                "sha256": sha256(config_path),
            },
        },
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
        "packages": {
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
            "matplotlib": version("matplotlib"),
            "PyYAML": version("PyYAML"),
        },
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    return run_dir


def validate_phase2d_run(run_dir: Path) -> list[str]:
    """Check Phase 2D hashes, predictions, probabilities, and frozen indices."""

    directory = Path(run_dir).resolve()
    errors = []
    try:
        root = directory.parents[2]
        manifest = _load_json(directory / "run_manifest.json")
    except (IndexError, ValueError) as exc:
        return [str(exc)]
    if manifest.get("labels") != list(LABELS):
        errors.append("manifest labels do not match the fixed labels")
    if manifest.get("official_dev_records_used") != 0:
        errors.append("official development records must remain unused")
    for name, entry in manifest.get("inputs", {}).items():
        relative = str(entry.get("path", ""))
        if Path(relative).name == "dev.json":
            errors.append("manifest lists dev.json as a model input")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            errors.append(f"invalid input path: {name}")
        elif sha256(path) != entry.get("sha256"):
            errors.append(f"input SHA-256 mismatch: {name}")
    for name, entry in manifest.get("outputs", {}).items():
        path = directory / name
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            errors.append(f"output integrity failure: {name}")
    prediction_path = directory / "validation_predictions.jsonl"
    try:
        predictions = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        indices = _load_json(
            root / "data/processed/averitec/phase2b_split_v0_1/validation_indices.json"
        )
        assert_prediction_alignment(indices, predictions)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        predictions = []
    for number, row in enumerate(predictions, start=1):
        probabilities = row.get("probabilities", {})
        if set(probabilities) != set(LABELS):
            errors.append(f"prediction {number} has invalid probability labels")
            continue
        values = np.asarray([probabilities[label] for label in LABELS], dtype=float)
        if not np.isclose(values.sum(), 1.0, atol=1e-6):
            errors.append(f"prediction {number} probabilities do not sum to one")
        if not np.isclose(float(row.get("confidence", -1)), values.max(), atol=1e-12):
            errors.append(f"prediction {number} confidence is inconsistent")
    if predictions and not any("probabilit" in error for error in errors):
        try:
            config = yaml.safe_load(
                (root / "config" / "imbalance_baseline.yaml").read_text(encoding="utf-8")
            )
            probability_matrix = np.asarray(
                [[row["probabilities"][label] for label in LABELS] for row in predictions]
            )
            recomputed = classification_metrics(
                [row["true_label"] for row in predictions],
                [row["predicted_label"] for row in predictions],
                probability_matrix,
                LABELS,
                ece_bins=int(config["ece_bins"]),
                coverages=[float(value) for value in config["target_coverages"]],
            )
            if _load_json(directory / "metrics.json") != recomputed:
                errors.append("recorded metrics do not match predictions")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"could not recompute metrics: {exc}")
    return errors
