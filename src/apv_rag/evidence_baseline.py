"""Leakage-controlled text construction for the Phase 2C evidence baseline."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS
from apv_rag.metrics import classification_metrics
from apv_rag.splits import sha256, write_json_atomic

LABELS = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)
VARIANT_RENDERERS = {
    "claim_only": lambda row: render_claim_only(row),
    "claim_plus_evidence": lambda row: render_claim_plus_evidence(row),
}

def _required_text(mapping: Mapping[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _optional_text(mapping: Mapping[str, Any], field: str) -> str:
    value = mapping.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text when present")
    return value.strip()


def _required_list(mapping: Mapping[str, Any], field: str) -> list[Any]:
    value = mapping.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def render_claim_only(record: Mapping[str, Any]) -> str:
    """Render the diagnostic input without evidence or metadata."""

    return _required_text(record, "claim")


def render_claim_plus_evidence(record: Mapping[str, Any]) -> str:
    """Render only the approved AVeriTeC claim and evidence fields."""

    blocks = [f"CLAIM: {_required_text(record, 'claim')}"]
    for question in _required_list(record, "questions"):
        if not isinstance(question, Mapping):
            raise ValueError("each question must be an object")
        question_text = _optional_text(question, "question")
        if question_text:
            blocks.append(f"QUESTION: {question_text}")
        for answer in _required_list(question, "answers"):
            if not isinstance(answer, Mapping):
                raise ValueError("each answer must be an object")
            for prefix, field in (
                ("ANSWER_TYPE", "answer_type"),
                ("ANSWER", "answer"),
                ("SOURCE_MEDIUM", "source_medium"),
            ):
                value = _optional_text(answer, field)
                if value:
                    blocks.append(f"{prefix}: {value}")
    return "\n".join(blocks)


def load_split_records(
    records: Sequence[dict[str, Any]], indices: Sequence[int]
) -> list[dict[str, Any]]:
    """Select records by frozen upstream index after strict validation."""

    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("split indices must be integers")
    if len(set(indices)) != len(indices):
        raise ValueError("split indices must be unique")
    selected: list[dict[str, Any]] = []
    for index in indices:
        if index < 0 or index >= len(records):
            raise ValueError(f"split index out of range: {index}")
        record = records[index]
        try:
            _required_text(record, "claim")
        except ValueError as exc:
            raise ValueError(f"record {index} must have a non-empty claim") from exc
        selected.append(record)
    return selected


def select_setting(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select the evidence candidate by macro-F1, then lower C."""

    evidence = [row for row in candidates if row.get("variant") == "claim_plus_evidence"]
    if not evidence:
        raise ValueError("no claim_plus_evidence candidate is available")
    return min(evidence, key=lambda row: (-float(row["metrics"]["macro_f1"]), float(row["c"])))


def assert_prediction_alignment(
    validation_indices: Sequence[int], predictions: Sequence[Mapping[str, Any]]
) -> None:
    actual = [row.get("upstream_index") for row in predictions]
    if actual != list(validation_indices):
        raise ValueError("prediction indices do not match validation indices")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _build_pipeline(config: Mapping[str, Any], c_value: float) -> Pipeline:
    tfidf = config["tfidf"]
    seed = int(config["random_seed"])
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(int(value) for value in tfidf["ngram_range"]),
                    min_df=int(tfidf["min_df"]),
                    max_features=int(tfidf["max_features"]),
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LogisticRegression(
                        C=float(c_value),
                        max_iter=2000,
                        random_state=seed,
                    ),
                    method="sigmoid",
                    cv=StratifiedKFold(
                        n_splits=int(config["calibration_cv"]),
                        shuffle=True,
                        random_state=seed,
                    ),
                ),
            ),
        ]
    )


def _fit_candidate(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    variant: str,
    c_value: float,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        renderer = VARIANT_RENDERERS[variant]
    except KeyError as exc:
        raise ValueError(f"unknown baseline variant: {variant}") from exc
    train_text = [renderer(row) for row in train_records]
    validation_text = [renderer(row) for row in validation_records]
    y_train = [str(row["label"]) for row in train_records]
    y_true = [str(row["label"]) for row in validation_records]
    unknown = (set(y_train) | set(y_true)) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels in baseline data: {sorted(unknown)}")
    minimum_class_count = min(Counter(y_train).values())
    if minimum_class_count < int(config["calibration_cv"]):
        raise ValueError("each training class needs at least calibration_cv records")

    model = _build_pipeline(config, c_value)
    model.fit(train_text, y_train)
    raw_probabilities = model.predict_proba(validation_text)
    class_positions = {label: index for index, label in enumerate(model.classes_)}
    probabilities = np.column_stack(
        [raw_probabilities[:, class_positions[label]] for label in LABELS]
    )
    predicted_positions = probabilities.argmax(axis=1)
    y_pred = [LABELS[index] for index in predicted_positions]
    metrics = classification_metrics(
        y_true,
        y_pred,
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
        for row, (truth, predicted) in enumerate(zip(y_true, y_pred, strict=True))
    ]
    return {"variant": variant, "c": float(c_value), "metrics": metrics}, details


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_baseline(
    project_root: Path,
    run_id: str | None = None,
    *,
    expected_source_sha256: str = SPLITS["train"].sha256,
) -> Path:
    """Fit and save the frozen internal-validation baseline without reading dev.json."""

    root = Path(project_root).resolve()
    config_path = root / "config" / "evidence_baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = root / "data" / "external" / "averitec" / DATASET_DIR_NAME / "train.json"
    if sha256(source_path) != expected_source_sha256:
        raise ValueError("train.json SHA-256 does not match the expected source")
    split_dir = root / "data" / "processed" / "averitec" / "phase2b_split_v0_1"
    split_manifest_path = split_dir / "split_manifest.json"
    split_manifest = _load_json(split_manifest_path)
    expected_relative_source = str(source_path.relative_to(root)).replace("\\", "/")
    if split_manifest.get("source", {}).get("path") != expected_relative_source:
        raise ValueError("split manifest source does not match the pinned training file")
    records = _load_json(source_path)
    if not isinstance(records, list):
        raise ValueError("train.json must contain a list")
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

    candidates: list[dict[str, Any]] = []
    predictions_by_key: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for variant in config["variants"]:
        for c_value in config["regularization_c"]:
            result, details = _fit_candidate(
                train_records, validation_records, str(variant), float(c_value), config
            )
            candidates.append(result)
            predictions_by_key[(str(variant), float(c_value))] = details
    selected = select_setting(candidates)
    predictions = predictions_by_key[(selected["variant"], float(selected["c"]))]
    for upstream_index, prediction in zip(validation_indices, predictions, strict=True):
        prediction["upstream_index"] = upstream_index
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
            "selection_rule": "claim_plus_evidence macro_f1 descending, c ascending",
            "selected_variant": selected["variant"],
            "selected_c": selected["c"],
            "candidates": candidates,
        },
    )
    outputs = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in (prediction_path, metrics_path, selection_path)
    }
    manifest = {
        "protocol_version": str(config["protocol_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "run_id": identifier,
        "random_seed": int(config["random_seed"]),
        "labels": list(LABELS),
        "selected_variant": selected["variant"],
        "selected_c": selected["c"],
        "official_dev_records_used": 0,
        "counts": {"train": len(train_indices), "validation": len(validation_indices)},
        "inputs": {
            "train.json": {
                "path": str(source_path.relative_to(root)),
                "sha256": sha256(source_path),
            },
            "split_manifest.json": {
                "path": str(split_manifest_path.relative_to(root)),
                "sha256": sha256(split_manifest_path),
            },
            "evidence_baseline.yaml": {
                "path": str(config_path.relative_to(root)),
                "sha256": sha256(config_path),
            },
        },
        "outputs": outputs,
        "packages": {
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
            "PyYAML": version("PyYAML"),
        },
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    return run_dir


def validate_run(run_dir: Path) -> list[str]:
    """Return every detected integrity or alignment error for a saved run."""

    directory = Path(run_dir).resolve()
    errors: list[str] = []
    try:
        manifest = _load_json(directory / "run_manifest.json")
    except ValueError as exc:
        return [str(exc)]
    try:
        project_root = directory.parents[2]
    except IndexError:
        return ["run directory is not inside the expected artifacts folder"]
    labels = manifest.get("labels")
    if labels != list(LABELS):
        errors.append("manifest labels do not match the fixed label order")
    if manifest.get("official_dev_records_used") != 0:
        errors.append("official development records must remain unused")
    for name, entry in manifest.get("inputs", {}).items():
        relative = str(entry.get("path", ""))
        if name == "dev.json" or Path(relative).name == "dev.json":
            errors.append("manifest lists dev.json as a model input")
        input_path = (project_root / relative).resolve()
        if project_root not in input_path.parents:
            errors.append(f"input path escapes the project: {name}")
        elif not input_path.is_file():
            errors.append(f"missing input file: {name}")
        elif sha256(input_path) != entry.get("sha256"):
            errors.append(f"input SHA-256 mismatch: {name}")

    for filename, entry in manifest.get("outputs", {}).items():
        path = directory / filename
        if not path.is_file():
            errors.append(f"missing output file: {filename}")
        elif sha256(path) != entry.get("sha256"):
            errors.append(f"output SHA-256 mismatch: {filename}")

    prediction_path = directory / "validation_predictions.jsonl"
    predictions: list[dict[str, Any]] = []
    if prediction_path.is_file():
        try:
            predictions = [
                json.loads(line)
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid validation_predictions.jsonl: {exc}")
    for row_number, row in enumerate(predictions, start=1):
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
            errors.append(f"prediction {row_number} has invalid probability labels")
            continue
        values = np.asarray(list(probabilities.values()), dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
            or not np.isclose(values.sum(), 1.0, atol=1e-6)
        ):
            errors.append(f"prediction {row_number} probabilities do not sum to one")
        if not np.isclose(float(row.get("confidence", -1)), values.max(), atol=1e-12):
            errors.append(f"prediction {row_number} confidence is inconsistent")

    try:
        validation_indices = _load_json(
            project_root
            / "data"
            / "processed"
            / "averitec"
            / "phase2b_split_v0_1"
            / "validation_indices.json"
        )
        assert_prediction_alignment(validation_indices, predictions)
    except (IndexError, ValueError) as exc:
        errors.append(str(exc))

    if predictions and not any("probabilit" in error for error in errors):
        try:
            config = yaml.safe_load(
                (project_root / "config" / "evidence_baseline.yaml").read_text(encoding="utf-8")
            )
            matrix = np.asarray(
                [[row["probabilities"][label] for label in LABELS] for row in predictions]
            )
            recomputed = classification_metrics(
                [row["true_label"] for row in predictions],
                [row["predicted_label"] for row in predictions],
                matrix,
                LABELS,
                ece_bins=int(config["ece_bins"]),
                coverages=[float(value) for value in config["target_coverages"]],
            )
            recorded = _load_json(directory / "metrics.json")
            if recorded != recomputed:
                errors.append("recorded metrics do not match the predictions")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"could not recompute metrics: {exc}")
    return errors
