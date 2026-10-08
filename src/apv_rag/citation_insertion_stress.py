"""Synthetic fabricated-citation insertion stress testing for Phase 2K."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml
from sklearn.metrics import f1_score

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS
from apv_rag.disagreement_model import (
    _load_json,
    _normalized_json_sha256,
    _normalized_split_manifest_sha256,
    build_phase2e_pipeline,
)
from apv_rag.evidence_baseline import LABELS, load_split_records
from apv_rag.splits import sha256, write_json_atomic

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def insert_fabricated_citations(
    record: Mapping[str, Any], count: int, *, upstream_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Insert deterministic, non-resolving synthetic citations into evidence answers."""
    if count < 0:
        raise ValueError("citation count must be non-negative")
    output = copy.deepcopy(dict(record))
    if count == 0:
        return output, []
    questions = output.setdefault("questions", [])
    if not questions:
        questions.append({"question": "Synthetic citation stress-test prompt", "answers": []})
    answers = questions[0].setdefault("answers", [])
    claim = str(output.get("claim") or "").strip()
    audit = []
    for ordinal in range(1, count + 1):
        source_url = f"https://citation-{upstream_index}-{ordinal}.invalid/source"
        citation = {
            "answer": f"The cited source states that {claim}",
            "answer_type": "Abstractive",
            "source_url": source_url,
            "source_medium": "Synthetic stress-test citation",
            "cached_source_url": None,
            "synthetic_test_citation": True,
        }
        answers.append(citation)
        audit.append(
            {
                "upstream_index": int(upstream_index),
                "ordinal": ordinal,
                "source_url": source_url,
                "answer": citation["answer"],
                "synthetic_test_citation": True,
            }
        )
    return output, audit


def citation_insertion_statistics(
    y_true: Sequence[str],
    clean_pred: Sequence[str],
    stressed_pred: Sequence[str],
    clean_support: np.ndarray,
    stressed_support: np.ndarray,
    clean_confidence: np.ndarray,
    stressed_confidence: np.ndarray,
) -> dict[str, float | int]:
    """Summarize prediction and probability changes after citation insertion."""
    size = len(y_true)
    arrays = [
        np.asarray(clean_pred, dtype=object),
        np.asarray(stressed_pred, dtype=object),
        np.asarray(clean_support, dtype=float),
        np.asarray(stressed_support, dtype=float),
        np.asarray(clean_confidence, dtype=float),
        np.asarray(stressed_confidence, dtype=float),
    ]
    if not size or any(value.shape != (size,) for value in arrays):
        raise ValueError("citation-stress inputs must be aligned non-empty vectors")
    clean_predicted, stressed_predicted = arrays[:2]
    support_shift = arrays[3] - arrays[2]
    confidence_shift = arrays[5] - arrays[4]
    return {
        "count": size,
        "clean_macro_f1": float(
            f1_score(y_true, clean_predicted, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "stressed_macro_f1": float(
            f1_score(
                y_true,
                stressed_predicted,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "prediction_flip_rate": float(np.mean(clean_predicted != stressed_predicted)),
        "mean_support_probability_shift": float(support_shift.mean()),
        "mean_absolute_support_probability_shift": float(np.abs(support_shift).mean()),
        "mean_confidence_change": float(confidence_shift.mean()),
        "mean_absolute_confidence_change": float(np.abs(confidence_shift).mean()),
        "confidence_increase_rate": float(np.mean(confidence_shift > 0)),
    }


def _predict(model, records):
    raw = model.predict_proba(records)
    positions = {label: index for index, label in enumerate(model.classes_)}
    probabilities = np.column_stack([raw[:, positions[label]] for label in LABELS])
    predicted = [LABELS[index] for index in probabilities.argmax(axis=1)]
    return predicted, probabilities[:, 0], probabilities.max(axis=1)


def _summarize(y_true, predictions, insertion_counts, repetitions, seed):
    baseline = predictions[0]
    metrics = {
        str(count): citation_insertion_statistics(
            y_true,
            baseline["predicted"],
            predictions[count]["predicted"],
            baseline["support"],
            predictions[count]["support"],
            baseline["confidence"],
            predictions[count]["confidence"],
        )
        for count in insertion_counts
    }
    rng = np.random.default_rng(seed)
    truth = np.asarray(y_true, dtype=object)
    clean_pred = np.asarray(baseline["predicted"], dtype=object)
    intervals = {}
    for count in insertion_counts[1:]:
        stressed = predictions[count]
        stressed_pred = np.asarray(stressed["predicted"], dtype=object)
        macro_changes, support_changes = [], []
        support_shift = stressed["support"] - baseline["support"]
        for _ in range(repetitions):
            indices = rng.integers(0, len(truth), size=len(truth))
            clean_f1 = f1_score(
                truth[indices],
                clean_pred[indices],
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
            stressed_f1 = f1_score(
                truth[indices],
                stressed_pred[indices],
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
            macro_changes.append(float(stressed_f1 - clean_f1))
            support_changes.append(float(support_shift[indices].mean()))

        def interval(values):
            return {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
                "repetitions": int(repetitions),
            }

        intervals[str(count)] = {
            "macro_f1_change": interval(macro_changes),
            "mean_support_probability_shift": interval(support_changes),
        }
    return metrics, intervals


def _save_figure(run_dir: Path, metrics, insertion_counts):
    path = run_dir / "citation_insertion_stress.png"
    labels = [str(value) for value in insertion_counts]
    macro = [metrics[label]["stressed_macro_f1"] for label in labels]
    flips = [metrics[label]["prediction_flip_rate"] for label in labels]
    shifts = [metrics[label]["mean_support_probability_shift"] for label in labels]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    axes[0].bar(labels, macro, color="#4C78A8")
    axes[0].set(
        title="Macro-F1", xlabel="Inserted citations", ylabel="Score", ylim=(0, max(macro) + 0.08)
    )
    axes[1].bar(labels, flips, color="#E45756")
    axes[1].set(
        title="Prediction flips",
        xlabel="Inserted citations",
        ylabel="Rate",
        ylim=(0, max(flips + [0.01]) + 0.02),
    )
    axes[2].bar(labels, shifts, color="#F2CF5B")
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set(
        title="Support-probability shift", xlabel="Inserted citations", ylabel="Mean signed shift"
    )
    fig.suptitle("Phase 2K fabricated-citation insertion stress")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run_phase2k(
    project_root: Path,
    run_id: str | None = None,
    *,
    expected_source_sha256: str = SPLITS["train"].sha256,
) -> Path:
    """Fit the frozen verifier and measure synthetic citation influence."""
    root = Path(project_root).resolve()
    config_path = root / "config/citation_insertion_stress.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = root / "data/external/averitec" / DATASET_DIR_NAME / "train.json"
    if sha256(source_path) != expected_source_sha256:
        raise ValueError("train.json SHA-256 does not match the expected source")
    split_dir = root / "data/processed/averitec/phase2b_split_v0_1"
    split_manifest_path = split_dir / "split_manifest.json"
    split_manifest = _load_json(split_manifest_path)
    relative_source = str(source_path.relative_to(root)).replace("\\", "/")
    if str(split_manifest.get("source", {}).get("path", "")).replace("\\", "/") != relative_source:
        raise ValueError("split manifest source does not match train.json")
    records = _load_json(source_path)
    train_indices = _load_json(split_dir / "train_indices.json")
    validation_indices = _load_json(split_dir / "validation_indices.json")
    if set(train_indices) & set(validation_indices):
        raise ValueError("training and validation indices overlap")
    train_records = load_split_records(records, train_indices)
    validation_records = load_split_records(records, validation_indices)
    insertion_counts = [int(value) for value in config["insertion_counts"]]
    if insertion_counts[0] != 0 or any(
        b <= a for a, b in zip(insertion_counts[:-1], insertion_counts[1:], strict=True)
    ):
        raise ValueError("insertion_counts must start at zero and increase")
    model = build_phase2e_pipeline(
        config, "text_plus_disagreement", float(config["regularization_c"])
    )
    model.fit(train_records, [str(row["label"]) for row in train_records])
    y_true = [str(row["label"]) for row in validation_records]
    predictions, prediction_rows, audit_rows = {}, [], []
    for count in insertion_counts:
        stressed, level_audit = [], []
        for upstream, record in zip(validation_indices, validation_records, strict=True):
            changed, audit = insert_fabricated_citations(record, count, upstream_index=upstream)
            stressed.append(changed)
            level_audit.extend({"insertion_count": count, **entry} for entry in audit)
        predicted, support, confidence = _predict(model, stressed)
        predictions[count] = {"predicted": predicted, "support": support, "confidence": confidence}
        audit_rows.extend(level_audit)
        for position, upstream in enumerate(validation_indices):
            prediction_rows.append(
                {
                    "upstream_index": upstream,
                    "insertion_count": count,
                    "true_label": y_true[position],
                    "predicted_label": predicted[position],
                    "support_probability": float(support[position]),
                    "confidence": float(confidence[position]),
                }
            )
    metrics, intervals = _summarize(
        y_true,
        predictions,
        insertion_counts,
        int(config["bootstrap_repetitions"]),
        int(config["random_seed"]),
    )
    identifier = run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = root / str(config["output_directory"]) / identifier
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "input_snapshot"
    snapshot.mkdir()
    for path in (
        split_manifest_path,
        split_dir / "train_indices.json",
        split_dir / "validation_indices.json",
    ):
        shutil.copy2(path, snapshot / path.name)
    prediction_path = run_dir / "citation_predictions.jsonl"
    audit_path = run_dir / "synthetic_citation_audit.jsonl"
    metrics_path = run_dir / "citation_insertion_metrics.json"
    interval_path = run_dir / "bootstrap_intervals.json"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows), encoding="utf-8"
    )
    audit_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit_rows), encoding="utf-8"
    )
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(interval_path, intervals)
    figure_path = _save_figure(run_dir, metrics, insertion_counts)
    outputs = [prediction_path, audit_path, metrics_path, interval_path, figure_path]
    manifest = {
        "protocol_version": str(config["protocol_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "run_id": identifier,
        "random_seed": int(config["random_seed"]),
        "insertion_counts": insertion_counts,
        "labels": list(LABELS),
        "official_dev_records_used": 0,
        "counts": {"train": len(train_indices), "validation": len(validation_indices)},
        "inputs": {
            "train.json": {"path": relative_source, "sha256": sha256(source_path)},
            "split_manifest.json": {
                "path": "artifact:input_snapshot/split_manifest.json",
                "sha256": sha256(snapshot / "split_manifest.json"),
                "normalized_sha256": _normalized_split_manifest_sha256(split_manifest),
            },
            "train_indices.json": {
                "path": "artifact:input_snapshot/train_indices.json",
                "sha256": sha256(snapshot / "train_indices.json"),
                "normalized_sha256": _normalized_json_sha256(train_indices),
            },
            "validation_indices.json": {
                "path": "artifact:input_snapshot/validation_indices.json",
                "sha256": sha256(snapshot / "validation_indices.json"),
                "normalized_sha256": _normalized_json_sha256(validation_indices),
            },
            "citation_insertion_stress.yaml": {
                "path": str(config_path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256(config_path),
            },
        },
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs
        },
        "packages": {
            "numpy": version("numpy"),
            "matplotlib": version("matplotlib"),
            "scikit-learn": version("scikit-learn"),
            "PyYAML": version("PyYAML"),
        },
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    return run_dir


def validate_phase2k_run(run_dir: Path) -> list[str]:
    """Validate Phase 2K hashes, safety markers, and recomputed metrics."""
    directory, errors = Path(run_dir).resolve(), []
    try:
        root, manifest = directory.parents[2], _load_json(directory / "run_manifest.json")
    except (IndexError, ValueError) as exc:
        return [str(exc)]
    if manifest.get("labels") != list(LABELS):
        errors.append("manifest labels do not match fixed labels")
    if manifest.get("official_dev_records_used") != 0:
        errors.append("official development records must remain unused")
    for name, entry in manifest.get("inputs", {}).items():
        relative = str(entry.get("path", "")).replace("\\", "/")
        if Path(relative).name == "dev.json":
            errors.append("manifest lists dev.json as an input")
        path = (
            directory / relative.removeprefix("artifact:")
            if relative.startswith("artifact:")
            else root / relative
        )
        if not path.is_file():
            errors.append(f"invalid input path: {name}")
        elif sha256(path) != entry.get("sha256"):
            errors.append(f"input SHA-256 mismatch: {name}")
    for name, entry in manifest.get("outputs", {}).items():
        path = directory / name
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            errors.append(f"output integrity failure: {name}")
    if errors:
        return errors
    try:
        audit = [
            json.loads(line)
            for line in (directory / "synthetic_citation_audit.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        if any(
            not row.get("synthetic_test_citation") or ".invalid/" not in row.get("source_url", "")
            for row in audit
        ):
            errors.append("citation audit contains an unsafe or unmarked synthetic source")
        rows = [
            json.loads(line)
            for line in (directory / "citation_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        insertion_counts = [int(value) for value in manifest["insertion_counts"]]
        predictions = {}
        for count in insertion_counts:
            subset = [row for row in rows if row["insertion_count"] == count]
            predictions[count] = {
                "predicted": [row["predicted_label"] for row in subset],
                "support": np.asarray([row["support_probability"] for row in subset]),
                "confidence": np.asarray([row["confidence"] for row in subset]),
            }
        baseline_rows = [row for row in rows if row["insertion_count"] == 0]
        y_true = [row["true_label"] for row in baseline_rows]
        config = yaml.safe_load(
            (root / "config/citation_insertion_stress.yaml").read_text(encoding="utf-8")
        )
        metrics, intervals = _summarize(
            y_true,
            predictions,
            insertion_counts,
            int(config["bootstrap_repetitions"]),
            int(config["random_seed"]),
        )
        if _load_json(directory / "citation_insertion_metrics.json") != metrics:
            errors.append("recorded citation-insertion metrics do not match predictions")
        if _load_json(directory / "bootstrap_intervals.json") != intervals:
            errors.append("recorded bootstrap intervals do not match predictions")
        expected_audit = len(y_true) * sum(insertion_counts)
        if len(audit) != expected_audit:
            errors.append("citation audit row count does not match insertion protocol")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return errors
