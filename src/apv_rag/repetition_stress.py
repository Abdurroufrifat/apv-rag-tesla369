"""Controlled repetition-induced belief-shift experiment for Phase 2H."""

from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import matplotlib
import numpy as np
import yaml

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS
from apv_rag.disagreement_model import (
    _load_json,
    _normalized_json_sha256,
    _normalized_split_manifest_sha256,
    build_phase2e_pipeline,
)
from apv_rag.evidence_baseline import LABELS, load_split_records
from apv_rag.provenance_model import normalize_source_url
from apv_rag.splits import sha256, write_json_atomic

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

MODELS = ("standard", "family_collapsed")


def _family(answer: Mapping[str, Any]) -> str:
    normalized = normalize_source_url(answer.get("source_url"))
    return urlsplit(f"https://{normalized}").hostname or "" if normalized else ""


def inject_derivative_copies(record: Mapping[str, Any], copy_count: int) -> dict[str, Any]:
    """Add exact copies from the first member of the dominant source family."""
    if copy_count < 0:
        raise ValueError("copy_count must be non-negative")
    output = copy.deepcopy(dict(record))
    located = []
    for question_index, question in enumerate(output.get("questions") or []):
        for answer_index, answer in enumerate(question.get("answers") or []):
            family = _family(answer)
            if family:
                located.append((question_index, answer_index, family))
    if not located or copy_count == 0:
        return output
    counts = Counter(item[2] for item in located)
    dominant = max(
        counts,
        key=lambda name: (
            counts[name],
            -next(i for i, row in enumerate(located) if row[2] == name),
        ),
    )
    question_index, answer_index, _ = next(row for row in located if row[2] == dominant)
    answer = output["questions"][question_index]["answers"][answer_index]
    output["questions"][question_index]["answers"].extend(
        copy.deepcopy(answer) for _ in range(copy_count)
    )
    return output


def collapse_provenance_families(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the first answer per source domain while preserving missing URLs."""
    output = copy.deepcopy(dict(record))
    seen: set[str] = set()
    for question_index, question in enumerate(output.get("questions") or []):
        kept = []
        for answer_index, answer in enumerate(question.get("answers") or []):
            family = _family(answer)
            key = family or f"missing:{question_index}:{answer_index}"
            if key not in seen:
                seen.add(key)
                kept.append(answer)
        question["answers"] = kept
    return output


def ribs_statistics(baseline: np.ndarray, stressed: np.ndarray) -> dict[str, float]:
    """Summarize support-probability shifts caused only by repetition."""
    original, changed = np.asarray(baseline, dtype=float), np.asarray(stressed, dtype=float)
    if original.shape != changed.shape or original.ndim != 1 or not len(original):
        raise ValueError("baseline and stressed probabilities must be aligned vectors")
    shift = changed - original
    return {
        "mean_signed_ribs": float(shift.mean()),
        "mean_absolute_ribs": float(np.abs(shift).mean()),
        "maximum_absolute_ribs": float(np.abs(shift).max()),
    }


def ribs_auc(copy_counts: Sequence[int], values: Sequence[float]) -> float:
    """Return copy-count-normalized trapezoidal area under a RIBS curve."""
    x, y = np.asarray(copy_counts, dtype=float), np.asarray(values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("copy counts and values must be aligned with increasing counts")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _predict(
    model, records: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, list[str], np.ndarray]:
    raw = model.predict_proba(records)
    positions = {label: index for index, label in enumerate(model.classes_)}
    probabilities = np.column_stack([raw[:, positions[label]] for label in LABELS])
    predicted = [LABELS[index] for index in probabilities.argmax(axis=1)]
    return probabilities[:, 0], predicted, probabilities.max(axis=1)


def _bootstrap_intervals(
    shifts: Mapping[str, Mapping[int, np.ndarray]], repetitions: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    total = len(next(iter(next(iter(shifts.values())).values())))
    samples = {
        model: {count: [] for count in by_count if count != 0} for model, by_count in shifts.items()
    }
    difference = {count: [] for count in next(iter(shifts.values())) if count != 0}
    for _ in range(repetitions):
        indices = rng.integers(0, total, size=total)
        for model, by_count in shifts.items():
            for count, values in by_count.items():
                if count:
                    samples[model][count].append(float(np.abs(values[indices]).mean()))
        for count in difference:
            standard = np.abs(shifts["standard"][count][indices]).mean()
            collapsed = np.abs(shifts["family_collapsed"][count][indices]).mean()
            difference[count].append(float(standard - collapsed))

    def interval(values):
        return {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
            "repetitions": int(repetitions),
        }

    return {
        "mean_absolute_ribs": {
            model: {str(count): interval(values) for count, values in by_count.items()}
            for model, by_count in samples.items()
        },
        "paired_standard_minus_collapsed": {
            str(count): interval(values) for count, values in difference.items()
        },
    }


def _summarize(copy_counts, support, repetitions, seed):
    metrics, shifts = {}, {}
    for model in MODELS:
        baseline = support[model][0]
        metrics[model], shifts[model] = {}, {}
        absolute = []
        for count in copy_counts:
            stats = ribs_statistics(baseline, support[model][count])
            metrics[model][str(count)] = stats
            shifts[model][count] = support[model][count] - baseline
            absolute.append(stats["mean_absolute_ribs"])
        metrics[model]["ribs_auc"] = ribs_auc(copy_counts, absolute)
    intervals = _bootstrap_intervals(shifts, repetitions, seed)
    return metrics, intervals, shifts


def _save_figures(run_dir: Path, copy_counts, metrics, shifts):
    curve_path = run_dir / "ribs_curves.png"
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    for model in MODELS:
        values = [metrics[model][str(count)]["mean_absolute_ribs"] for count in copy_counts]
        axis.plot(copy_counts, values, marker="o", label=model.replace("_", " "))
    axis.set(
        xlabel="Injected derivative copies",
        ylabel="Mean absolute RIBS",
        title="Phase 2H repetition-induced belief shift",
    )
    axis.set_xticks(copy_counts)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)

    distribution_path = run_dir / "ribs_distribution_k25.png"
    final_count = copy_counts[-1]
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.hist(shifts["standard"][final_count], bins=30, alpha=0.7, label="standard")
    axis.hist(
        shifts["family_collapsed"][final_count],
        bins=30,
        alpha=0.7,
        label="family collapsed",
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set(
        xlabel="Signed support-probability shift",
        ylabel="Records",
        title=f"RIBS distribution at {final_count} copies",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=200)
    plt.close(fig)
    return curve_path, distribution_path


def run_phase2h(
    project_root: Path,
    run_id: str | None = None,
    *,
    expected_source_sha256: str = SPLITS["train"].sha256,
) -> Path:
    """Fit two frozen verifiers and measure copy-only belief shifts."""
    root = Path(project_root).resolve()
    config_path = root / "config/repetition_stress.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = root / "data/external/averitec" / DATASET_DIR_NAME / "train.json"
    if sha256(source_path) != expected_source_sha256:
        raise ValueError("train.json SHA-256 does not match the expected source")
    split_dir = root / "data/processed/averitec/phase2b_split_v0_1"
    manifest_path = split_dir / "split_manifest.json"
    manifest = _load_json(manifest_path)
    relative_source = str(source_path.relative_to(root)).replace("\\", "/")
    if str(manifest.get("source", {}).get("path", "")).replace("\\", "/") != relative_source:
        raise ValueError("split manifest source does not match train.json")
    records = _load_json(source_path)
    train_indices = _load_json(split_dir / "train_indices.json")
    validation_indices = _load_json(split_dir / "validation_indices.json")
    if set(train_indices) & set(validation_indices):
        raise ValueError("training and validation indices overlap")
    train_records = load_split_records(records, train_indices)
    validation_records = load_split_records(records, validation_indices)
    copy_counts = [int(value) for value in config["copy_counts"]]
    adjacent = zip(copy_counts[:-1], copy_counts[1:], strict=True)
    if copy_counts[0] != 0 or any(b <= a for a, b in adjacent):
        raise ValueError("copy_counts must start at zero and increase")
    models = {}
    standard = build_phase2e_pipeline(
        config, "text_plus_disagreement", float(config["regularization_c"])
    )
    standard.fit(train_records, [str(row["label"]) for row in train_records])
    models["standard"] = standard
    collapsed_train = [collapse_provenance_families(row) for row in train_records]
    collapsed = build_phase2e_pipeline(
        config, "text_plus_disagreement", float(config["regularization_c"])
    )
    collapsed.fit(collapsed_train, [str(row["label"]) for row in collapsed_train])
    models["family_collapsed"] = collapsed

    support, predictions = {model: {} for model in MODELS}, []
    for count in copy_counts:
        stressed = [inject_derivative_copies(row, count) for row in validation_records]
        for model_name in MODELS:
            inputs = (
                stressed
                if model_name == "standard"
                else [collapse_provenance_families(row) for row in stressed]
            )
            support_values, predicted, confidence = _predict(models[model_name], inputs)
            support[model_name][count] = support_values
            for row_index, upstream in enumerate(validation_indices):
                predictions.append(
                    {
                        "upstream_index": upstream,
                        "model": model_name,
                        "copy_count": count,
                        "true_label": validation_records[row_index]["label"],
                        "predicted_label": predicted[row_index],
                        "support_probability": float(support_values[row_index]),
                        "confidence": float(confidence[row_index]),
                    }
                )
    metrics, intervals, shifts = _summarize(
        copy_counts,
        support,
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
        manifest_path,
        split_dir / "train_indices.json",
        split_dir / "validation_indices.json",
    ):
        shutil.copy2(path, snapshot / path.name)
    prediction_path = run_dir / "repetition_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    metrics_path, interval_path = (
        run_dir / "ribs_metrics.json",
        run_dir / "bootstrap_intervals.json",
    )
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(interval_path, intervals)
    curve_path, distribution_path = _save_figures(run_dir, copy_counts, metrics, shifts)
    outputs = [prediction_path, metrics_path, interval_path, curve_path, distribution_path]
    run_manifest = {
        "protocol_version": str(config["protocol_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "run_id": identifier,
        "random_seed": int(config["random_seed"]),
        "copy_counts": copy_counts,
        "models": list(MODELS),
        "labels": list(LABELS),
        "official_dev_records_used": 0,
        "counts": {"train": len(train_indices), "validation": len(validation_indices)},
        "inputs": {
            "train.json": {"path": relative_source, "sha256": sha256(source_path)},
            "split_manifest.json": {
                "path": "artifact:input_snapshot/split_manifest.json",
                "sha256": sha256(snapshot / "split_manifest.json"),
                "normalized_sha256": _normalized_split_manifest_sha256(manifest),
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
            "repetition_stress.yaml": {
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
    write_json_atomic(run_dir / "run_manifest.json", run_manifest)
    return run_dir


def validate_phase2h_run(run_dir: Path) -> list[str]:
    """Validate Phase 2H hashes, alignment, and recomputed RIBS outputs."""
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
        rows = [
            json.loads(line)
            for line in (directory / "repetition_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        counts = [int(value) for value in manifest["copy_counts"]]
        validation_count = int(manifest["counts"]["validation"])
        if len(rows) != len(MODELS) * len(counts) * validation_count:
            errors.append("prediction row count does not match manifest")
        support = {model: {} for model in MODELS}
        for model in MODELS:
            for count in counts:
                subset = [
                    row for row in rows if row["model"] == model and row["copy_count"] == count
                ]
                support[model][count] = np.asarray([row["support_probability"] for row in subset])
        config = yaml.safe_load(
            (root / "config/repetition_stress.yaml").read_text(encoding="utf-8")
        )
        metrics, intervals, _ = _summarize(
            counts, support, int(config["bootstrap_repetitions"]), int(config["random_seed"])
        )
        if _load_json(directory / "ribs_metrics.json") != metrics:
            errors.append("recorded RIBS metrics do not match predictions")
        if _load_json(directory / "bootstrap_intervals.json") != intervals:
            errors.append("recorded bootstrap intervals do not match predictions")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return errors
