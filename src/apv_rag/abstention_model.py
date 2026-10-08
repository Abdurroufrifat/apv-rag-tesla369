"""Calibrated abstention and selective-prediction analysis for Phase 2G."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

from apv_rag.evidence_baseline import LABELS
from apv_rag.splits import sha256, write_json_atomic

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SCORE_ORDER = ("maximum_probability", "probability_margin", "normalized_entropy")


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("probabilities must be finite values between zero and one")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    return matrix


def abstention_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Compute label-free confidence scores where larger means safer."""
    matrix = _validate_probabilities(probabilities)
    ordered = np.sort(matrix, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(matrix > 0, matrix * np.log(matrix), 0.0)
    entropy_score = 1.0 + terms.sum(axis=1) / math.log(matrix.shape[1])
    return {
        "maximum_probability": ordered[:, -1],
        "probability_margin": ordered[:, -1] - ordered[:, -2],
        "normalized_entropy": np.clip(entropy_score, 0.0, 1.0),
    }


def risk_coverage_curve(y_true: Sequence[str], y_pred: Sequence[str], scores: np.ndarray):
    """Return the full empirical risk-coverage curve and discrete area."""
    values = np.asarray(scores, dtype=float)
    if len(y_true) == 0 or len(y_pred) != len(y_true) or values.shape != (len(y_true),):
        raise ValueError("truth, predictions, and scores must have equal non-zero length")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    order = np.argsort(-values, kind="stable")
    errors = np.asarray([a != b for a, b in zip(y_true, y_pred, strict=True)], dtype=float)[order]
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    points = [
        {
            "selected": i + 1,
            "coverage": float((i + 1) / len(errors)),
            "risk": float(risks[i]),
            "minimum_score": float(values[order[i]]),
        }
        for i in range(len(errors))
    ]
    return {"aurc": float(risks.mean()), "points": points}


def select_abstention_score(candidates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Select minimum AURC, breaking ties by the frozen score order."""
    names = [name for name in SCORE_ORDER if name in candidates]
    if not names:
        raise ValueError("no recognized abstention score candidates are available")
    name = min(names, key=lambda x: (float(candidates[x]["aurc"]), SCORE_ORDER.index(x)))
    return {"name": name, **dict(candidates[name])}


def threshold_policy(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores: np.ndarray,
    target_coverages: Sequence[float],
):
    """Freeze deterministic top-score policies for requested coverages."""
    values = np.asarray(scores, dtype=float)
    if len(y_true) == 0 or len(y_pred) != len(y_true) or values.shape != (len(y_true),):
        raise ValueError("truth, predictions, and scores must have equal non-zero length")
    order = np.argsort(-values, kind="stable")
    correct = np.asarray([a == b for a, b in zip(y_true, y_pred, strict=True)])
    output = {}
    for coverage in target_coverages:
        if not 0 < float(coverage) <= 1:
            raise ValueError("coverage values must be in (0, 1]")
        selected = min(len(values), max(1, math.ceil(float(coverage) * len(values))))
        retained = np.zeros(len(values), dtype=bool)
        retained[order[:selected]] = True
        accuracy = float(correct[retained].mean())
        output[f"{float(coverage):.6f}"] = {
            "target_coverage": float(coverage),
            "realized_coverage": float(selected / len(values)),
            "threshold": float(values[order[selected - 1]]),
            "selected": int(selected),
            "abstained": int(len(values) - selected),
            "selective_accuracy": accuracy,
            "selective_risk": 1.0 - accuracy,
            "decisions": ["machine_candidate" if keep else "abstain" for keep in retained],
        }
    return output


def bootstrap_aurc_intervals(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    score_sets: Mapping[str, np.ndarray],
    *,
    repetitions: int,
    seed: int,
):
    """Compute seeded percentile intervals for empirical AURC."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    total = len(y_true)
    if total == 0 or len(y_pred) != total:
        raise ValueError("truth and predictions must have equal non-zero length")
    arrays = {name: np.asarray(values, dtype=float) for name, values in score_sets.items()}
    if any(values.shape != (total,) for values in arrays.values()):
        raise ValueError("every score set must align with predictions")
    rng = np.random.default_rng(seed)
    truth, predicted = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    samples = {name: [] for name in arrays}
    for _ in range(repetitions):
        indices = rng.integers(0, total, size=total)
        for name, values in arrays.items():
            samples[name].append(
                risk_coverage_curve(truth[indices], predicted[indices], values[indices])["aurc"]
            )
    return {
        name: {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
            "repetitions": int(repetitions),
        }
        for name, values in samples.items()
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid predictions from {path}: {exc}") from exc
    if not rows:
        raise ValueError("prediction input is empty")
    return rows


def _analyze(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]):
    y_true = [str(row["true_label"]) for row in rows]
    y_pred = [str(row["predicted_label"]) for row in rows]
    probabilities = np.asarray(
        [[float(row["probabilities"][label]) for label in LABELS] for row in rows]
    )
    scores = abstention_scores(probabilities)
    curves = {name: risk_coverage_curve(y_true, y_pred, values) for name, values in scores.items()}
    selected = select_abstention_score(curves)
    policy = threshold_policy(
        y_true, y_pred, scores[selected["name"]], [float(x) for x in config["target_coverages"]]
    )
    intervals = bootstrap_aurc_intervals(
        y_true,
        y_pred,
        scores,
        repetitions=int(config["bootstrap_repetitions"]),
        seed=int(config["random_seed"]),
    )
    return y_true, y_pred, scores, curves, selected, policy, intervals


def _save_figures(run_dir: Path, curves, rows, selected_name, policy):
    risk_path = run_dir / "risk_coverage_curves.png"
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for name in SCORE_ORDER:
        points = curves[name]["points"]
        axis.plot(
            [p["coverage"] for p in points],
            [p["risk"] for p in points],
            label=f"{name.replace('_', ' ')} (AURC={curves[name]['aurc']:.3f})",
        )
    axis.set(
        xlabel="Coverage",
        ylabel="Selective risk",
        title="Phase 2G risk-coverage comparison",
        xlim=(0, 1),
    )
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(risk_path, dpi=200)
    plt.close(fig)
    retention_path = run_dir / "class_retention.png"
    keys = list(policy)
    counts = []
    for key in keys:
        retained = Counter(
            row["true_label"]
            for row, decision in zip(rows, policy[key]["decisions"], strict=True)
            if decision == "machine_candidate"
        )
        counts.append([retained[label] for label in LABELS])
    x, bottom = np.arange(len(keys)), np.zeros(len(keys))
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for column, label in enumerate(LABELS):
        values = np.asarray([row[column] for row in counts])
        axis.bar(
            x,
            values,
            bottom=bottom,
            label=label.replace("Conflicting Evidence/Cherrypicking", "Conflict"),
        )
        bottom += values
    axis.set_xticks(x)
    axis.set_xticklabels([f"{float(key) * 100:.0f}%" for key in keys], fontsize=9)
    axis.set(
        xlabel="Target coverage",
        ylabel="Retained records",
        title=f"Class retention using {selected_name.replace('_', ' ')}",
    )
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(retention_path, dpi=200)
    plt.close(fig)
    return risk_path, retention_path


def run_phase2g(
    project_root: Path, source_run_dir: Path | None = None, run_id: str | None = None
) -> Path:
    """Analyze Phase 2F predictions without reading the official development set."""
    root = Path(project_root).resolve()
    config_path = root / "config/abstention_model.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if source_run_dir is None:
        candidates = sorted((root / "artifacts/phase2f_provenance_model").glob("run_*"))
        if not candidates:
            raise ValueError("no Phase 2F run folder found")
        source_run_dir = candidates[-1]
    source_dir = Path(source_run_dir).resolve()
    source_manifest_path, source_predictions_path = (
        source_dir / "run_manifest.json",
        source_dir / "validation_predictions.jsonl",
    )
    source_manifest = _load_json(source_manifest_path)
    if source_manifest.get("official_dev_records_used") != 0:
        raise ValueError("Phase 2F source used official development records")
    if sha256(source_predictions_path) != source_manifest.get("outputs", {}).get(
        "validation_predictions.jsonl", {}
    ).get("sha256"):
        raise ValueError("Phase 2F prediction SHA-256 mismatch")
    rows = _load_predictions(source_predictions_path)
    if any(set(row.get("probabilities", {})) != set(LABELS) for row in rows):
        raise ValueError("prediction probability labels do not match fixed labels")
    y_true, y_pred, scores, curves, selected, policy, intervals = _analyze(rows, config)
    identifier = run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = root / str(config["output_directory"]) / identifier
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "input_snapshot"
    snapshot.mkdir()
    shutil.copy2(source_manifest_path, snapshot / "phase2f_run_manifest.json")
    shutil.copy2(source_predictions_path, snapshot / "validation_predictions.jsonl")
    summary_path, interval_path = (
        run_dir / "abstention_metrics.json",
        run_dir / "bootstrap_intervals.json",
    )
    policy_path, predictions_path = (
        run_dir / "threshold_policy.json",
        run_dir / "selective_predictions.jsonl",
    )
    write_json_atomic(
        summary_path,
        {
            "selection_rule": "minimum_aurc_then_frozen_score_order",
            "selected_score": selected["name"],
            "score_metrics": {name: {"aurc": curves[name]["aurc"]} for name in SCORE_ORDER},
        },
    )
    write_json_atomic(interval_path, intervals)
    clean_policy = {
        key: {name: value for name, value in entry.items() if name != "decisions"}
        for key, entry in policy.items()
    }
    write_json_atomic(policy_path, clean_policy)
    selected_scores = scores[selected["name"]]
    prediction_rows = [
        {
            "upstream_index": row["upstream_index"],
            "true_label": y_true[i],
            "predicted_label": y_pred[i],
            "selected_score": float(selected_scores[i]),
            "score_name": selected["name"],
            "decisions": {key: entry["decisions"][i] for key, entry in policy.items()},
        }
        for i, row in enumerate(rows)
    ]
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows), encoding="utf-8"
    )
    risk_path, retention_path = _save_figures(run_dir, curves, rows, selected["name"], policy)
    outputs = [
        summary_path,
        interval_path,
        policy_path,
        predictions_path,
        risk_path,
        retention_path,
    ]
    manifest = {
        "protocol_version": str(config["protocol_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "run_id": identifier,
        "random_seed": int(config["random_seed"]),
        "bootstrap_repetitions": int(config["bootstrap_repetitions"]),
        "labels": list(LABELS),
        "selected_score": selected["name"],
        "official_dev_records_used": 0,
        "counts": {"validation": len(rows)},
        "inputs": {
            "phase2f_run_manifest.json": {
                "path": "artifact:input_snapshot/phase2f_run_manifest.json",
                "sha256": sha256(snapshot / "phase2f_run_manifest.json"),
            },
            "validation_predictions.jsonl": {
                "path": "artifact:input_snapshot/validation_predictions.jsonl",
                "sha256": sha256(snapshot / "validation_predictions.jsonl"),
            },
            "abstention_model.yaml": {
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
            "PyYAML": version("PyYAML"),
        },
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    return run_dir


def validate_phase2g_run(run_dir: Path) -> list[str]:
    """Validate Phase 2G hashes and recompute every numeric output."""
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
        config = yaml.safe_load((root / "config/abstention_model.yaml").read_text(encoding="utf-8"))
        rows = _load_predictions(directory / "input_snapshot/validation_predictions.jsonl")
        _, _, _, curves, selected, policy, intervals = _analyze(rows, config)
        metrics = {
            "selection_rule": "minimum_aurc_then_frozen_score_order",
            "selected_score": selected["name"],
            "score_metrics": {name: {"aurc": curves[name]["aurc"]} for name in SCORE_ORDER},
        }
        clean_policy = {
            key: {name: value for name, value in entry.items() if name != "decisions"}
            for key, entry in policy.items()
        }
        if _load_json(directory / "abstention_metrics.json") != metrics:
            errors.append("recorded abstention metrics do not match snapshot")
        if _load_json(directory / "threshold_policy.json") != clean_policy:
            errors.append("recorded threshold policy does not match snapshot")
        if _load_json(directory / "bootstrap_intervals.json") != intervals:
            errors.append("recorded bootstrap intervals do not match snapshot")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return errors
