"""Controlled evidence-text OCR corruption experiment for Phase 2I."""

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
)
from apv_rag.evidence_baseline import LABELS, load_split_records, render_claim_plus_evidence
from apv_rag.imbalance_baseline import build_phase2d_pipeline
from apv_rag.splits import sha256, write_json_atomic

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPRESENTATIONS = ("word", "char", "hybrid")
_CONFUSIONS = {
    "0": "O",
    "1": "l",
    "5": "S",
    "8": "B",
    "a": "o",
    "c": "e",
    "e": "c",
    "i": "l",
    "l": "1",
    "m": "rn",
    "n": "r",
    "o": "0",
    "r": "n",
    "s": "5",
    "u": "v",
    "v": "u",
}


def _replacement(character: str) -> str:
    replacement = _CONFUSIONS.get(character.casefold(), "#")
    return replacement[0].upper() if character.isupper() else replacement[0]


def corrupt_text(text: str, rate: float, *, seed: int) -> str:
    """Replace an exact rounded fraction of alphanumeric characters."""
    if not 0 <= float(rate) <= 1:
        raise ValueError("OCR corruption rate must be between zero and one")
    characters = list(str(text))
    eligible = [index for index, character in enumerate(characters) if character.isalnum()]
    count = min(len(eligible), int(round(float(rate) * len(eligible))))
    if count == 0:
        return str(text)
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, size=count, replace=False)
    for index in selected:
        replacement = _replacement(characters[index])
        characters[index] = replacement if replacement != characters[index] else "#"
    return "".join(characters)


def corrupt_evidence_record(record: Mapping[str, Any], rate: float, *, seed: int) -> dict[str, Any]:
    """Corrupt evidence answers only, preserving claims, labels, questions, and URLs."""
    output = copy.deepcopy(dict(record))
    for question_index, question in enumerate(output.get("questions") or []):
        for answer_index, answer in enumerate(question.get("answers") or []):
            answer_seed = seed + question_index * 10_000 + answer_index
            answer["answer"] = corrupt_text(str(answer.get("answer") or ""), rate, seed=answer_seed)
    return output


def robustness_statistics(
    y_true: Sequence[str],
    clean_pred: Sequence[str],
    noisy_pred: Sequence[str],
    clean_confidence: np.ndarray,
    noisy_confidence: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compare clean and OCR-corrupted predictions on aligned records."""
    total = len(y_true)
    clean_values = np.asarray(clean_confidence, dtype=float)
    noisy_values = np.asarray(noisy_confidence, dtype=float)
    if total == 0 or len(clean_pred) != total or len(noisy_pred) != total:
        raise ValueError("truth and predictions must have equal non-zero length")
    if clean_values.shape != (total,) or noisy_values.shape != (total,):
        raise ValueError("confidence vectors must align with predictions")
    fixed_labels = list(labels or sorted(set(y_true) | set(clean_pred) | set(noisy_pred)))
    clean_f1 = float(
        f1_score(y_true, clean_pred, labels=fixed_labels, average="macro", zero_division=0)
    )
    noisy_f1 = float(
        f1_score(y_true, noisy_pred, labels=fixed_labels, average="macro", zero_division=0)
    )
    flips = np.asarray([a != b for a, b in zip(clean_pred, noisy_pred, strict=True)])
    return {
        "clean_macro_f1": clean_f1,
        "noisy_macro_f1": noisy_f1,
        "macro_f1_change": noisy_f1 - clean_f1,
        "prediction_flip_rate": float(flips.mean()),
        "mean_confidence_change": float((noisy_values - clean_values).mean()),
        "mean_absolute_confidence_change": float(np.abs(noisy_values - clean_values).mean()),
    }


def _predict(model, texts):
    raw = model.predict_proba(texts)
    positions = {label: index for index, label in enumerate(model.classes_)}
    probabilities = np.column_stack([raw[:, positions[label]] for label in LABELS])
    predicted = [LABELS[index] for index in probabilities.argmax(axis=1)]
    return predicted, probabilities.max(axis=1)


def _bootstrap(y_true, clean, noisy_by_rate, repetitions, seed):
    rng = np.random.default_rng(seed)
    truth = np.asarray(y_true, dtype=object)
    output = {representation: {} for representation in REPRESENTATIONS}
    for representation in REPRESENTATIONS:
        clean_pred = np.asarray(clean[representation]["predicted"], dtype=object)
        for rate, noisy in noisy_by_rate[representation].items():
            if rate == 0.0:
                continue
            noisy_pred = np.asarray(noisy["predicted"], dtype=object)
            changes = []
            for _ in range(repetitions):
                indices = rng.integers(0, len(truth), size=len(truth))
                clean_f1 = f1_score(
                    truth[indices], clean_pred[indices], labels=list(LABELS), average="macro"
                )
                noisy_f1 = f1_score(
                    truth[indices], noisy_pred[indices], labels=list(LABELS), average="macro"
                )
                changes.append(float(noisy_f1 - clean_f1))
            output[representation][f"{rate:.2f}"] = {
                "lower": float(np.percentile(changes, 2.5)),
                "upper": float(np.percentile(changes, 97.5)),
                "repetitions": int(repetitions),
            }
    return output


def _summarize(y_true, predictions, rates, repetitions, seed):
    clean = {name: predictions[name][0.0] for name in REPRESENTATIONS}
    metrics = {name: {} for name in REPRESENTATIONS}
    for name in REPRESENTATIONS:
        for rate in rates:
            noisy = predictions[name][rate]
            metrics[name][f"{rate:.2f}"] = robustness_statistics(
                y_true,
                clean[name]["predicted"],
                noisy["predicted"],
                clean[name]["confidence"],
                noisy["confidence"],
                labels=LABELS,
            )
    intervals = _bootstrap(y_true, clean, predictions, repetitions, seed)
    return metrics, intervals


def _save_figures(run_dir: Path, rates, metrics):
    f1_path = run_dir / "ocr_macro_f1.png"
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for name in REPRESENTATIONS:
        values = [metrics[name][f"{rate:.2f}"]["noisy_macro_f1"] for rate in rates]
        axis.plot(rates, values, marker="o", label=name)
    axis.set(xlabel="OCR corruption rate", ylabel="Macro-F1", title="Phase 2I OCR robustness")
    axis.set_xticks(rates, [f"{rate * 100:.0f}%" for rate in rates])
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(f1_path, dpi=200)
    plt.close(fig)
    flip_path = run_dir / "ocr_prediction_flips.png"
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(rates))
    width = 0.24
    for index, name in enumerate(REPRESENTATIONS):
        values = [metrics[name][f"{rate:.2f}"]["prediction_flip_rate"] for rate in rates]
        axis.bar(x + (index - 1) * width, values, width, label=name)
    axis.set_xticks(x)
    axis.set_xticklabels([f"{rate * 100:.0f}%" for rate in rates], fontsize=9)
    axis.set(
        xlabel="OCR corruption rate",
        ylabel="Prediction-flip rate",
        title="OCR-induced verdict changes",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(flip_path, dpi=200)
    plt.close(fig)
    return f1_path, flip_path


def run_phase2i(
    project_root: Path,
    run_id: str | None = None,
    *,
    expected_source_sha256: str = SPLITS["train"].sha256,
) -> Path:
    """Fit frozen text models and evaluate evidence-only OCR corruption."""
    root = Path(project_root).resolve()
    config_path = root / "config/ocr_stress.yaml"
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
    train_text = [render_claim_plus_evidence(row) for row in train_records]
    y_train = [str(row["label"]) for row in train_records]
    y_true = [str(row["label"]) for row in validation_records]
    rates = [float(value) for value in config["ocr_noise_rates"]]
    if rates[0] != 0.0 or any(rate < 0 or rate > 1 for rate in rates):
        raise ValueError("OCR rates must start at zero and stay in [0, 1]")
    predictions, output_rows = {name: {} for name in REPRESENTATIONS}, []
    for representation in REPRESENTATIONS:
        model = build_phase2d_pipeline(config, representation, float(config["regularization_c"]))
        model.fit(train_text, y_train)
        for rate in rates:
            changed = [
                corrupt_evidence_record(
                    row, rate, seed=int(config["random_seed"]) + int(index) * 1009
                )
                for index, row in zip(validation_indices, validation_records, strict=True)
            ]
            predicted, confidence = _predict(
                model, [render_claim_plus_evidence(row) for row in changed]
            )
            predictions[representation][rate] = {"predicted": predicted, "confidence": confidence}
            for position, upstream in enumerate(validation_indices):
                output_rows.append(
                    {
                        "upstream_index": upstream,
                        "representation": representation,
                        "ocr_rate": rate,
                        "true_label": y_true[position],
                        "predicted_label": predicted[position],
                        "confidence": float(confidence[position]),
                    }
                )
    metrics, intervals = _summarize(
        y_true, predictions, rates, int(config["bootstrap_repetitions"]), int(config["random_seed"])
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
    prediction_path = run_dir / "ocr_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8"
    )
    metrics_path, interval_path = run_dir / "ocr_metrics.json", run_dir / "bootstrap_intervals.json"
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(interval_path, intervals)
    f1_path, flip_path = _save_figures(run_dir, rates, metrics)
    outputs = [prediction_path, metrics_path, interval_path, f1_path, flip_path]
    manifest = {
        "protocol_version": str(config["protocol_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "run_id": identifier,
        "random_seed": int(config["random_seed"]),
        "ocr_noise_rates": rates,
        "representations": list(REPRESENTATIONS),
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
            "ocr_stress.yaml": {
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


def validate_phase2i_run(run_dir: Path) -> list[str]:
    """Validate Phase 2I hashes and recompute metrics from saved predictions."""
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
            for line in (directory / "ocr_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        rates = [float(value) for value in manifest["ocr_noise_rates"]]
        predictions = {name: {} for name in REPRESENTATIONS}
        for name in REPRESENTATIONS:
            for rate in rates:
                subset = [
                    row for row in rows if row["representation"] == name and row["ocr_rate"] == rate
                ]
                predictions[name][rate] = {
                    "predicted": [row["predicted_label"] for row in subset],
                    "confidence": np.asarray([row["confidence"] for row in subset]),
                }
        base = [row for row in rows if row["representation"] == "word" and row["ocr_rate"] == 0.0]
        y_true = [row["true_label"] for row in base]
        config = yaml.safe_load((root / "config/ocr_stress.yaml").read_text(encoding="utf-8"))
        metrics, intervals = _summarize(
            y_true,
            predictions,
            rates,
            int(config["bootstrap_repetitions"]),
            int(config["random_seed"]),
        )
        if _load_json(directory / "ocr_metrics.json") != metrics:
            errors.append("recorded OCR metrics do not match predictions")
        if _load_json(directory / "bootstrap_intervals.json") != intervals:
            errors.append("recorded bootstrap intervals do not match predictions")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return errors
