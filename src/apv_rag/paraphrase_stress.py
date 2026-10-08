"""Conservative rule-based claim-paraphrase stress testing for Phase 2J."""

from __future__ import annotations

import copy
import json
import re
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
from apv_rag.ocr_stress import robustness_statistics
from apv_rag.splits import sha256, write_json_atomic

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

STRENGTHS = ("none", "mild", "strong")
_RULES = (
    (r"\bthe report says\b", "the report states"),
    (r"\breportedly\b", "according to reports"),
    (r"\baccording to\b", "as stated by"),
    (r"\bclaims that\b", "asserts that"),
    (r"\bsaid that\b", "stated that"),
    (r"\bsays that\b", "states that"),
    (r"\bshows that\b", "indicates that"),
    (r"\bhappened\b", "occurred"),
    (r"\bbecause\b", "as a result of"),
    (r"\bfound evidence\b", "identified evidence"),
    (r"\bwas created by\b", "was produced by"),
    (r"\bwas written by\b", "was authored by"),
    (r"\bis false\b", "is not true"),
    (r"\bis true\b", "is accurate"),
)


def _case_match(source: str, replacement: str) -> str:
    return replacement[:1].upper() + replacement[1:] if source[:1].isupper() else replacement


def paraphrase_claim(claim: str, strength: str) -> str:
    """Apply at most one mild or three strong conservative phrase substitutions."""
    if strength not in STRENGTHS:
        raise ValueError(f"unknown paraphrase strength: {strength}")
    if strength == "none":
        return str(claim)
    limit = 1 if strength == "mild" else 3
    output, changes = str(claim), 0
    for pattern, replacement in _RULES:
        if changes >= limit:
            break
        output, count = re.subn(
            pattern,
            lambda match, value=replacement: _case_match(match.group(0), value),
            output,
            count=1,
            flags=re.IGNORECASE,
        )
        changes += count
    return output


def paraphrase_record(record: Mapping[str, Any], strength: str) -> dict[str, Any]:
    """Paraphrase only the claim field."""
    output = copy.deepcopy(dict(record))
    output["claim"] = paraphrase_claim(str(output.get("claim") or ""), strength)
    return output


def _subset_statistics(y_true, clean_pred, noisy_pred, clean_conf, noisy_conf, mask, labels):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return {"count": 0}
    result = robustness_statistics(
        np.asarray(y_true, dtype=object)[indices],
        np.asarray(clean_pred, dtype=object)[indices],
        np.asarray(noisy_pred, dtype=object)[indices],
        np.asarray(clean_conf)[indices],
        np.asarray(noisy_conf)[indices],
        labels=labels,
    )
    return {"count": int(len(indices)), **result}


def paraphrase_statistics(
    y_true: Sequence[str],
    clean_pred: Sequence[str],
    noisy_pred: Sequence[str],
    clean_confidence: np.ndarray,
    noisy_confidence: np.ndarray,
    changed_mask: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report all-record and actually-transformed-record robustness separately."""
    mask = np.asarray(changed_mask, dtype=bool)
    if mask.shape != (len(y_true),):
        raise ValueError("changed mask must align with records")
    fixed_labels = list(labels or sorted(set(y_true) | set(clean_pred) | set(noisy_pred)))
    all_records = robustness_statistics(
        y_true,
        clean_pred,
        noisy_pred,
        clean_confidence,
        noisy_confidence,
        labels=fixed_labels,
    )
    return {
        "transformation_coverage": float(mask.mean()),
        "all_records": {"count": len(y_true), **all_records},
        "changed_records": _subset_statistics(
            y_true,
            clean_pred,
            noisy_pred,
            clean_confidence,
            noisy_confidence,
            mask,
            fixed_labels,
        ),
    }


def _predict(model, records):
    raw = model.predict_proba(records)
    positions = {label: index for index, label in enumerate(model.classes_)}
    probabilities = np.column_stack([raw[:, positions[label]] for label in LABELS])
    predicted = [LABELS[index] for index in probabilities.argmax(axis=1)]
    return predicted, probabilities.max(axis=1)


def _bootstrap(y_true, predictions, masks, repetitions, seed):
    rng = np.random.default_rng(seed)
    truth = np.asarray(y_true, dtype=object)
    clean = np.asarray(predictions["none"]["predicted"], dtype=object)
    output = {}
    for strength in ("mild", "strong"):
        noisy = np.asarray(predictions[strength]["predicted"], dtype=object)
        changed = np.flatnonzero(masks[strength])
        all_changes, changed_changes = [], []
        for _ in range(repetitions):
            indices = rng.integers(0, len(truth), size=len(truth))
            clean_f1 = f1_score(
                truth[indices],
                clean[indices],
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
            noisy_f1 = f1_score(
                truth[indices],
                noisy[indices],
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
            all_changes.append(float(noisy_f1 - clean_f1))
            if len(changed):
                subset = rng.choice(changed, size=len(changed), replace=True)
                clean_f1 = f1_score(
                    truth[subset],
                    clean[subset],
                    labels=list(LABELS),
                    average="macro",
                    zero_division=0,
                )
                noisy_f1 = f1_score(
                    truth[subset],
                    noisy[subset],
                    labels=list(LABELS),
                    average="macro",
                    zero_division=0,
                )
                changed_changes.append(float(noisy_f1 - clean_f1))

        def interval(values):
            return {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
                "repetitions": repetitions,
            }

        output[strength] = {
            "all_records": interval(all_changes),
            "changed_records": interval(changed_changes) if changed_changes else None,
        }
    return output


def _summarize(y_true, predictions, masks, repetitions, seed):
    clean = predictions["none"]
    metrics = {}
    for strength in STRENGTHS:
        current = predictions[strength]
        metrics[strength] = paraphrase_statistics(
            y_true,
            clean["predicted"],
            current["predicted"],
            clean["confidence"],
            current["confidence"],
            masks[strength],
            labels=LABELS,
        )
    return metrics, _bootstrap(y_true, predictions, masks, repetitions, seed)


def _save_figures(run_dir, metrics):
    path = run_dir / "paraphrase_robustness.png"
    strengths = list(STRENGTHS)
    macro = [metrics[name]["all_records"]["noisy_macro_f1"] for name in strengths]
    flips = [metrics[name]["all_records"]["prediction_flip_rate"] for name in strengths]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    axes[0].bar(strengths, macro, color="#4C78A8")
    axes[0].set(title="Macro-F1", ylabel="Score", ylim=(0, max(macro) + 0.08))
    axes[1].bar(strengths, flips, color="#E45756")
    axes[1].set(title="Prediction flips", ylabel="Rate", ylim=(0, max(flips + [0.01]) + 0.02))
    fig.suptitle("Phase 2J conservative claim paraphrasing")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run_phase2j(
    project_root: Path,
    run_id: str | None = None,
    *,
    expected_source_sha256: str = SPLITS["train"].sha256,
) -> Path:
    """Fit the frozen verifier and evaluate conservative claim substitutions."""
    root = Path(project_root).resolve()
    config_path = root / "config/paraphrase_stress.yaml"
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
    y_true = [str(row["label"]) for row in validation_records]
    model = build_phase2e_pipeline(
        config, "text_plus_disagreement", float(config["regularization_c"])
    )
    model.fit(train_records, [str(row["label"]) for row in train_records])
    predictions, masks, rows, transformations = {}, {}, [], []
    for strength in STRENGTHS:
        changed = [paraphrase_record(row, strength) for row in validation_records]
        mask = np.asarray(
            [a["claim"] != b["claim"] for a, b in zip(validation_records, changed, strict=True)]
        )
        predicted, confidence = _predict(model, changed)
        predictions[strength] = {"predicted": predicted, "confidence": confidence}
        masks[strength] = mask
        for position, upstream in enumerate(validation_indices):
            rows.append(
                {
                    "upstream_index": upstream,
                    "strength": strength,
                    "true_label": y_true[position],
                    "predicted_label": predicted[position],
                    "confidence": float(confidence[position]),
                    "changed": bool(mask[position]),
                }
            )
            transformations.append(
                {
                    "upstream_index": upstream,
                    "strength": strength,
                    "original_claim": validation_records[position]["claim"],
                    "transformed_claim": changed[position]["claim"],
                    "changed": bool(mask[position]),
                }
            )
    metrics, intervals = _summarize(
        y_true, predictions, masks, int(config["bootstrap_repetitions"]), int(config["random_seed"])
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
    prediction_path, transform_path = (
        run_dir / "paraphrase_predictions.jsonl",
        run_dir / "claim_transformations.jsonl",
    )
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    transform_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transformations), encoding="utf-8"
    )
    metrics_path, interval_path = (
        run_dir / "paraphrase_metrics.json",
        run_dir / "bootstrap_intervals.json",
    )
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(interval_path, intervals)
    figure_path = _save_figures(run_dir, metrics)
    outputs = [prediction_path, transform_path, metrics_path, interval_path, figure_path]
    manifest = {
        "protocol_version": str(config["protocol_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "run_id": identifier,
        "random_seed": int(config["random_seed"]),
        "strengths": list(STRENGTHS),
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
            "paraphrase_stress.yaml": {
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


def validate_phase2j_run(run_dir: Path) -> list[str]:
    """Validate Phase 2J hashes and recompute metrics from predictions."""
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
            for line in (directory / "paraphrase_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        predictions, masks = {}, {}
        for strength in STRENGTHS:
            subset = [row for row in rows if row["strength"] == strength]
            predictions[strength] = {
                "predicted": [row["predicted_label"] for row in subset],
                "confidence": np.asarray([row["confidence"] for row in subset]),
            }
            masks[strength] = np.asarray([row["changed"] for row in subset])
        base = [row for row in rows if row["strength"] == "none"]
        y_true = [row["true_label"] for row in base]
        config = yaml.safe_load(
            (root / "config/paraphrase_stress.yaml").read_text(encoding="utf-8")
        )
        metrics, intervals = _summarize(
            y_true,
            predictions,
            masks,
            int(config["bootstrap_repetitions"]),
            int(config["random_seed"]),
        )
        if _load_json(directory / "paraphrase_metrics.json") != metrics:
            errors.append("recorded paraphrase metrics do not match predictions")
        if _load_json(directory / "bootstrap_intervals.json") != intervals:
            errors.append("recorded bootstrap intervals do not match predictions")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return errors
