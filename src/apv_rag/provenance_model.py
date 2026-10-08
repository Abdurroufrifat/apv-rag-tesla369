"""Source-family and duplicate-evidence features for Phase 2F."""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import matplotlib
import numpy as np
import yaml
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from apv_rag.averitec import DATASET_DIR_NAME, SPLITS
from apv_rag.disagreement_model import (
    DisagreementTransformer,
    _load_json,
    _normalized_json_sha256,
    _normalized_split_manifest_sha256,
    _render_records,
    _tfidf,
)
from apv_rag.evidence_baseline import LABELS, assert_prediction_alignment, load_split_records
from apv_rag.metrics import classification_metrics
from apv_rag.splits import sha256, write_json_atomic

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PROVENANCE_FEATURES = (
    "source_count",
    "family_count",
    "family_diversity",
    "largest_family_fraction",
    "repeated_family_fraction",
    "duplicate_url_fraction",
    "duplicate_answer_fraction",
    "archive_fraction",
    "missing_url_fraction",
    "family_entropy",
)

_ARCHIVE = re.compile(r"^https?://web\.archive\.org/web/[^/]+/(https?://.+)$", re.I)
_TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_ANSWER_TOKEN = re.compile(r"\w+", re.UNICODE)


class ProvenanceTransformer(BaseEstimator, TransformerMixin):
    """Convert AVeriTeC records to provenance feature rows."""

    def fit(self, records: Sequence[Mapping[str, Any]], y: object = None):
        return self

    def transform(self, records: Sequence[Mapping[str, Any]]):
        return sparse.csr_matrix(
            np.vstack([extract_provenance_features(record) for record in records])
        )


def normalize_source_url(value: object) -> str:
    """Return a stable host/path/query key, unwrapping Wayback URLs."""

    text = str(value or "").strip()
    match = _ARCHIVE.match(text)
    if match:
        text = match.group(1)
    if not text:
        return ""
    parsed = urlsplit(text if "://" in text else f"https://{text}")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or "/"
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING
    ]
    result = f"{host}{path}"
    if query:
        result += "?" + urlencode(sorted(query))
    return result


def _answer_key(value: object) -> str:
    return " ".join(token.casefold() for token in _ANSWER_TOKEN.findall(str(value or "")))


def extract_provenance_features(record: Mapping[str, Any]) -> np.ndarray:
    """Extract label-free source-family and duplicate-evidence features."""

    answers = [
        answer
        for question in record.get("questions") or []
        for answer in question.get("answers") or []
    ]
    count = len(answers)
    urls = [normalize_source_url(answer.get("source_url")) for answer in answers]
    families = [urlsplit(f"https://{url}").hostname or "" for url in urls if url]
    family_counts = Counter(families)
    unique_families = len(family_counts)
    repeated_sources = sum(amount for amount in family_counts.values() if amount > 1)
    answer_keys = [_answer_key(answer.get("answer")) for answer in answers]
    nonempty_answers = [value for value in answer_keys if value]
    archive_count = sum(
        "web.archive.org/web/" in str(answer.get("source_url") or "").casefold()
        or bool(answer.get("cached_source_url"))
        for answer in answers
    )
    entropy = 0.0
    if families:
        for amount in family_counts.values():
            probability = amount / len(families)
            entropy -= probability * math.log2(probability)
        if unique_families > 1:
            entropy /= math.log2(unique_families)
    values = (
        float(count),
        float(unique_families),
        unique_families / count if count else 0.0,
        max(family_counts.values(), default=0) / count if count else 0.0,
        repeated_sources / count if count else 0.0,
        (count - len(set(urls))) / count if count else 0.0,
        (len(nonempty_answers) - len(set(nonempty_answers))) / count if count else 0.0,
        archive_count / count if count else 0.0,
        sum(not url for url in urls) / count if count else 0.0,
        entropy,
    )
    return np.asarray(values, dtype=float)


def build_phase2f_pipeline(
    config: Mapping[str, Any], representation: str, c_value: float
) -> Pipeline:
    """Build one calibrated Phase 2F ablation candidate."""

    text = Pipeline(
        [
            ("render", FunctionTransformer(_render_records, validate=False)),
            (
                "tfidf",
                FeatureUnion([("word", _tfidf(config, "word")), ("char", _tfidf(config, "char"))]),
            ),
        ]
    )
    disagreement = Pipeline(
        [
            ("extract", DisagreementTransformer()),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    phase2e = FeatureUnion([("text", text), ("disagreement", disagreement)])
    provenance = Pipeline(
        [
            ("extract", ProvenanceTransformer()),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    branches = {
        "phase2e": [("phase2e", phase2e)],
        "provenance_only": [("provenance", provenance)],
        "phase2e_plus_provenance": [
            ("phase2e", phase2e),
            ("provenance", provenance),
        ],
    }
    if representation not in branches:
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
        cv=StratifiedKFold(n_splits=int(config["calibration_cv"]), shuffle=True, random_state=seed),
    )
    return Pipeline(
        [("features", FeatureUnion(branches[representation])), ("classifier", classifier)]
    )


def select_phase2f_setting(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen Phase 2F selection rule."""

    if not candidates:
        raise ValueError("no Phase 2F candidates are available")
    conflict = LABELS[3]
    order = {"phase2e": 0, "provenance_only": 1, "phase2e_plus_provenance": 2}
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


def _fit_candidate(train_records, validation_records, representation, c_value, config):
    y_train = [str(row["label"]) for row in train_records]
    y_true = [str(row["label"]) for row in validation_records]
    if (set(y_train) | set(y_true)) - set(LABELS):
        raise ValueError("Phase 2F data contain an unknown label")
    if min(Counter(y_train).values()) < int(config["calibration_cv"]):
        raise ValueError("each training class needs at least calibration_cv records")
    model = build_phase2f_pipeline(config, representation, c_value)
    model.fit(train_records, y_train)
    raw = model.predict_proba(validation_records)
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
                label: float(probabilities[row, column]) for column, label in enumerate(LABELS)
            },
            "confidence": float(probabilities[row].max()),
        }
        for row, (truth, predicted) in enumerate(zip(y_true, predictions, strict=True))
    ]
    return {"representation": representation, "c": float(c_value), "metrics": metrics}, details


def _save_figures(run_dir: Path, candidates, predictions) -> tuple[Path, Path]:
    best = {}
    for row in candidates:
        name = row["representation"]
        if name not in best or row["metrics"]["macro_f1"] > best[name]["metrics"]["macro_f1"]:
            best[name] = row
    names = [
        name for name in ("phase2e", "provenance_only", "phase2e_plus_provenance") if name in best
    ]
    macro = [best[name]["metrics"]["macro_f1"] for name in names]
    conflict = [best[name]["metrics"]["per_class"][LABELS[3]]["f1"] for name in names]
    path = run_dir / "provenance_ablation.png"
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(x - 0.18, macro, 0.36, label="Macro-F1", color="#4C78A8")
    axis.bar(x + 0.18, conflict, 0.36, label="Conflict F1", color="#E45756")
    axis.set_xticks(x, [name.replace("_", "\n") for name in names])
    axis.set_ylim(0, max(0.65, max(macro + conflict) + 0.08))
    axis.set_title("Phase 2F provenance-family ablation")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    matrix = confusion_matrix(
        [row["true_label"] for row in predictions],
        [row["predicted_label"] for row in predictions],
        labels=list(LABELS),
    )
    matrix_path = run_dir / "confusion_matrix.png"
    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(4):
        for column in range(4):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    short = ["Supported", "Refuted", "NEI", "Conflict"]
    axis.set_xticks(range(4), short, rotation=25, ha="right")
    axis.set_yticks(range(4), short)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Selected Phase 2F model")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=200)
    plt.close(fig)
    return path, matrix_path


def run_phase2f(
    project_root: Path,
    run_id: str | None = None,
    *,
    expected_source_sha256: str = SPLITS["train"].sha256,
) -> Path:
    """Fit Phase 2F candidates using only the frozen Phase 2B split."""

    root = Path(project_root).resolve()
    config_path = root / "config/provenance_model.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = root / "data/external/averitec" / DATASET_DIR_NAME / "train.json"
    if sha256(source_path) != expected_source_sha256:
        raise ValueError("train.json SHA-256 does not match the expected source")
    split_dir = root / "data/processed/averitec/phase2b_split_v0_1"
    split_manifest_path = split_dir / "split_manifest.json"
    split_manifest = _load_json(split_manifest_path)
    relative_source = str(source_path.relative_to(root)).replace("\\", "/")
    manifest_source = str(split_manifest.get("source", {}).get("path", "")).replace("\\", "/")
    if manifest_source != relative_source:
        raise ValueError("split manifest source does not match train.json")
    records = _load_json(source_path)
    train_indices = _load_json(split_dir / "train_indices.json")
    validation_indices = _load_json(split_dir / "validation_indices.json")
    if set(train_indices) & set(validation_indices):
        raise ValueError("training and validation indices overlap")
    train_records = load_split_records(records, train_indices)
    validation_records = load_split_records(records, validation_indices)
    identifier = run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = root / str(config["output_directory"]) / identifier
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    candidates, prediction_sets = [], {}
    for representation in config["representations"]:
        for c_value in config["regularization_c"]:
            result, details = _fit_candidate(
                train_records, validation_records, str(representation), float(c_value), config
            )
            candidates.append(result)
            prediction_sets[(str(representation), float(c_value))] = details
    selected = select_phase2f_setting(candidates)
    predictions = prediction_sets[(selected["representation"], float(selected["c"]))]
    for index, row in zip(validation_indices, predictions, strict=True):
        row["upstream_index"] = index
    assert_prediction_alignment(validation_indices, predictions)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "input_snapshot"
    snapshot_dir.mkdir()
    for source in (
        split_manifest_path,
        split_dir / "train_indices.json",
        split_dir / "validation_indices.json",
    ):
        shutil.copy2(source, snapshot_dir / source.name)
    prediction_path = run_dir / "validation_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.json"
    selection_path = run_dir / "model_selection.json"
    write_json_atomic(metrics_path, selected["metrics"])
    write_json_atomic(
        selection_path,
        {
            "selection_rule": "macro_f1, conflict_f1, balanced_accuracy, lower_c",
            "selected_representation": selected["representation"],
            "selected_c": selected["c"],
            "provenance_features": list(PROVENANCE_FEATURES),
            "candidates": candidates,
        },
    )
    comparison_path, matrix_path = _save_figures(run_dir, candidates, predictions)
    outputs = [prediction_path, metrics_path, selection_path, comparison_path, matrix_path]
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
                "path": "artifact:input_snapshot/split_manifest.json",
                "sha256": sha256(snapshot_dir / "split_manifest.json"),
                "normalized_sha256": _normalized_split_manifest_sha256(split_manifest),
            },
            "train_indices.json": {
                "path": "artifact:input_snapshot/train_indices.json",
                "sha256": sha256(snapshot_dir / "train_indices.json"),
                "normalized_sha256": _normalized_json_sha256(train_indices),
            },
            "validation_indices.json": {
                "path": "artifact:input_snapshot/validation_indices.json",
                "sha256": sha256(snapshot_dir / "validation_indices.json"),
                "normalized_sha256": _normalized_json_sha256(validation_indices),
            },
            "provenance_model.yaml": {
                "path": str(config_path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256(config_path),
            },
        },
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs
        },
        "packages": {
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "scikit-learn": version("scikit-learn"),
            "matplotlib": version("matplotlib"),
            "PyYAML": version("PyYAML"),
        },
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    return run_dir


def validate_phase2f_run(run_dir: Path) -> list[str]:
    """Validate Phase 2F inputs, outputs, alignment, and recorded metrics."""

    directory = Path(run_dir).resolve()
    errors: list[str] = []
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
        relative = str(entry.get("path", "")).replace("\\", "/")
        if Path(relative).name == "dev.json":
            errors.append("manifest lists dev.json as a model input")
        if relative.startswith("artifact:"):
            path = (directory / relative.removeprefix("artifact:")).resolve()
            valid_parent = directory
        else:
            path = (root / relative).resolve()
            valid_parent = root
        if valid_parent not in path.parents or not path.is_file():
            errors.append(f"invalid input path: {name}")
        elif sha256(path) != entry.get("sha256"):
            errors.append(f"input SHA-256 mismatch: {name}")
    for name, entry in manifest.get("outputs", {}).items():
        path = directory / name
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            errors.append(f"output integrity failure: {name}")
    try:
        predictions = [
            json.loads(line)
            for line in (directory / "validation_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        validation_path = str(manifest["inputs"]["validation_indices.json"]["path"])
        indices = _load_json(directory / validation_path.removeprefix("artifact:"))
        assert_prediction_alignment(indices, predictions)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
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
    if predictions and not any("probabilit" in error for error in errors):
        config = yaml.safe_load((root / "config/provenance_model.yaml").read_text(encoding="utf-8"))
        probabilities = np.asarray(
            [[row["probabilities"][label] for label in LABELS] for row in predictions]
        )
        recomputed = classification_metrics(
            [row["true_label"] for row in predictions],
            [row["predicted_label"] for row in predictions],
            probabilities,
            LABELS,
            ece_bins=int(config["ece_bins"]),
            coverages=[float(value) for value in config["target_coverages"]],
        )
        if _load_json(directory / "metrics.json") != recomputed:
            errors.append("recorded metrics do not match predictions")
    return errors
