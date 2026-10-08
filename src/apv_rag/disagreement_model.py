"""Label-free evidence-disagreement features for Phase 2E."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256 as hashlib_sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

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

DISAGREEMENT_FEATURES = (
    "question_count",
    "answer_count",
    "answers_per_question",
    "multi_answer_question_fraction",
    "unique_answer_fraction",
    "boolean_answer_fraction",
    "boolean_yes_no_mixed",
    "negation_fraction",
    "negation_mixed",
    "source_medium_diversity",
    "answer_type_diversity",
    "mean_pairwise_answer_jaccard_distance",
)

_TOKEN = re.compile(r"[\w']+", re.UNICODE)
_NEGATIONS = {"no", "not", "never", "none", "neither", "nor", "without", "n't"}


class DisagreementTransformer(BaseEstimator, TransformerMixin):
    """Convert AVeriTeC records to the fixed structural feature matrix."""

    def fit(self, records: Sequence[Mapping[str, Any]], y: object = None):
        return self

    def transform(self, records: Sequence[Mapping[str, Any]]):
        return sparse.csr_matrix(
            np.vstack([extract_disagreement_features(record) for record in records])
        )


def _tokens(text: object) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(str(text))}


def extract_disagreement_features(record: Mapping[str, Any]) -> np.ndarray:
    """Extract structural features without using labels or provenance URLs."""

    questions = record.get("questions") or []
    answer_groups = [question.get("answers") or [] for question in questions]
    answers = [answer for group in answer_groups for answer in group]
    texts = [str(answer.get("answer") or "").strip() for answer in answers]
    count = len(answers)
    question_count = len(questions)
    normalized = [" ".join(sorted(_tokens(text))) for text in texts if text]
    boolean = [
        text.casefold()
        for text, answer in zip(texts, answers, strict=True)
        if str(answer.get("answer_type") or "").casefold() == "boolean"
    ]
    yes_no = {text.strip(" .!?") for text in boolean} & {"yes", "no"}
    negated = [bool(_tokens(text) & _NEGATIONS) for text in texts]
    media = {
        str(answer.get("source_medium") or "").strip().casefold()
        for answer in answers
        if str(answer.get("source_medium") or "").strip()
    }
    answer_types = {
        str(answer.get("answer_type") or "").strip().casefold()
        for answer in answers
        if str(answer.get("answer_type") or "").strip()
    }
    distances = []
    token_sets = [_tokens(text) for text in texts if text]
    for left_index, left in enumerate(token_sets):
        for right in token_sets[left_index + 1 :]:
            union = left | right
            distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    values = (
        float(question_count),
        float(count),
        count / question_count if question_count else 0.0,
        sum(len(group) > 1 for group in answer_groups) / question_count if question_count else 0.0,
        len(set(normalized)) / count if count else 0.0,
        len(boolean) / count if count else 0.0,
        float(yes_no == {"yes", "no"}),
        sum(negated) / count if count else 0.0,
        float(bool(negated) and any(negated) and not all(negated)),
        len(media) / count if count else 0.0,
        len(answer_types) / count if count else 0.0,
        float(np.mean(distances)) if distances else 0.0,
    )
    return np.asarray(values, dtype=float)


def _render_records(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return [render_claim_plus_evidence(record) for record in records]


def _tfidf(config: Mapping[str, Any], analyzer: str) -> TfidfVectorizer:
    settings = config[f"{analyzer}_tfidf"]
    return TfidfVectorizer(
        analyzer="word" if analyzer == "word" else "char_wb",
        ngram_range=tuple(int(value) for value in settings["ngram_range"]),
        min_df=int(settings["min_df"]),
        max_features=int(settings["max_features"]),
        sublinear_tf=True,
    )


def build_phase2e_pipeline(
    config: Mapping[str, Any], representation: str, c_value: float
) -> Pipeline:
    """Build a calibrated Phase 2E ablation model over complete records."""

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
    branches = {
        "text_only": [("text", text)],
        "disagreement_only": [("disagreement", disagreement)],
        "text_plus_disagreement": [("text", text), ("disagreement", disagreement)],
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


def select_phase2e_setting(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen Macro-F1 and conflict-class selection rule."""

    if not candidates:
        raise ValueError("no Phase 2E candidates are available")
    conflict = LABELS[3]
    order = {
        "disagreement_only": 0,
        "text_only": 1,
        "text_plus_disagreement": 2,
    }
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


def _normalized_split_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    normalized = json.loads(json.dumps(manifest))
    source = normalized.get("source", {})
    if isinstance(source, dict) and "path" in source:
        source["path"] = str(source["path"]).replace("\\", "/")
    return _normalized_json_sha256(normalized)


def _normalized_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib_sha256(payload).hexdigest()


def _fit_candidate(train_records, validation_records, representation, c_value, config):
    y_train = [str(row["label"]) for row in train_records]
    y_true = [str(row["label"]) for row in validation_records]
    if (set(y_train) | set(y_true)) - set(LABELS):
        raise ValueError("Phase 2E data contain an unknown label")
    if min(Counter(y_train).values()) < int(config["calibration_cv"]):
        raise ValueError("each training class needs at least calibration_cv records")
    model = build_phase2e_pipeline(config, representation, c_value)
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
        name
        for name in ("text_only", "disagreement_only", "text_plus_disagreement")
        if name in best
    ]
    macro = [best[name]["metrics"]["macro_f1"] for name in names]
    conflict = [best[name]["metrics"]["per_class"][LABELS[3]]["f1"] for name in names]
    path = run_dir / "ablation_comparison.png"
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.bar(x - 0.18, macro, 0.36, label="Macro-F1", color="#4C78A8")
    axis.bar(x + 0.18, conflict, 0.36, label="Conflict F1", color="#E45756")
    axis.set_xticks(x, [name.replace("_", "\n") for name in names])
    axis.set_ylim(0, max(0.65, max(macro + conflict) + 0.08))
    axis.set_title("Phase 2E disagreement-feature ablation")
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
    axis.set_title("Selected Phase 2E model")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=200)
    plt.close(fig)
    return path, matrix_path


def run_phase2e(
    project_root: Path,
    run_id: str | None = None,
    *,
    expected_source_sha256: str = SPLITS["train"].sha256,
) -> Path:
    """Fit frozen Phase 2E ablations without reading the official dev set."""

    root = Path(project_root).resolve()
    config_path = root / "config/disagreement_model.yaml"
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
    selected = select_phase2e_setting(candidates)
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
            "features": list(DISAGREEMENT_FEATURES),
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
            "disagreement_model.yaml": {
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


def validate_phase2e_run(run_dir: Path) -> list[str]:
    """Validate Phase 2E hashes, alignment, probabilities, and recomputed metrics."""

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
            if name == "split_manifest.json" and entry.get("normalized_sha256"):
                try:
                    normalized_sha = _normalized_split_manifest_sha256(_load_json(path))
                except ValueError:
                    normalized_sha = ""
                if normalized_sha != entry["normalized_sha256"]:
                    errors.append(f"input SHA-256 mismatch: {name}")
            elif entry.get("normalized_sha256"):
                try:
                    normalized_sha = _normalized_json_sha256(_load_json(path))
                except ValueError:
                    normalized_sha = ""
                if normalized_sha != entry["normalized_sha256"]:
                    errors.append(f"input SHA-256 mismatch: {name}")
            else:
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
        validation_entry = manifest["inputs"]["validation_indices.json"]
        validation_path = str(validation_entry["path"]).replace("\\", "/")
        if validation_path.startswith("artifact:"):
            indices_path = directory / validation_path.removeprefix("artifact:")
        else:
            indices_path = root / validation_path
        indices = _load_json(indices_path)
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
    if predictions and not any("probabilit" in error for error in errors):
        config = yaml.safe_load(
            (root / "config/disagreement_model.yaml").read_text(encoding="utf-8")
        )
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
