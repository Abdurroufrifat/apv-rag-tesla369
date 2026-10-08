"""Exploratory training augmentation; no evaluation labels enter fitting."""
import copy

import numpy as np
from sklearn.linear_model import LogisticRegression

from apv_rag.provenance_model import build_phase2f_pipeline
from apv_rag.repetition_stress import inject_derivative_copies


def training_variants(records, counts=(0, 5, 10)):
    if not counts or counts[0] != 0 or len(set(counts)) != len(counts) or any(
            type(k) is not int or k < 0 for k in counts):
        raise ValueError("Unique nonnegative counts starting at zero required")
    variants, labels, weights, parents = [], [], [], []
    for parent, record in enumerate(records):
        for k in counts:
            variant = inject_derivative_copies(record, k)
            label = variant.pop("label")
            variant.pop("justification", None)
            variants.append(variant)
            labels.append(label)
            weights.append(1 / len(counts))
            parents.append(parent)
    return variants, labels, np.asarray(weights), parents


def build_model(config):
    pipeline = build_phase2f_pipeline(config, "phase2e", 1.0)
    # A matched uncalibrated control keeps this training comparison distinct
    # from the earlier calibrated Phase 2F result.
    pipeline.set_params(classifier=LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2500, random_state=369))
    return pipeline


def model_records(records):
    result = copy.deepcopy(records)
    for row in result:
        row.pop("label", None)
        row.pop("justification", None)
    return result
